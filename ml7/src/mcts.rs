// ASTRA-7 AZ-league MCTS (ASTRA7.md §4.3): PUCT search over concrete action
// tokens, generic over which of the five decision kinds is being searched.
//
// Tree shape: EXACTLY one level deep (root decision -> candidate action ->
// leaf), not a classical multi-ply game tree. This is a deliberate reading
// of §4.3's "leaf = one real engine playout of the current turn (not a
// 20-turn rollout)" - the "not a rollout" instruction is about how a LEAF is
// scored, and there is no engine-level way to treat a FUTURE decision point
// (an opponent's next turn, or a later action this same turn) as a re-
// enterable tree node the way classical AlphaZero MCTS does for e.g. chess:
// this engine's `Player` callback structure only exposes two genuinely
// resumable simulation entry points - `GameEngine::from_state(..).run()`
// (restart search from the top of a turn, capped via `set_max_turns`, the
// technique `engine/src/cpu.rs`'s `ExpertCpuPlayer`/v6's `value_lookahead`
// already use) and `GameEngine::simulate_finish_chain` (resume a chain
// response window from its exact `ChainResumeCursor`). Both are one-shot:
// neither lets a caller pause again mid-continuation to expand a deeper
// node. So instead of a tree, `sims >= 400` PUCT visits are spent choosing
// WHICH candidate at the CURRENT decision to re-sample next (favoring
// promising ones per the running Q + prior, exactly like a contextual
// bandit) - each visit is one full "force this candidate, let AdvCPU play
// out the rest of the turn/chain, score the result with the value head"
// evaluation. This is a strict superset of `ExpertCpuPlayer`'s technique
// (which spends a FIXED sample count per candidate); PUCT instead spends the
// budget where the running estimates say it matters, which is what makes
// >=400 total sims (vs. ExpertCpu's ~2-per-candidate) worthwhile at all.
//
// Decision-kind coverage: `search_choose_action`/`search_chain_response`
// cover the two kinds with a resumable continuation. `decide`/`pick_targets`/
// `choose_discards` do NOT have one - they're reached from deep inside
// `execute_action`'s effect-resolution control flow, which isn't
// reconstructable from a `GameState` snapshot alone (unlike chain response,
// which the engine deliberately made resumable via `ChainResumeCursor`
// specifically because it mattered this much - ASTRA7.md §1.1). AZ's
// self-play driver (az.rs) uses the net's raw policy (no search) for those
// three kinds; per §1.1 this is a low-stakes simplification, not a gap in
// the thing that was actually measured to matter (chain-response timing).

use std::sync::atomic::{AtomicBool, Ordering};

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

use burn::tensor::backend::Backend;

use novacana_engine::card::{Card, CardId};
use novacana_engine::chain::{Chain, ChainResumeCursor};
use novacana_engine::cpu::AdvancedCpuPlayer;
use novacana_engine::game::{GameEngine, GameState};
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption};

use crate::batch_infer::BatchedNet;
use crate::tokens::{self, ACTION_MAX};

pub struct McConfig {
    pub sims: usize,
    pub c_puct: f32,
    pub dirichlet_eps: f32,
}

impl Default for McConfig {
    // ASTRA7.md §4.3 defaults. dirichlet_eps=0 is the right choice for eval
    // play (no exploration noise); self-play (az.rs) overrides it to 0.25.
    fn default() -> Self {
        Self { sims: 400, c_puct: 1.5, dirichlet_eps: 0.0 }
    }
}

/// One decision's search result: the training-target visit distribution
/// (raw N/sum(N) over ACTION_MAX slots, 0 past the real legal-action count -
/// same Pad convention `action_valid_mask` uses) and the index actually
/// played (temperature-sampled from visits for exploration, or greedy).
pub struct McOutcome {
    pub visit_dist: [f32; ACTION_MAX],
    pub chosen: usize,
}

// Same role as `SIM_MAX_DECISIONS` in engine/src/cpu.rs - bounds a single
// leaf-eval sub-simulation's decision cost so a pathological chain-response
// window can't blow up one PUCT visit. Not exported: purely an internal
// safety net, no caller should ever need to tune it.
const LEAF_MAX_DECISIONS: usize = 3000;

// ── Forced-once players (leaf simulation) ──────────────────────────────────
//
// Mirrors `ForcedThenAdv`/`ForcedChainThenAdv` in engine/src/cpu.rs exactly
// (force one specific decision, AdvCPU for everything else in the
// simulation) - proven pattern, not reinvented. Index-based rather than
// Action-equality-based: the candidate index was computed from a
// `compute_legal_actions`/`compute_chain_responses` call on the SAME cloned
// state, and both are pure functions of state (asserted by
// `adv_cpu_decision_is_pure_given_identical_state` in engine/src/game.rs),
// so re-deriving the legal list inside the fresh simulated engine reproduces
// the identical ordering - no need to match by value.

struct ForcedActionOnce {
    forced_index: usize,
    played: AtomicBool,
    adv: AdvancedCpuPlayer,
}

impl Player for ForcedActionOnce {
    fn name(&self) -> &str {
        "mcts-forced-action"
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        if !self.played.swap(true, Ordering::Relaxed) {
            return self.forced_index.min(legal.len().saturating_sub(1));
        }
        self.adv.choose_action(view, legal)
    }
    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &novacana_engine::chain::ChainView,
        legal: &[ChainAction],
    ) -> Option<usize> {
        self.adv.respond_to_chain(view, chain, legal)
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.adv.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
        self.adv.pick_targets(view, p, opts, c)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.adv.choose_discards(view, hand, n)
    }
}

struct ForcedChainOnce {
    forced: Option<usize>,
    played: AtomicBool,
    adv: AdvancedCpuPlayer,
}

impl Player for ForcedChainOnce {
    fn name(&self) -> &str {
        "mcts-forced-chain"
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.adv.choose_action(view, legal)
    }
    fn respond_to_chain(
        &self,
        view: &GameView,
        chain: &novacana_engine::chain::ChainView,
        legal: &[ChainAction],
    ) -> Option<usize> {
        if !self.played.swap(true, Ordering::Relaxed) {
            return self.forced;
        }
        self.adv.respond_to_chain(view, chain, legal)
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.adv.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
        self.adv.pick_targets(view, p, opts, c)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.adv.choose_discards(view, hand, n)
    }
}

// ── Value readout at an arbitrary reached state ─────────────────────────
//
// The successor state a leaf-eval lands on isn't necessarily "my_idx's
// current decision" (the turn may have rolled over to another seat) - there
// is no live legal-action list inherently attached to it the way a real BC/
// TD-value sample has. Since state tokens attend to action tokens (self-
// attention over one sequence, tokens.rs §3.1), the action block isn't
// value-irrelevant noise, so this uses `compute_legal_actions(my_idx)` as a
// best-effort "what my_idx could hypothetically play right now" stand-in
// (falling back to a single dummy Pass token if that's empty, e.g. the
// successor state has my_idx's action phase already skipped this round) -
// this keeps the value forward pass shaped like every other sample the net
// was trained on instead of inventing a new zero-action-token convention.
fn value_at_state<B: Backend>(bnet: &BatchedNet<B>, engine: &GameEngine, my_idx: usize) -> f32
where
    B::Device: Clone + Send + Sync + 'static,
{
    let view = engine.build_view(my_idx);
    let legal = engine.compute_legal_actions(my_idx);
    let action_tokens = if legal.is_empty() {
        tokens::encode_action_tokens_choose_action(&view, std::slice::from_ref(&Action::Pass))
    } else {
        tokens::encode_action_tokens_choose_action(&view, &legal)
    };
    match tokens::assemble(&view, &action_tokens) {
        Some(enc) => bnet.forward_single(&enc.tokens).1,
        None => 0.5,
    }
}

/// Leaf evaluation for a `choose_action` candidate: force `forced_index`,
/// resolve the REST of the current turn via AdvCPU (follow-up actions,
/// chain reactions, decides, discards), cap the simulation to exactly this
/// turn (`set_max_turns(state.turn_number)` - after this turn's number
/// increments past the cap, `run()` stops), then read the value head off
/// the successor. A natural game end inside the turn returns the true
/// outcome instead of a value-head guess. Mirrors `expert_sim_action` in
/// engine/src/cpu.rs, but capped to ONE turn (ExpertCpu deliberately does a
/// full round - different job, ranking against a static heuristic that
/// doesn't need many samples; MCTS gets its robustness from repeated PUCT
/// visits instead).
fn leaf_value_choose_action<B: Backend>(
    bnet: &BatchedNet<B>,
    state: &GameState,
    my_idx: usize,
    forced_index: usize,
    num_players: usize,
) -> f32
where
    B::Device: Clone + Send + Sync + 'static,
{
    let players: Vec<Box<dyn Player>> = (0..num_players)
        .map(|i| -> Box<dyn Player> {
            if i == my_idx {
                Box::new(ForcedActionOnce {
                    forced_index,
                    played: AtomicBool::new(false),
                    adv: AdvancedCpuPlayer::new(format!("mf{i}")),
                })
            } else {
                Box::new(AdvancedCpuPlayer::new(format!("mc{i}")))
            }
        })
        .collect();

    let cap = state.turn_number;
    let mut engine = GameEngine::from_state(state.clone(), players);
    engine.set_silent();
    engine.set_max_turns(cap);
    engine.set_max_decisions(LEAF_MAX_DECISIONS);
    let result = engine.run();

    if result.turns_elapsed <= cap {
        return if result.winner_idx == my_idx { 1.0 } else { 0.0 };
    }
    value_at_state(bnet, &engine, my_idx)
}

/// Leaf evaluation for a `respond_to_chain` candidate (`forced = None` is
/// "pass"): force the response, resume the response window from the exact
/// live `cursor`, resolve the chain. Mirrors `simulate_chain_response` in
/// engine/src/cpu.rs exactly.
#[allow(clippy::too_many_arguments)]
fn leaf_value_chain_response<B: Backend>(
    bnet: &BatchedNet<B>,
    state: &GameState,
    chain: &Chain,
    my_idx: usize,
    cursor: ChainResumeCursor,
    forced: Option<usize>,
    num_players: usize,
) -> f32
where
    B::Device: Clone + Send + Sync + 'static,
{
    let players: Vec<Box<dyn Player>> = (0..num_players)
        .map(|i| -> Box<dyn Player> {
            if i == my_idx {
                Box::new(ForcedChainOnce {
                    forced,
                    played: AtomicBool::new(false),
                    adv: AdvancedCpuPlayer::new(format!("mcf{i}")),
                })
            } else {
                Box::new(AdvancedCpuPlayer::new(format!("mcc{i}")))
            }
        })
        .collect();

    let mut engine = GameEngine::from_state(state.clone(), players);
    engine.set_silent();
    engine.set_max_decisions(LEAF_MAX_DECISIONS);
    match engine.simulate_finish_chain(chain.clone(), cursor) {
        Some(result) => {
            if result.winner_idx == my_idx {
                1.0
            } else {
                0.0
            }
        }
        None => value_at_state(bnet, &engine, my_idx),
    }
}

// ── PUCT core ──────────────────────────────────────────────────────────────

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum::<f32>().max(1e-9);
    exps.iter().map(|&x| x / s).collect()
}

fn root_priors<B: Backend>(bnet: &BatchedNet<B>, view: &GameView, action_tokens: &[tokens::Token], n: usize) -> Vec<f32>
where
    B::Device: Clone + Send + Sync + 'static,
{
    match tokens::assemble(view, action_tokens) {
        Some(enc) => {
            let (masked, _v) = bnet.forward_single(&enc.tokens);
            softmax_slice(&masked[..n])
        }
        None => vec![1.0 / n as f32; n],
    }
}

// Standard normal via Box-Muller, and a Marsaglia-Tsang gamma sampler
// boosted for alpha<1 - hand-rolled instead of pulling in `rand_distr`
// (not already a workspace dependency, and the Dirichlet root-noise use
// case here only needs one distribution).
fn sample_standard_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-7);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn sample_gamma(alpha: f32, rng: &mut impl Rng) -> f32 {
    if alpha < 1.0 {
        let u: f32 = rng.gen::<f32>().max(1e-7);
        return sample_gamma(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, mut v);
        loop {
            let xc = sample_standard_normal(rng);
            let vc = 1.0 + c * xc;
            if vc > 0.0 {
                x = xc;
                v = vc;
                break;
            }
        }
        v = v * v * v;
        let u: f32 = rng.gen::<f32>();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn sample_dirichlet(alpha: f32, n: usize, rng: &mut impl Rng) -> Vec<f32> {
    let g: Vec<f32> = (0..n).map(|_| sample_gamma(alpha, rng).max(1e-6)).collect();
    let s: f32 = g.iter().sum();
    g.iter().map(|&x| x / s).collect()
}

/// Runs `cfg.sims` PUCT visits over `n = priors.len()` candidates, applying
/// root Dirichlet noise first (ASTRA7.md §4.3: eps=0.25, alpha=10/n_legal -
/// the "washed out by sims>=400" assumption is §8.5's escape-hatch-guarded
/// hypothesis, not proven here; this function just implements the formula).
/// Returns raw visit counts per candidate.
fn run_puct<F: Fn(usize) -> f32>(priors: &[f32], cfg: &McConfig, rng: &mut impl Rng, leaf_eval: F) -> Vec<u32> {
    let n = priors.len();
    let mut p = priors.to_vec();
    if cfg.dirichlet_eps > 0.0 {
        let alpha = 10.0 / n as f32;
        let noise = sample_dirichlet(alpha, n, rng);
        for i in 0..n {
            p[i] = (1.0 - cfg.dirichlet_eps) * p[i] + cfg.dirichlet_eps * noise[i];
        }
    }

    let mut visits = vec![0u32; n];
    let mut total_value = vec![0f32; n];
    for _ in 0..cfg.sims.max(1) {
        let total_n = visits.iter().sum::<u32>() as f32;
        // (1 + total_n).sqrt() rather than the textbook total_n.sqrt(): at
        // total_n=0 the exploration term would otherwise vanish for every
        // candidate identically (sqrt(0)=0), collapsing the very first
        // selection to a fixed tie-break instead of favoring the highest
        // prior - this +1 is a standard PUCT-implementation fix, not a
        // deviation from ASTRA7.md's formula (unspecified there at that
        // level of detail).
        let explore_scale = (1.0 + total_n).sqrt();
        let i = (0..n)
            .max_by(|&a, &b| {
                let ucb = |j: usize| {
                    let q = if visits[j] > 0 { total_value[j] / visits[j] as f32 } else { 0.0 };
                    q + cfg.c_puct * p[j] * explore_scale / (1.0 + visits[j] as f32)
                };
                ucb(a).partial_cmp(&ucb(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let v = leaf_eval(i);
        visits[i] += 1;
        total_value[i] += v;
    }
    visits
}

/// Builds the training-target visit distribution + the actually-played
/// index. `move_temperature` <= ~1e-3 means greedy (argmax visits) - used
/// past ASTRA7.md §4.3's first-15-decisions exploration window; otherwise
/// sampled from N^(1/T)/sum(N^(1/T)), the standard AlphaZero move-selection
/// temperature (distinct from the visit_dist training target itself, which
/// is always the raw, untempered N/sum(N) - only the ACTED move uses T).
fn finalize(visits: Vec<u32>, move_temperature: f32, rng: &mut impl Rng) -> McOutcome {
    let n = visits.len();
    let total: u32 = visits.iter().sum();
    let mut visit_dist = [0f32; ACTION_MAX];
    for i in 0..n.min(ACTION_MAX) {
        visit_dist[i] = visits[i] as f32 / total.max(1) as f32;
    }
    let chosen = if move_temperature <= 1e-3 {
        (0..n).max_by_key(|&i| visits[i]).unwrap_or(0)
    } else {
        let weights: Vec<f64> = visits.iter().map(|&v| (v as f64).powf(1.0 / move_temperature as f64)).collect();
        match WeightedIndex::new(&weights) {
            Ok(dist) => dist.sample(rng),
            Err(_) => (0..n).max_by_key(|&i| visits[i]).unwrap_or(0),
        }
    };
    McOutcome { visit_dist, chosen }
}

// ── Public search entry points ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn search_choose_action<B: Backend>(
    bnet: &BatchedNet<B>,
    view: &GameView,
    state: &GameState,
    legal: &[Action],
    cfg: &McConfig,
    move_temperature: f32,
    rng: &mut impl Rng,
) -> McOutcome
where
    B::Device: Clone + Send + Sync + 'static,
{
    let n = legal.len();
    if n <= 1 {
        let mut visit_dist = [0f32; ACTION_MAX];
        if n == 1 {
            visit_dist[0] = 1.0;
        }
        return McOutcome { visit_dist, chosen: 0 };
    }
    let action_tokens = tokens::encode_action_tokens_choose_action(view, legal);
    let priors = root_priors(bnet, view, &action_tokens, n);
    let my_idx = view.my_idx;
    let num_players = state.players.len();
    let visits = run_puct(&priors, cfg, rng, |i| leaf_value_choose_action(bnet, state, my_idx, i, num_players));
    finalize(visits, move_temperature, rng)
}

#[allow(clippy::too_many_arguments)]
pub fn search_chain_response<B: Backend>(
    bnet: &BatchedNet<B>,
    view: &GameView,
    state: &GameState,
    chain: &Chain,
    cursor: ChainResumeCursor,
    legal: &[ChainAction],
    cfg: &McConfig,
    move_temperature: f32,
    rng: &mut impl Rng,
) -> McOutcome
where
    B::Device: Clone + Send + Sync + 'static,
{
    let n = legal.len() + 1; // index 0 = pass, always legal (tokens.rs convention)
    if n <= 1 {
        let mut visit_dist = [0f32; ACTION_MAX];
        visit_dist[0] = 1.0;
        return McOutcome { visit_dist, chosen: 0 };
    }
    let action_tokens = tokens::encode_action_tokens_chain(view, legal);
    let priors = root_priors(bnet, view, &action_tokens, n);
    let my_idx = view.my_idx;
    let num_players = state.players.len();
    let visits = run_puct(&priors, cfg, rng, |i| {
        let forced = if i == 0 { None } else { Some(i - 1) };
        leaf_value_chain_response(bnet, state, chain, my_idx, cursor, forced, num_players)
    });
    finalize(visits, move_temperature, rng)
}

/// Successor value for ONE candidate `choose_action` (no search - a single
/// leaf-eval sample, `dirichlet_eps=0`). Shared by `evals.rs`'s `value-rank`
/// diagnostic (ASTRA7.md §4.2), which needs exactly this "apply this action,
/// score the result" primitive without wrapping it in a PUCT budget.
pub fn successor_value_choose_action<B: Backend>(
    bnet: &BatchedNet<B>,
    state: &GameState,
    my_idx: usize,
    action_index: usize,
    num_players: usize,
) -> f32
where
    B::Device: Clone + Send + Sync + 'static,
{
    leaf_value_choose_action(bnet, state, my_idx, action_index, num_players)
}

#[cfg(test)]
mod tests {
    use super::*;
    use novacana_engine::deck::Deck;

    fn fresh_state(num_players: usize) -> GameState {
        let players: Vec<Box<dyn Player>> =
            (0..num_players).map(|i| -> Box<dyn Player> { Box::new(AdvancedCpuPlayer::new(format!("s{i}"))) }).collect();
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        // Force hands to be dealt without playing a full game: run_start_of_turn
        // + run_draw_phase happen inside run(), so instead just deal directly
        // via a 1-turn cap and take whatever state results - good enough to
        // get a live mid-game GameState to search from.
        engine.set_max_turns(1);
        engine.set_max_decisions(500);
        let _ = engine.run();
        engine.state
    }

    #[test]
    fn search_choose_action_returns_valid_distribution() {
        use burn_ndarray::NdArray;
        use burn_ndarray::NdArrayDevice;
        use std::sync::Arc;

        let state = fresh_state(4);
        let device = NdArrayDevice::Cpu;
        let net = crate::net::Net::<NdArray>::fresh(&device);
        let bnet = BatchedNet::spawn(Arc::new(net), device, 32);
        let mut rng = rand::thread_rng();

        let idx = state.active_player_idx;
        let players: Vec<Box<dyn Player>> =
            (0..4).map(|i| -> Box<dyn Player> { Box::new(AdvancedCpuPlayer::new(format!("v{i}"))) }).collect();
        let engine = GameEngine::from_state(state.clone(), players);
        let legal = engine.compute_legal_actions(idx);
        assert!(!legal.is_empty());
        let view = engine.build_view(idx);

        let cfg = McConfig { sims: 8, c_puct: 1.5, dirichlet_eps: 0.25 };
        let outcome = search_choose_action(&bnet, &view, &state, &legal, &cfg, 1.0, &mut rng);

        let sum: f32 = outcome.visit_dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "visit_dist must sum to ~1, got {sum}");
        assert!(outcome.chosen < legal.len());
        for i in legal.len()..ACTION_MAX {
            assert_eq!(outcome.visit_dist[i], 0.0, "padding slots must stay zero");
        }
    }

    #[test]
    fn dirichlet_samples_are_a_valid_simplex() {
        let mut rng = rand::thread_rng();
        for n in [2usize, 5, 20] {
            let d = sample_dirichlet(10.0 / n as f32, n, &mut rng);
            assert_eq!(d.len(), n);
            let s: f32 = d.iter().sum();
            assert!((s - 1.0).abs() < 1e-3, "dirichlet sample must sum to 1, got {s}");
            assert!(d.iter().all(|&x| x >= 0.0));
        }
    }
}
