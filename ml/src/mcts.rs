// AlphaZero-style PUCT MCTS over the 31-action abstract space.
//
// PUCT: Q(a) + c_puct * P(a) * sqrt(N) / (1 + n(a))
//
// Simulations run for SIM_TURNS turns then use the value head instead of full
// rollouts (see simulate_from_v6 for why pure value-leaf failed). Root
// Dirichlet noise during training searches. Opponent seats use
// AdvancedCpuPlayer to cut network forward passes to only the MCTS player's
// decisions. Sims run in parallel batches.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use novacana_engine::{
    card::{Card, CardId},
    cpu::AdvancedCpuPlayer,
    game::{GameEngine, GameState},
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, ProtostarChoice, TargetOption},
    chain::ChainView,
};

use crate::action::{
    NUM_ACTIONS, abstract_to_action, abstract_to_chain, action_to_abstract,
    chain_legal_mask, legal_mask,
};
use crate::candidate::{CAND_DIM, encode_action_candidate};
use crate::feature::{encode, FEATURE_DIM};
use crate::net::{masked_softmax, Net, Weights, INPUT_DIM};

// ── StateStack (pub so self_play.rs can import it) ────────────────────────────

pub struct StateStack {
    buf:   [[f32; FEATURE_DIM]; 3],
    count: usize,
}

impl StateStack {
    pub fn new() -> Self {
        Self { buf: [[0.0; FEATURE_DIM]; 3], count: 0 }
    }

    pub fn push(&mut self, snap: [f32; FEATURE_DIM]) {
        self.buf[0] = self.buf[1];
        self.buf[1] = self.buf[2];
        self.buf[2] = snap;
        if self.count < 3 { self.count += 1; }
    }

    pub fn stacked(&self) -> [f32; INPUT_DIM] {
        let mut out = [0.0f32; INPUT_DIM];
        out[..FEATURE_DIM].copy_from_slice(&self.buf[0]);
        out[FEATURE_DIM..2 * FEATURE_DIM].copy_from_slice(&self.buf[1]);
        out[2 * FEATURE_DIM..].copy_from_slice(&self.buf[2]);
        out
    }
}

// ── SimPlayerV6 — greedy policy, fresh stack per game ────────────────────────

struct SimPlayerV6 {
    net:   Arc<Net>,
    stack: Mutex<StateStack>,
}

impl SimPlayerV6 {
    fn new(net: Arc<Net>) -> Self {
        Self { net, stack: Mutex::new(StateStack::new()) }
    }

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }
}

impl Player for SimPlayerV6 {
    fn name(&self) -> &str { "sim-v6" }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _, _) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);

        let mut best_p  = -1.0f32;
        let mut best_abs = 0usize;
        for i in 0..NUM_ACTIONS {
            if mask & (1u64 << i) != 0 && probs[i] > best_p {
                best_p = probs[i];
                best_abs = i;
            }
        }
        abstract_to_action(best_abs, legal, &view.my_hand)
            .and_then(|a| legal.iter().position(|x| std::ptr::eq(x, a)))
            .unwrap_or(0)
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (logits, _, _) = self.net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);

        let mut best_p   = -1.0f32;
        let mut best_abs = 0usize;
        for i in 0..NUM_ACTIONS {
            if mask & (1u64 << i) != 0 && probs[i] > best_p {
                best_p = probs[i];
                best_abs = i;
            }
        }
        if best_abs == 0 { return None; }
        abstract_to_chain(best_abs, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    fn decide(&self, _view: &GameView, req: &ChoiceRequest) -> usize {
        if req.options.is_empty() { 0 } else { rand::random::<usize>() % req.options.len() }
    }

    fn pick_targets(&self, _view: &GameView, _prompt: &str, opts: &[TargetOption], count: usize) -> Vec<usize> {
        (0..opts.len().min(count)).collect()
    }

    fn choose_discards(&self, _view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        hand.iter().take(n).map(|c| c.id).collect()
    }
}

// ── ForcedThenSimV6 — plays forced abstract action first, then greedy ─────────

struct ForcedThenSimV6 {
    forced_abstract: usize,
    played:          AtomicBool,
    sim:             SimPlayerV6,
}

impl Player for ForcedThenSimV6 {
    fn name(&self) -> &str { "forced-sim-v6" }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        if !self.played.swap(true, Ordering::Relaxed) {
            if let Some(idx) = legal.iter().position(|a| {
                action_to_abstract(a, &view.my_hand) == Some(self.forced_abstract)
            }) {
                return idx;
            }
        }
        self.sim.choose_action(view, legal)
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.sim.respond_to_chain(view, chain, legal)
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.sim.decide(view, req)
    }

    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
        self.sim.pick_targets(view, p, opts, c)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.sim.choose_discards(view, hand, n)
    }
}

// ── 1-step value lookahead ────────────────────────────────────────────────────
//
// Scores a CONCRETE action by applying it to a cloned state (AdvCPU logic
// resolves within-turn choices and opponent chain responses), then reading the
// value head on the successor. No learned disambiguation involved — this
// sidesteps the candidate head's label-noise ceiling entirely.

// Structural equality for Action (the engine type doesn't derive PartialEq).
fn same_action(a: &Action, b: &Action) -> bool {
    use novacana_engine::player::{Action::*, OmenAttachment as OA};
    match (a, b) {
        (Pass, Pass) => true,
        (PlayStar(x), PlayStar(y)) => x == y,
        (PlayDiscovery { card: c1, onto_star: s1 }, PlayDiscovery { card: c2, onto_star: s2 }) =>
            c1 == c2 && s1 == s2,
        (PlayFlare(x), PlayFlare(y)) => x == y,
        (PlayFracture(x), PlayFracture(y)) => x == y,
        (PlayOmen { card: c1, attachment: a1 }, PlayOmen { card: c2, attachment: a2 }) =>
            c1 == c2 && match (a1, a2) {
                (OA::SolarEclipse { system_star: x }, OA::SolarEclipse { system_star: y }) => x == y,
                (OA::PlanetOfLife { rocky_planet_disc: x }, OA::PlanetOfLife { rocky_planet_disc: y }) => x == y,
                _ => false,
            },
        (PlaySolarSystem { star: s1, discoveries: d1 }, PlaySolarSystem { star: s2, discoveries: d2 }) =>
            s1 == s2 && d1 == d2,
        (PlaySupernovas { fusion: f1, gravity: g1, gas: x1 }, PlaySupernovas { fusion: f2, gravity: g2, gas: x2 }) =>
            f1 == f2 && g1 == g2 && x1 == x2,
        (PlayProtostar { card: c1, choice: ch1 }, PlayProtostar { card: c2, choice: ch2 }) =>
            c1 == c2 && matches!(
                (ch1, ch2),
                (ProtostarChoice::DrawTwo, ProtostarChoice::DrawTwo)
                | (ProtostarChoice::SearchWhiteDwarf, ProtostarChoice::SearchWhiteDwarf)
            ),
        _ => false,
    }
}

// Plays the stored concrete action on its first turn, then behaves as AdvCPU.
// Within-turn choices (targets, decides) also use AdvCPU logic.
struct ForcedConcrete {
    forced: Action,
    played: AtomicBool,
    adv:    AdvancedCpuPlayer,
}

impl Player for ForcedConcrete {
    fn name(&self) -> &str { "forced-concrete" }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        if !self.played.swap(true, Ordering::Relaxed) {
            if let Some(idx) = legal.iter().position(|a| same_action(a, &self.forced)) {
                return idx;
            }
        }
        self.adv.choose_action(view, legal)
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
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

/// Value of applying `action` in `state`: run exactly one turn (chains resolve,
/// opponents respond via AdvCPU), then query the value head on the successor.
/// Natural game end inside the turn returns the true outcome instead.
pub fn value_lookahead(state: &GameState, my_idx: usize, action: &Action, net: &Arc<Net>) -> f32 {
    let n = state.players.len();
    let players: Vec<Box<dyn Player>> = (0..n).map(|i| -> Box<dyn Player> {
        if i == my_idx {
            Box::new(ForcedConcrete {
                forced: action.clone(),
                played: AtomicBool::new(false),
                adv:    AdvancedCpuPlayer::new(format!("fc{}", i)),
            })
        } else {
            Box::new(AdvancedCpuPlayer::new(format!("cpu{}", i)))
        }
    }).collect();

    let cap = state.turn_number + 1;
    let mut engine = GameEngine::from_state(state.clone(), players);
    engine.set_silent();
    engine.set_max_turns(cap);
    let result = engine.run();

    if result.turns_elapsed <= cap {
        return if result.winner_idx == my_idx { 1.0 } else { 0.0 };
    }

    let view = engine.build_view(my_idx);
    let snap = encode(&view);
    let mut stack = StateStack::new();
    stack.push(snap);
    stack.push(snap);
    stack.push(snap);
    let (_, value, _) = net.infer_with_body(&stack.stacked());
    value.clamp(0.0, 1.0)
}

// ── One MCTS simulation ───────────────────────────────────────────────────────
//
// Rollout leaves: up to SIM_TURNS turns of real play, then the value head.
// A pure value-leaf variant (1 turn + value head) was tried 2026-07-08 and
// CRASHED eval 38.7→28.2 in 5 iters while pl converged — the policy faithfully
// learned targets that were worse than itself, because the value head can't
// rank near states (same weakness the value-lookahead experiment measured).
// Rollouts are the trustworthy part of the leaf signal; keep them.

const SIM_TURNS: u32 = 20;

fn simulate_from_v6(state: &GameState, my_idx: usize, forced_abstract: usize, net: &Arc<Net>) -> f32 {
    let n = state.players.len();
    let players: Vec<Box<dyn Player>> = (0..n).map(|i| -> Box<dyn Player> {
        if i == my_idx {
            Box::new(ForcedThenSimV6 {
                forced_abstract,
                played: AtomicBool::new(false),
                sim:    SimPlayerV6::new(Arc::clone(net)),
            })
        } else {
            Box::new(AdvancedCpuPlayer::new(format!("cpu{}", i)))
        }
    }).collect();

    // turn_number is absolute, so the cap must be relative to the state's
    // current turn.
    let cap = state.turn_number + SIM_TURNS;
    let mut engine = GameEngine::from_state(state.clone(), players);
    engine.set_silent();
    engine.set_max_turns(cap);
    let result = engine.run();

    if result.turns_elapsed <= cap {
        return if result.winner_idx == my_idx { 1.0 } else { 0.0 };
    }

    // Truncated — value head on the final state.
    let view = engine.build_view(my_idx);
    let snap = encode(&view);
    let mut stack = StateStack::new();
    stack.push(snap);
    stack.push(snap);
    stack.push(snap);
    let features = stack.stacked();
    let (_, value, _) = net.infer_with_body(&features);
    value.clamp(0.0, 1.0)
}

// ── mcts_search_v6 ────────────────────────────────────────────────────────────
//
// Returns (visit_distribution, best_abstract_idx).
// visit_distribution sums to 1.0 over legal abstract actions; used as the
// policy training target during AlphaZero self-play.

pub fn mcts_search_v6(
    state:      &GameState,
    net:        &Arc<Net>,
    _view:      &GameView,
    _legal:     &[Action],
    features:   &[f32; INPUT_DIM],
    mask:       u64,
    sims:       usize,
    c_puct:     f32,
    root_noise: bool,
) -> ([f32; NUM_ACTIONS], usize) {
    // Policy priors from network
    let (logits, _, _) = net.infer_with_body(features);
    let mut priors = masked_softmax(&logits, mask);

    // Legal abstract indices (bits set in mask)
    let legal_abs: Vec<usize> = (0..NUM_ACTIONS).filter(|&i| mask & (1u64 << i) != 0).collect();

    if legal_abs.is_empty() {
        return ([0.0; NUM_ACTIONS], 0);
    }

    // Root Dirichlet noise (training only, OPT-IN): lets the self-play search
    // explore beyond the policy's current preferences. WARNING: at our scale
    // (~5 legal actions, ≤200 sims, 200 games/iter) ε=0.25 poisoned the visit
    // targets and crashed eval 38.7→27-28 twice (v6_az, v6b, 2026-07-08) —
    // AlphaZero's constants assume 35+ actions and millions of games. Only
    // enable together with much higher sim counts.
    if root_noise && legal_abs.len() >= 2 {
        use ndarray_rand::rand_distr::{Distribution, Gamma};
        let gamma = Gamma::new(0.3f32, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let noise: Vec<f32> = legal_abs.iter().map(|_| gamma.sample(&mut rng)).collect();
        let ns: f32 = noise.iter().sum::<f32>().max(1e-9);
        for (k, &i) in legal_abs.iter().enumerate() {
            priors[i] = 0.75 * priors[i] + 0.25 * noise[k] / ns;
        }
    }

    // Single legal action → trivial
    if legal_abs.len() == 1 || sims == 0 {
        let mut dist = [0.0f32; NUM_ACTIONS];
        dist[legal_abs[0]] = 1.0;
        return (dist, legal_abs[0]);
    }

    let my_idx = state.active_player_idx;
    let mut visits    = [0u32;  NUM_ACTIONS];
    let mut total_val = [0.0f32; NUM_ACTIONS];
    let mut total     = 0u32;

    // Run sims in small batches: one PUCT selection per batch, parallel rollouts.
    // Virtual counts applied before the batch prevent re-selection; corrected after.
    const SIM_BATCH: usize = 4;
    let mut sims_done = 0;
    while sims_done < sims {
        let n = (sims - sims_done).min(SIM_BATCH);

        let mut best_score = f32::NEG_INFINITY;
        let mut selected = legal_abs[0];
        for &i in &legal_abs {
            let q = if visits[i] == 0 { 0.0 } else { total_val[i] / visits[i] as f32 };
            let s = q + c_puct * priors[i] * ((total as f32).max(1.0)).sqrt() / (1 + visits[i]) as f32;
            if s > best_score { best_score = s; selected = i; }
        }

        visits[selected] += n as u32;
        total            += n as u32;

        let sum_val: f32 = (0..n).into_par_iter()
            .map(|_| simulate_from_v6(state, my_idx, selected, net))
            .sum();

        total_val[selected] += sum_val;
        sims_done += n;
    }

    // Build visit distribution
    let mut dist = [0.0f32; NUM_ACTIONS];
    if total > 0 {
        let tf = total as f32;
        for &i in &legal_abs {
            dist[i] = visits[i] as f32 / tf;
        }
    }

    let best = *legal_abs.iter()
        .max_by_key(|&&i| visits[i])
        .unwrap_or(&legal_abs[0]);

    (dist, best)
}

// ── MctsPlayer — eval player using v6 PUCT MCTS ──────────────────────────────

pub struct MctsPlayer {
    name_str:  String,
    weights:   Arc<Weights>,
    num_sims:  usize,
    c_puct:    f32,
    stack:     Mutex<StateStack>,
    fallback:  AdvancedCpuPlayer,
}

impl MctsPlayer {
    pub fn new(name: impl Into<String>, weights: Arc<Weights>, num_sims: usize) -> Self {
        let n = name.into();
        let fb = AdvancedCpuPlayer::new(n.clone());
        Self {
            name_str: n,
            weights,
            num_sims,
            c_puct: 1.5,
            stack: Mutex::new(StateStack::new()),
            fallback: fb,
        }
    }

    fn get_net(&self) -> Arc<Net> {
        Arc::new(Net::from_weights(&self.weights))
    }

    fn push_and_get_features(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode(view);
        let mut st = self.stack.lock().unwrap();
        st.push(snap);
        st.stacked()
    }
}

impl Player for MctsPlayer {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.push_and_get_features(view);
        let net = Net::from_weights(&self.weights);
        let mask = legal_mask(legal, &view.my_hand);
        let (logits, _, body) = net.infer_with_body(&features);
        let probs = masked_softmax(&logits, mask);

        let mut best_p  = -1.0f32;
        let mut best_abs = 0usize;
        for i in 0..NUM_ACTIONS {
            if mask & (1u64 << i) != 0 && probs[i] > best_p {
                best_p = probs[i]; best_abs = i;
            }
        }

        // Candidate head for disambiguation
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(best_abs))
            .map(|(i, _)| i)
            .collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }

        let total = matching.len();
        let cands: Vec<[f32; CAND_DIM]> = matching.iter().enumerate()
            .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, total))
            .collect();
        let scores = net.score_candidates(&body, &cands);
        let best = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        if legal.len() <= 1 { return 0; }
        let features = self.push_and_get_features(view);
        let net      = self.get_net();
        let mask     = legal_mask(legal, &view.my_hand);

        let (_, best_abs) = mcts_search_v6(state, &net, view, legal, &features, mask, self.num_sims, self.c_puct, false);

        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(best_abs))
            .map(|(i, _)| i)
            .collect();

        if matching.len() <= 1 { return matching.into_iter().next().unwrap_or(0); }

        // Candidate head for disambiguation
        let net_nd   = Net::from_weights(&self.weights);
        let (_, _, body) = net_nd.infer_with_body(&features);
        let total    = matching.len();
        let cands: Vec<[f32; CAND_DIM]> = matching.iter().enumerate()
            .map(|(slot, &ci)| encode_action_candidate(&legal[ci], view, slot, total))
            .collect();
        let scores = net_nd.score_candidates(&body, &cands);
        let best = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.fallback.respond_to_chain(view, chain, legal)
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.fallback.decide(view, req)
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.fallback.pick_targets(view, prompt, options, count)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.fallback.choose_discards(view, hand, n)
    }
}
