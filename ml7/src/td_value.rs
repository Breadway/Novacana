// ASTRA-7 TD-value stage (ASTRA7.md §4.2): self-play with the BC-trained net
// sampling its own policy at temperature T≈0.8, then TD(λ) targets for
// the value head only (trunk frozen - see net.rs::td_train_step / ASTRA7.md
// §8.2 for the freezing mechanism).
//
// TD(λ) return formulation (not spelled out in ASTRA7.md beyond "TD(λ)
// blend of successor value estimates and final outcome" - this is the exact
// math used, backward recursion, standard λ-return with NO intermediate
// reward (this game only pays out at the very end) and γ=1:
//
//   G_t = λ · G_{t+1} + (1 - λ) · V(s_{t+1})
//
// with the boundary condition that the LAST decision in a seat's trajectory
// has no recorded successor state, so both G and V at that boundary are
// defined to be the actual terminal outcome R (win=1/loss=0) - i.e. the
// outcome plays the role of "the value of the state right after the game
// ends" fictitiously, which collapses G at the last real decision to R
// exactly (λR + (1-λ)R = R) and flows the terminal signal backward from
// there. V(s) at each intermediate step is the value head's OWN estimate
// recorded live during self-play generation (the same net used to sample
// actions that game) - this is a single-pass target computation, not an
// iterated one: ASTRA7.md's "2-3 passes" means 2-3 training epochs over one
// generated batch, not 2-3 rounds of re-bootstrapping the targets.
//
// LAMBDA=1.0, not 0.9 (2026-07-10, first real run's postmortem - see
// STATE.md's "TD-value stage" section): at λ=0.9 the terminal outcome's
// weight at step t decays as 0.9^(steps-from-end), which over this game's
// long trajectories (~105 decisions/seat measured) is ~0.4% by the
// midpoint - meaning two-thirds of every trajectory's target is >99% a
// smoothed function of the net's OWN current value predictions, near-zero
// true gradient by construction (confirmed: value_rank top-2 hit rate came
// back at 50.8% vs the 65% gate, and a Fable-5-directed measurement found
// the pre-TD BC-warm-started head scored exactly at the theoretical chance
// floor - i.e. TD only ever propagated real signal a few turns per pass,
// consistent with 3 passes covering roughly the last third of the game).
// λ=1.0 collapses the recursion to G_t = R for every t (pure Monte Carlo
// regression against the actual terminal outcome, no bootstrapping) -
// higher variance per sample but zero bias and no reliance on the value
// head's own (still-forming) estimates, which is the right tradeoff given
// this stage generates 600K+ samples per chunk from a single training-head
// (value_hidden/value_out only, trunk frozen) with ample data to absorb the
// extra variance.

use std::sync::{Arc, Mutex};

use rand::{thread_rng, Rng};

use burn::tensor::backend::Backend;

use novacana_engine::card::{Card, CardId};
use novacana_engine::chain::ChainView;
use novacana_engine::cpu::AdvancedCpuPlayer;
use novacana_engine::deck::Deck;
use novacana_engine::game::GameEngine;
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption};

use crate::batch_infer::BatchedNet;
use crate::net::Net;
use crate::sampling;
use crate::tokens::{self, TokenSeq};

/// One seat's recorded (state tokens, value estimate) trajectory buffer -
/// see `compute_td_targets` for how this becomes TD(λ) training samples.
type TdBuffer = Arc<Mutex<Vec<(TokenSeq, f32)>>>;

pub const LAMBDA: f32 = 1.0;

/// One training example for `net::td_train_step`: a state's full token
/// sequence (value head reads only the state block, but the action block
/// still shapes it via self-attention - see mcts.rs::value_at_state's doc
/// for why encoding real action tokens matters) + its TD(λ) target.
pub struct TdSample {
    pub tokens: TokenSeq,
    pub td_target: f32,
}

// ── Self-play sampler: net's own policy, temperature-sampled ─────────────

struct TdSampler<B: Backend>
where
    B::Device: Clone + Send + Sync + 'static,
{
    name_str: String,
    bnet: Arc<BatchedNet<B>>,
    temperature: f32,
    buffer: TdBuffer,
}

impl<B: Backend> TdSampler<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn new(name: impl Into<String>, bnet: Arc<BatchedNet<B>>, temperature: f32, buffer: TdBuffer) -> Self {
        Self { name_str: name.into(), bnet, temperature, buffer }
    }

    /// Runs one forward pass, samples an index from the first `n` legal
    /// slots at `self.temperature`, records (tokens, value) if the decision
    /// was encodable. Returns the sampled index (0 if unencodable - should
    /// be exceedingly rare, ACTION_MAX has generous headroom, tokens.rs).
    fn decide_single(&self, view: &GameView, action_tokens: &[tokens::Token]) -> usize {
        let n = action_tokens.len();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, value) = self.bnet.forward_single(&enc.tokens);
                self.buffer.lock().unwrap().push((enc.tokens, value));
                sampling::sample_temperature(&masked, n, self.temperature, &mut thread_rng())
            }
            None => 0,
        }
    }

    /// Multi-pick variant: one forward pass, greedy top-k fill (§3.2 "start
    /// greedy") on top of a temperature-sampled first pick, matching the BC
    /// recorder's own greedy-first convention (tokens.rs only ever labels
    /// the first pick as the CE target).
    fn decide_multi(&self, view: &GameView, action_tokens: &[tokens::Token], count: usize) -> Vec<usize> {
        let n = action_tokens.len();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, value) = self.bnet.forward_single(&enc.tokens);
                self.buffer.lock().unwrap().push((enc.tokens, value));
                if count <= 1 {
                    vec![sampling::sample_temperature(&masked, n, self.temperature, &mut thread_rng())]
                } else {
                    sampling::top_k_indices(&masked, n, count)
                }
            }
            None => Vec::new(),
        }
    }
}

impl<B: Backend> Player for TdSampler<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name_str
    }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let action_tokens = tokens::encode_action_tokens_choose_action(view, legal);
        self.decide_single(view, &action_tokens)
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let action_tokens = tokens::encode_action_tokens_chain(view, legal);
        // index 0 = pass, matches tokens::encode_action_tokens_chain's convention.
        let idx = self.decide_single(view, &action_tokens);
        if idx == 0 {
            None
        } else {
            Some((idx - 1).min(legal.len().saturating_sub(1)))
        }
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        if req.options.is_empty() {
            return 0;
        }
        let action_tokens = tokens::encode_action_tokens_decide(view, req);
        self.decide_single(view, &action_tokens)
    }

    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() {
            return Vec::new();
        }
        let action_tokens = tokens::encode_action_tokens_targets(view, options);
        self.decide_multi(view, &action_tokens, count)
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        if hand.is_empty() {
            return Vec::new();
        }
        let action_tokens = tokens::encode_action_tokens_discards(hand);
        let idxs = self.decide_multi(view, &action_tokens, must_discard);
        idxs.into_iter().filter_map(|i| hand.get(i)).map(|c| c.id).collect()
    }
}

// ── TD(λ) target computation ─────────────────────────────────────────────

fn compute_td_targets(trajectory: &[(TokenSeq, f32)], outcome: f32) -> Vec<TdSample> {
    let n = trajectory.len();
    let mut targets = vec![0f32; n];
    let mut g_next = outcome; // "G" one step past the trajectory's end
    for i in (0..n).rev() {
        let v_next = if i + 1 < n { trajectory[i + 1].1 } else { outcome };
        let g = LAMBDA * g_next + (1.0 - LAMBDA) * v_next;
        targets[i] = g;
        g_next = g;
    }
    trajectory.iter().zip(targets).map(|((tok, _), g)| TdSample { tokens: *tok, td_target: g }).collect()
}

fn run_one_game<B: Backend>(bnet: &Arc<BatchedNet<B>>, num_players: usize, temperature: f32, advcpu_share: f32) -> Vec<TdSample>
where
    B::Device: Clone + Send + Sync + 'static,
{
    let mut rng = thread_rng();
    // AdvCPU seat mixing (2026-07-10, Fable 5's unfrozen-trunk postmortem):
    // `run_value_rank`'s gate scores exclusively on AdvCPU-vs-AdvCPU game
    // states (evals.rs::collect_decision_points), but self-play here used to
    // be 100% net-vs-net - the value head never trained on the distribution
    // it's graded against. Mixing some AdvCPU seats in closes that gap.
    // Only net seats produce buffered samples (bufs[i]=None for AdvCPU
    // seats), same pattern as az.rs's opponent-mix self-play.
    let mut bufs: Vec<Option<TdBuffer>> = Vec::new();
    let mut players: Vec<Box<dyn Player>> = Vec::new();
    for i in 0..num_players {
        if rng.gen::<f32>() < advcpu_share {
            bufs.push(None);
            players.push(Box::new(AdvancedCpuPlayer::new(format!("TdAdv{}", i + 1))));
        } else {
            let buf = Arc::new(Mutex::new(Vec::new()));
            bufs.push(Some(Arc::clone(&buf)));
            players.push(Box::new(TdSampler::new(format!("TdNet{}", i + 1), Arc::clone(bnet), temperature, buf)));
        }
    }
    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    // ASTRA7.md §8.1: net-driven self-play (unlike AdvCPU-only BC data) can
    // behave more erratically early in training - 400 matches §4.3's AZ
    // outer-game cap, same risk profile applies here.
    engine.set_max_turns(400);
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, buf) in bufs.into_iter().enumerate() {
        let Some(buf) = buf else { continue };
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let trajectory = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| Mutex::new(std::mem::take(&mut *b.lock().unwrap())))
            .into_inner()
            .unwrap();
        out.extend(compute_td_targets(&trajectory, outcome));
    }
    out
}

/// Generate TD-value training samples from `num_games` fresh self-play games
/// (`num_players` seats, policy sampled at `temperature`; `advcpu_share` of
/// seats are `AdvancedCpuPlayer` instead of the net, per-seat-per-game
/// independent rolls - 0.0 reproduces the original all-net self-play).
/// Rayon-parallel across games, same streaming-chunk convention as
/// `recorder::generate_samples` (ASTRA7.md §8.3 - never pregenerate more
/// than ~8-10K games at once). Callers pass `net.valid()` (inner backend) -
/// generation is forward-only, never build the autodiff graph for it
/// (ASTRA7.md §8.2).
pub fn generate_td_samples<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
    num_games: usize,
    num_players: usize,
    temperature: f32,
    advcpu_share: f32,
    concurrency: usize,
) -> Vec<TdSample>
where
    B::Device: Clone + Send + Sync + 'static,
{
    // Batched inference (2026-07-10, STATE.md's value-rank postmortem): a
    // shared BatchedNet coalesces concurrent games' single-decision forward
    // calls instead of each rayon worker serializing its own batch-of-1 GPU
    // round trip. Outer loop is `parallel_for_each` (plain OS threads), NOT
    // rayon - see batch_infer.rs's module doc for the nested-rayon-pool
    // deadlock this avoids. `max_batch` is tied to `concurrency` (2026-07-16,
    // Fable 5 consult) - a batch can never exceed the number of concurrent
    // callers anyway, so a fixed 64 was just dead headroom on boxes with
    // fewer than 64 callers running at once.
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency);
    crate::batch_infer::parallel_for_each(num_games, concurrency, |_| {
        run_one_game(&bnet, num_players, temperature, advcpu_share)
    })
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn td_targets_collapse_to_outcome_at_final_step() {
        let dummy = crate::tokens::assemble(
            &empty_view(),
            &tokens::encode_action_tokens_choose_action(&empty_view(), &[Action::Pass]),
        )
        .unwrap()
        .tokens;
        let trajectory = vec![(dummy, 0.3f32), (dummy, 0.6f32), (dummy, 0.9f32)];
        let samples = compute_td_targets(&trajectory, 1.0);
        assert_eq!(samples.len(), 3);
        // Last step's target must equal the outcome exactly (see doc comment's
        // boundary condition) - true regardless of LAMBDA.
        assert!((samples[2].td_target - 1.0).abs() < 1e-6);
        // At LAMBDA=1.0 (pure Monte Carlo - see module doc's 2026-07-10
        // postmortem) every step's target collapses to the outcome exactly,
        // ignoring the recorded V(s_next) values entirely.
        assert!((samples[0].td_target - 1.0).abs() < 1e-6);
        assert!((samples[1].td_target - 1.0).abs() < 1e-6);
    }

    fn empty_view() -> GameView {
        GameView {
            my_idx: 0,
            my_name: "Me".into(),
            my_hand: Vec::new(),
            my_star_chart: novacana_engine::star_chart::StarChart::new(),
            opponents: Vec::new(),
            deck_size: 90,
            active_player_idx: 0,
            turn_number: 1,
            dark_matter_void: None,
            history: Vec::new(),
            recent_plays: Vec::new(),
            discard_pile_counts: [0; 29],
        }
    }

    #[test]
    fn real_games_produce_finite_td_targets() {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device);
        let samples = generate_td_samples(&net, &device, 2, 4, 0.8, 0.0, 4);
        assert!(!samples.is_empty());
        for s in &samples {
            assert!(s.td_target.is_finite());
            assert!((-0.01..=1.01).contains(&s.td_target), "TD target out of expected [0,1] range: {}", s.td_target);
        }
    }
}
