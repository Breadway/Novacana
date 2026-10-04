// ASTRA-7 `value-rank` diagnostic (ASTRA7.md §4.2): does the value head rank
// AdvCPU's actual choice among the best options for a state? Gate: >=65% of
// sampled decision points must put AdvCPU's pick in the value head's top-2
// by successor value. This is the check that guards the entire AZ stage -
// MCTS's leaf-eval quality is only as good as this ranking.
//
// Restricted to `choose_action` decisions only (not chain/decide/targets/
// discards): "apply each legal action via the engine" needs a real
// clone-and-simulate successor per candidate, and `choose_action` is the
// only decision kind with that machinery already proven out (mirrors
// `expert_sim_action`/v6's `value_lookahead` - see mcts.rs's
// `successor_value_choose_action`, which this reuses directly rather than
// re-deriving the same technique a third time). `choose_action` is also the
// dominant decision kind by sample count (ASTRA7.md §10.1's fidelity table),
// so it's a representative measurement, not a cherry-picked one.
//
// Trivial decisions (<=2 legal actions) are excluded: "did the value head
// rank the pick in the top-2" is meaningless when there's no other option to
// rank against, and would silently inflate the aggregate percentage with
// free successes.

use std::sync::{Arc, Mutex};

use rand::seq::SliceRandom;
use rayon::prelude::*;

use burn::tensor::backend::Backend;

use novacana_engine::card::{Card, CardId};
use novacana_engine::chain::ChainView;
use novacana_engine::cpu::AdvancedCpuPlayer;
use novacana_engine::deck::Deck;
use novacana_engine::game::{GameEngine, GameState};
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption};

use crate::batch_infer::BatchedNet;
use crate::mcts;
use crate::net::Net;

const MIN_LEGAL_FOR_RANKING: usize = 3;

struct Captured {
    state: GameState,
    my_idx: usize,
    legal: Vec<Action>,
    chosen: usize,
}

struct ValueRankRecorder {
    name_str: String,
    fallback: AdvancedCpuPlayer,
    buffer: Arc<Mutex<Vec<Captured>>>,
}

impl Player for ValueRankRecorder {
    fn name(&self) -> &str {
        &self.name_str
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.fallback.choose_action(view, legal)
    }
    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        let chosen = self.fallback.choose_action(view, legal);
        if legal.len() >= MIN_LEGAL_FOR_RANKING {
            self.buffer.lock().unwrap().push(Captured {
                state: state.clone(),
                my_idx: view.my_idx,
                legal: legal.to_vec(),
                chosen,
            });
        }
        chosen
    }
    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.fallback.respond_to_chain(view, chain, legal)
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.fallback.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
        self.fallback.pick_targets(view, p, opts, c)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        self.fallback.choose_discards(view, hand, n)
    }
}

fn collect_decision_points(num_games: usize, num_players: usize) -> Vec<Captured> {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    (0..num_games).into_par_iter().for_each(|_| {
        let players: Vec<Box<dyn Player>> = (0..num_players)
            .map(|i| -> Box<dyn Player> {
                Box::new(ValueRankRecorder {
                    name_str: format!("VR{}", i + 1),
                    fallback: AdvancedCpuPlayer::new(format!("VRadv{}", i + 1)),
                    buffer: Arc::clone(&buffer),
                })
            })
            .collect();
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        let _ = engine.run();
    });
    Arc::try_unwrap(buffer).unwrap_or_else(|a| Mutex::new(std::mem::take(&mut *a.lock().unwrap()))).into_inner().unwrap()
}

/// Runs the value-rank diagnostic: `num_games` fresh AdvCPU self-play games
/// to source decision points from, subsampled (shuffled, then truncated) to
/// `target_points`, each scored by applying every legal action and reading
/// the value head at the successor. Returns (top-2 hit rate, points actually
/// scored - may be less than `target_points` if generation didn't produce
/// enough qualifying decisions).
pub fn run_value_rank<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
    num_games: usize,
    num_players: usize,
    target_points: usize,
    concurrency: usize,
) -> (f32, usize)
where
    B::Device: Clone + Send + Sync + 'static,
{
    let mut points = collect_decision_points(num_games, num_players);
    points.shuffle(&mut rand::thread_rng());
    points.truncate(target_points);

    let n = points.len();
    if n == 0 {
        return (0.0, 0);
    }

    // Batched inference (2026-07-10, see batch_infer.rs's module doc): each
    // point scores every legal action (a handful of sequential calls), and
    // ~13-14 points are scored concurrently - same batch-of-1-serialization
    // problem as self-play generation had. Plain OS threads
    // (`parallel_for_each`), not rayon - nests badly with BatchedNet's own
    // use of `net::forward`'s rayon-based `ndarray` internals otherwise.
    // `max_batch` tied to `concurrency` (2026-07-16, Fable 5 consult) - see
    // td_value.rs::generate_td_samples's comment on the same change.
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency);
    let hits: usize = crate::batch_infer::parallel_for_each(n, concurrency, |i| {
        let p = &points[i];
        let mut scored: Vec<(usize, f32)> = (0..p.legal.len())
            .map(|j| (j, mcts::successor_value_choose_action(&bnet, &p.state, p.my_idx, j, num_players)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(2).any(|&(j, _)| j == p.chosen)
    })
    .into_iter()
    .filter(|&hit| hit)
    .count();

    (hits as f32 / n as f32, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn value_rank_runs_and_returns_a_fraction() {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device);
        let (rate, n) = run_value_rank(&net, &device, 4, 4, 20, 4);
        assert!((0.0..=1.0).contains(&rate));
        // With only 4 games there may not be 20 qualifying points, but there
        // should be at least a handful (>=3 legal actions is common).
        assert!(n > 0, "expected at least one qualifying decision point");
    }
}
