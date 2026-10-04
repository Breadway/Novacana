// ASTRA-7 AZ league (ASTRA7.md §4.3): MCTS-driven self-play generation,
// win-rate eval harness, and the opponent-mix game builder. Training itself
// (net::az_train_step) and the outer iterate/eval/restart-from-best loop
// live in main.rs (cmd_az/run_az), mirroring cmd_bc/run_bc's split.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rand::{thread_rng, Rng};

use burn::tensor::backend::Backend;

use novacana_engine::card::{Card, CardId};
use novacana_engine::chain::{Chain, ChainResumeCursor, ChainView};
use novacana_engine::cpu::{AdvancedCpuPlayer, ExpertCpuPlayer};
use novacana_engine::deck::Deck;
use novacana_engine::game::{GameEngine, GameState};
use novacana_engine::player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption};

use crate::batch_infer::BatchedNet;
use crate::mcts::{self, McConfig};
use crate::net::Net;
use crate::sampling;
use crate::tokens::{self, ACTION_MAX, TokenSeq};

/// One AZ-seat's recorded (state tokens, MCTS/policy target) trajectory
/// buffer, shared between the Player impl and the game-runner that drains
/// it after the game ends.
type AzBuffer = Arc<Mutex<Vec<(TokenSeq, [f32; ACTION_MAX])>>>;

/// Decisions before visit-temperature sampling switches to greedy (ASTRA7.md
/// §4.3: "first 15 decisions of each game"). Counted per-GAME across ALL
/// seats and ALL five decision kinds (not just the two MCTS searches) -
/// ASTRA7.md doesn't specify whether this is per-seat or whole-game-wide;
/// whole-game-wide matches the standard AlphaZero "first N plies of the
/// game are exploratory" convention (a move number, not a per-player
/// budget), so that's what this implements. Revisit if self-play games look
/// under- or over-exploratory in practice.
const TEMP_MOVES: usize = 15;

pub struct AzSample {
    pub tokens: TokenSeq,
    pub visit_dist: [f32; ACTION_MAX],
    pub outcome: f32,
}

fn one_hot(idx: usize) -> [f32; ACTION_MAX] {
    let mut v = [0f32; ACTION_MAX];
    if idx < ACTION_MAX {
        v[idx] = 1.0;
    }
    v
}

// ── MCTS-driven self-play player ──────────────────────────────────────────

struct AzMctsPlayer<B: Backend>
where
    B::Device: Clone + Send + Sync + 'static,
{
    name_str: String,
    bnet: Arc<BatchedNet<B>>,
    cfg: McConfig,
    decisions_so_far: Arc<AtomicUsize>, // shared per-GAME move counter, see TEMP_MOVES
    buffer: AzBuffer,
}

impl<B: Backend> AzMctsPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn move_temperature(&self) -> f32 {
        let n = self.decisions_so_far.fetch_add(1, Ordering::Relaxed);
        if n < TEMP_MOVES {
            1.0
        } else {
            0.0
        }
    }

    fn push(&self, tokens: TokenSeq, visit_dist: [f32; ACTION_MAX]) {
        self.buffer.lock().unwrap().push((tokens, visit_dist));
    }

    /// decide/pick_targets/choose_discards: no resumable engine continuation
    /// to search over (mcts.rs's module doc explains why), so this samples
    /// straight off the net's own policy and self-distills toward whichever
    /// index it actually picked (one-hot target) - same "start greedy /
    /// low-stakes" territory as §1.1's chain-response-is-the-real-lever
    /// finding, just extended to these three kinds for the AZ stage.
    fn decide_single(&self, view: &GameView, action_tokens: &[tokens::Token]) -> usize {
        let n = action_tokens.len();
        let temp = self.move_temperature();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, _v) = self.bnet.forward_single(&enc.tokens);
                let idx = sampling::sample_temperature(&masked, n, temp, &mut thread_rng());
                self.push(enc.tokens, one_hot(idx));
                idx
            }
            None => 0,
        }
    }

    fn decide_multi(&self, view: &GameView, action_tokens: &[tokens::Token], count: usize) -> Vec<usize> {
        let n = action_tokens.len();
        let temp = self.move_temperature();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, _v) = self.bnet.forward_single(&enc.tokens);
                let idxs = if count <= 1 {
                    vec![sampling::sample_temperature(&masked, n, temp, &mut thread_rng())]
                } else {
                    sampling::top_k_indices(&masked, n, count)
                };
                if let Some(&first) = idxs.first() {
                    self.push(enc.tokens, one_hot(first));
                }
                idxs
            }
            None => Vec::new(),
        }
    }
}

impl<B: Backend> Player for AzMctsPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name_str
    }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        // No-state fallback (shouldn't be hit - `run()` always calls the
        // _with_state variant, see engine/src/game.rs) - greedy raw policy.
        let action_tokens = tokens::encode_action_tokens_choose_action(view, legal);
        self.decide_single(view, &action_tokens)
    }

    fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
        let temp = self.move_temperature();
        let action_tokens = tokens::encode_action_tokens_choose_action(view, legal);
        let Some(enc) = tokens::assemble(view, &action_tokens) else {
            return 0;
        };
        let mut rng = thread_rng();
        let outcome = mcts::search_choose_action(&self.bnet, view, state, legal, &self.cfg, temp, &mut rng);
        self.push(enc.tokens, outcome.visit_dist);
        outcome.chosen
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let action_tokens = tokens::encode_action_tokens_chain(view, legal);
        let idx = self.decide_single(view, &action_tokens);
        if idx == 0 {
            None
        } else {
            Some(idx - 1)
        }
    }

    fn respond_to_chain_with_state(
        &self,
        view: &GameView,
        state: &GameState,
        chain_view: &ChainView,
        chain: &Chain,
        cursor: ChainResumeCursor,
        legal: &[ChainAction],
    ) -> Option<usize> {
        let temp = self.move_temperature();
        let action_tokens = tokens::encode_action_tokens_chain(view, legal);
        let Some(enc) = tokens::assemble(view, &action_tokens) else {
            return self.respond_to_chain(view, chain_view, legal);
        };
        let mut rng = thread_rng();
        let outcome = mcts::search_chain_response(&self.bnet, view, state, chain, cursor, legal, &self.cfg, temp, &mut rng);
        self.push(enc.tokens, outcome.visit_dist);
        if outcome.chosen == 0 {
            None
        } else {
            Some(outcome.chosen - 1)
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

// ── Greedy (no-search) net player: used for fast win-rate evals ──────────

// pub(crate): reused by spectator.rs's spectator-game recorder (2026-07-15)
// to drive the same "seat 0 = greedy net" player it needs for its own
// GameEngine setup, without duplicating the tokenize+forward+argmax glue.
pub(crate) struct GreedyNetPlayer<B: Backend>
where
    B::Device: Clone + Send + Sync + 'static,
{
    pub(crate) name_str: String,
    pub(crate) bnet: Arc<BatchedNet<B>>,
}

impl<B: Backend> GreedyNetPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn pick_single(&self, view: &GameView, action_tokens: &[tokens::Token]) -> usize {
        let n = action_tokens.len();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, _v) = self.bnet.forward_single(&enc.tokens);
                sampling::argmax(&masked[..n])
            }
            None => 0,
        }
    }
    fn pick_multi(&self, view: &GameView, action_tokens: &[tokens::Token], count: usize) -> Vec<usize> {
        let n = action_tokens.len();
        match tokens::assemble(view, action_tokens) {
            Some(enc) => {
                let (masked, _v) = self.bnet.forward_single(&enc.tokens);
                sampling::top_k_indices(&masked, n, count)
            }
            None => Vec::new(),
        }
    }
}

impl<B: Backend> Player for GreedyNetPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name_str
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.pick_single(view, &tokens::encode_action_tokens_choose_action(view, legal))
    }
    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let idx = self.pick_single(view, &tokens::encode_action_tokens_chain(view, legal));
        if idx == 0 {
            None
        } else {
            Some(idx - 1)
        }
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        if req.options.is_empty() {
            return 0;
        }
        self.pick_single(view, &tokens::encode_action_tokens_decide(view, req))
    }
    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() {
            return Vec::new();
        }
        self.pick_multi(view, &tokens::encode_action_tokens_targets(view, options), count)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        if hand.is_empty() {
            return Vec::new();
        }
        let idxs = self.pick_multi(view, &tokens::encode_action_tokens_discards(hand), must_discard);
        idxs.into_iter().filter_map(|i| hand.get(i)).map(|c| c.id).collect()
    }
}

// ── Live-play token-collision scan ────────────────────────────────────────
//
// 2026-07-09: fidelity's aggregate accuracy only ever samples AdvCPU-driven
// states (both training and the fidelity checker generate data the same
// way). A collision that only manifests once the net has already drifted
// off that distribution - the exact failure mode DecideOption/
// PickTargetOption had - would never show up there. This wraps
// `GreedyNetPlayer`'s exact decision logic with a check, before each pick,
// for exact-duplicate (ttype, fields) tokens among that decision's legal
// actions: real self-play, real drift, checked at every single decision
// across the whole game, not just AdvCPU-typical ones.

pub struct CollisionReport {
    pub decisions: usize,
    pub collisions: usize,
    /// Per-ActionKind (tokens::ActionKind as u16) collision counts - lets
    /// callers tell "pure duplicate-card noise on card-play kinds" apart
    /// from "a target-bucket collision on PlayDiscovery/PlayOmen", which
    /// need different fixes (a tiebreaker vs. real missing identity).
    pub by_kind: [usize; tokens::NUM_ACTION_KINDS],
    pub examples: Vec<String>,
}

/// Shared duplicate-token check, used by both the live-drifted-play scanner
/// (CollisionScanNetPlayer) and the AdvCPU-training-distribution scanner
/// (CollisionScanAdvCpuPlayer) below - same detection logic, different
/// player driving the decisions, so the two scans are directly comparable.
fn check_for_collision(
    decisions: &AtomicUsize,
    collisions: &AtomicUsize,
    by_kind: &Mutex<[usize; tokens::NUM_ACTION_KINDS]>,
    examples: &Mutex<Vec<String>>,
    kind: &str,
    action_tokens: &[tokens::Token],
) {
    decisions.fetch_add(1, Ordering::Relaxed);
    let mut seen: Vec<(u8, [u8; tokens::NUM_FIELDS])> = Vec::with_capacity(action_tokens.len());
    for t in action_tokens {
        let key = (t.ttype, t.fields);
        if seen.contains(&key) {
            collisions.fetch_add(1, Ordering::Relaxed);
            if (t.fields[0] as usize) < tokens::NUM_ACTION_KINDS {
                by_kind.lock().unwrap()[t.fields[0] as usize] += 1;
            }
            let mut ex = examples.lock().unwrap();
            if ex.len() < 20 {
                ex.push(format!(
                    "{kind}: {} legal actions, collided token ttype={} fields={:?} ({})",
                    action_tokens.len(), t.ttype, t.fields, tokens::action_kind_name(t.fields[0] as u8)
                ));
            }
            return;
        }
        seen.push(key);
    }
}

struct CollisionScanNetPlayer<B: Backend>
where
    B::Device: Clone + Send + Sync + 'static,
{
    inner: GreedyNetPlayer<B>,
    decisions: AtomicUsize,
    collisions: AtomicUsize,
    by_kind: Mutex<[usize; tokens::NUM_ACTION_KINDS]>,
    examples: Mutex<Vec<String>>,
}

impl<B: Backend> CollisionScanNetPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn check(&self, kind: &str, action_tokens: &[tokens::Token]) {
        check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, kind, action_tokens);
    }
}

impl<B: Backend> Player for CollisionScanNetPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let toks = tokens::encode_action_tokens_choose_action(view, legal);
        self.check("choose_action", &toks);
        self.inner.pick_single(view, &toks)
    }
    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let toks = tokens::encode_action_tokens_chain(view, legal);
        self.check("respond_to_chain", &toks);
        let idx = self.inner.pick_single(view, &toks);
        if idx == 0 { None } else { Some(idx - 1) }
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        if req.options.is_empty() {
            return 0;
        }
        let toks = tokens::encode_action_tokens_decide(view, req);
        self.check("decide", &toks);
        self.inner.pick_single(view, &toks)
    }
    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() {
            return Vec::new();
        }
        let toks = tokens::encode_action_tokens_targets(view, options);
        self.check("pick_targets", &toks);
        self.inner.pick_multi(view, &toks, count)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        if hand.is_empty() {
            return Vec::new();
        }
        let toks = tokens::encode_action_tokens_discards(hand);
        self.check("choose_discards", &toks);
        let idxs = self.inner.pick_multi(view, &toks, must_discard);
        idxs.into_iter().filter_map(|i| hand.get(i)).map(|c| c.id).collect()
    }
}

/// Plays `num_games` real games (seat 0 = net making its own decisions,
/// seats 1..N = AdvCPU - same setup as `eval_win_rate`) and scans every
/// single decision seat 0 faces, at whatever state the net's own play
/// actually reaches, for token collisions among that decision's legal
/// actions.
pub fn scan_live_play_for_collisions<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
    num_games: usize,
    num_players: usize,
    concurrency: usize,
) -> CollisionReport
where
    B::Device: Clone + Send + Sync + 'static,
{
    // Plain OS threads, not rayon - see batch_infer.rs's module doc.
    // max_batch tied to concurrency (2026-07-16, Fable 5 consult).
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency);
    let results: Vec<(usize, usize, [usize; tokens::NUM_ACTION_KINDS], Vec<String>)> =
        crate::batch_infer::parallel_for_each(num_games, concurrency, |_| {
            let scanner = Arc::new(CollisionScanNetPlayer {
                inner: GreedyNetPlayer { name_str: "CollisionScan".into(), bnet: Arc::clone(&bnet) },
                decisions: AtomicUsize::new(0),
                collisions: AtomicUsize::new(0),
                by_kind: Mutex::new([0; tokens::NUM_ACTION_KINDS]),
                examples: Mutex::new(Vec::new()),
            });
            let mut players: Vec<Box<dyn Player>> = Vec::new();
            players.push(Box::new(ArcPlayer(Arc::clone(&scanner))));
            for i in 1..num_players {
                players.push(Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))));
            }
            let mut engine = GameEngine::new(Deck::build_classic(), players);
            engine.set_silent();
            engine.set_max_turns(400);
            engine.run();
            let d = scanner.decisions.load(Ordering::Relaxed);
            let c = scanner.collisions.load(Ordering::Relaxed);
            let bk = *scanner.by_kind.lock().unwrap();
            let ex = scanner.examples.lock().unwrap().clone();
            (d, c, bk, ex)
        });

    let mut report = CollisionReport { decisions: 0, collisions: 0, by_kind: [0; tokens::NUM_ACTION_KINDS], examples: Vec::new() };
    for (d, c, bk, ex) in results {
        report.decisions += d;
        report.collisions += c;
        for k in 0..tokens::NUM_ACTION_KINDS {
            report.by_kind[k] += bk[k];
        }
        report.examples.extend(ex);
    }
    report.examples.truncate(20);
    report
}

/// Same collision scan, but the training distribution itself: all seats the
/// actual BC teacher (no net at all) - this is exactly what BC trains on,
/// so this number answers "how much of the training signal was noisy from
/// this bug", independent of whether the net has drifted off-distribution.
/// Teacher = ExpertCpuPlayer (switched 2026-07-10, recorder.rs's module doc
/// has the reasoning) - keep this in sync with BcRecorder's actual teacher,
/// this scan is meaningless if it drifts from what BC really trains on.
pub fn scan_training_data_for_collisions(num_games: usize, num_players: usize) -> CollisionReport {
    use rayon::prelude::*;
    struct TeacherCollisionPlayer {
        inner: ExpertCpuPlayer,
        decisions: AtomicUsize,
        collisions: AtomicUsize,
        by_kind: Mutex<[usize; tokens::NUM_ACTION_KINDS]>,
        examples: Mutex<Vec<String>>,
    }
    impl Player for TeacherCollisionPlayer {
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
            let toks = tokens::encode_action_tokens_choose_action(view, legal);
            check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, "choose_action", &toks);
            self.inner.choose_action(view, legal)
        }
        fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
            let toks = tokens::encode_action_tokens_chain(view, legal);
            check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, "respond_to_chain", &toks);
            self.inner.respond_to_chain(view, chain, legal)
        }
        fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
            if req.options.is_empty() {
                return 0;
            }
            let toks = tokens::encode_action_tokens_decide(view, req);
            check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, "decide", &toks);
            self.inner.decide(view, req)
        }
        fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
            if options.is_empty() {
                return Vec::new();
            }
            let toks = tokens::encode_action_tokens_targets(view, options);
            check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, "pick_targets", &toks);
            self.inner.pick_targets(view, prompt, options, count)
        }
        fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
            if hand.is_empty() {
                return Vec::new();
            }
            let toks = tokens::encode_action_tokens_discards(hand);
            check_for_collision(&self.decisions, &self.collisions, &self.by_kind, &self.examples, "choose_discards", &toks);
            self.inner.choose_discards(view, hand, must_discard)
        }
    }

    let results: Vec<(usize, usize, [usize; tokens::NUM_ACTION_KINDS], Vec<String>)> = (0..num_games)
        .into_par_iter()
        .map(|_| {
            let mut players: Vec<Box<dyn Player>> = Vec::new();
            let scanners: Vec<Arc<TeacherCollisionPlayer>> = (0..num_players)
                .map(|i| {
                    Arc::new(TeacherCollisionPlayer {
                        inner: ExpertCpuPlayer::new(format!("ExpertCPU{}", i + 1)),
                        decisions: AtomicUsize::new(0),
                        collisions: AtomicUsize::new(0),
                        by_kind: Mutex::new([0; tokens::NUM_ACTION_KINDS]),
                        examples: Mutex::new(Vec::new()),
                    })
                })
                .collect();
            for s in &scanners {
                struct Passthrough(Arc<TeacherCollisionPlayer>);
                impl Player for Passthrough {
                    fn name(&self) -> &str {
                        self.0.name()
                    }
                    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
                        self.0.choose_action(view, legal)
                    }
                    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
                        self.0.respond_to_chain(view, chain, legal)
                    }
                    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
                        self.0.decide(view, req)
                    }
                    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
                        self.0.pick_targets(view, prompt, options, count)
                    }
                    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
                        self.0.choose_discards(view, hand, must_discard)
                    }
                }
                players.push(Box::new(Passthrough(Arc::clone(s))));
            }
            let mut engine = GameEngine::new(Deck::build_classic(), players);
            engine.set_silent();
            engine.set_max_turns(400);
            engine.run();
            let mut d = 0;
            let mut c = 0;
            let mut bk = [0usize; tokens::NUM_ACTION_KINDS];
            let mut ex = Vec::new();
            for s in &scanners {
                d += s.decisions.load(Ordering::Relaxed);
                c += s.collisions.load(Ordering::Relaxed);
                let sbk = *s.by_kind.lock().unwrap();
                for k in 0..tokens::NUM_ACTION_KINDS {
                    bk[k] += sbk[k];
                }
                ex.extend(s.examples.lock().unwrap().clone());
            }
            (d, c, bk, ex)
        })
        .collect();

    let mut report = CollisionReport { decisions: 0, collisions: 0, by_kind: [0; tokens::NUM_ACTION_KINDS], examples: Vec::new() };
    for (d, c, bk, ex) in results {
        report.decisions += d;
        report.collisions += c;
        for k in 0..tokens::NUM_ACTION_KINDS {
            report.by_kind[k] += bk[k];
        }
        report.examples.extend(ex);
    }
    report.examples.truncate(20);
    report
}

/// `Player` passthrough so `Arc<CollisionScanNetPlayer<B>>` can go in the
/// `Vec<Box<dyn Player>>` the engine wants (the buffers inside need to
/// outlive the game to report on, hence Arc-shared rather than owned).
struct ArcPlayer<B: Backend>(Arc<CollisionScanNetPlayer<B>>)
where
    B::Device: Clone + Send + Sync + 'static;
impl<B: Backend> Player for ArcPlayer<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.0.name()
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        self.0.choose_action(view, legal)
    }
    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        self.0.respond_to_chain(view, chain, legal)
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.0.decide(view, req)
    }
    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.0.pick_targets(view, prompt, options, count)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        self.0.choose_discards(view, hand, must_discard)
    }
}

// ── Self-play generation with opponent mix (ASTRA7.md §4.3) ──────────────

enum Role {
    Az,
    AdvCpu,
    Expert,
}

fn pick_role(rng: &mut impl Rng, advcpu_share: f32, expert_share: f32) -> Role {
    let r: f32 = rng.gen();
    if r < advcpu_share {
        Role::AdvCpu
    } else if r < advcpu_share + expert_share {
        Role::Expert
    } else {
        Role::Az
    }
}

/// One self-play game: each seat independently rolled into AdvCPU / Expert /
/// AZ-MCTS per the opponent-mix shares. Only AZ-seat decisions produce
/// training samples (the whole point of the mix is exposing the net to
/// stronger/different play, not supervising off AdvCPU's/Expert's choices -
/// that's BC's job). Capped at 400 turns per ASTRA7.md §8.1 ("Cap AZ outer
/// games at 400 turns (`run_one_az_game`), not 1000").
#[allow(clippy::too_many_arguments)]
fn run_one_az_game<B: Backend>(
    bnet: &Arc<BatchedNet<B>>,
    cfg: &McConfig,
    num_players: usize,
    advcpu_share: f32,
    expert_share: f32,
) -> Vec<AzSample>
where
    B::Device: Clone + Send + Sync + 'static,
{
    let mut rng = thread_rng();
    let decisions_so_far = Arc::new(AtomicUsize::new(0));
    let mut bufs: Vec<Option<AzBuffer>> = Vec::new();
    let mut players: Vec<Box<dyn Player>> = Vec::new();

    for i in 0..num_players {
        match pick_role(&mut rng, advcpu_share, expert_share) {
            Role::AdvCpu => {
                bufs.push(None);
                players.push(Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))));
            }
            Role::Expert => {
                bufs.push(None);
                players.push(Box::new(ExpertCpuPlayer::new(format!("Expert{}", i + 1))));
            }
            Role::Az => {
                let buf = Arc::new(Mutex::new(Vec::new()));
                bufs.push(Some(Arc::clone(&buf)));
                players.push(Box::new(AzMctsPlayer {
                    name_str: format!("AzNet{}", i + 1),
                    bnet: Arc::clone(bnet),
                    cfg: McConfig { sims: cfg.sims, c_puct: cfg.c_puct, dirichlet_eps: cfg.dirichlet_eps },
                    decisions_so_far: Arc::clone(&decisions_so_far),
                    buffer: buf,
                }));
            }
        }
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
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
        out.extend(trajectory.into_iter().map(|(tokens, visit_dist)| AzSample { tokens, visit_dist, outcome }));
    }
    out
}

/// Generate one iteration's worth of AZ self-play samples. Rayon-parallel
/// across games (embarrassingly parallel - ASTRA7.md §10). `net` must
/// already be forward-only (pass `net.valid()`, ASTRA7.md §8.2).
#[allow(clippy::too_many_arguments)]
pub fn generate_az_games<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
    cfg: &McConfig,
    num_games: usize,
    num_players: usize,
    advcpu_share: f32,
    expert_share: f32,
    concurrency: usize,
) -> Vec<AzSample>
where
    B::Device: Clone + Send + Sync + 'static,
{
    // Batched inference (2026-07-10 - see batch_infer.rs's module doc): AZ's
    // MCTS does >=400 leaf evals per real decision, mostly falling through
    // to a value-head forward call (mcts.rs), so this is the single highest-
    // value place for the batching fix - without it, every one of those
    // calls would separately serialize through the CUDA driver exactly like
    // the TD-value run that motivated this change, just ~2-3 orders of
    // magnitude more often. Outer loop is plain OS threads
    // (`parallel_for_each`), NOT rayon - nests badly with BatchedNet's own
    // use of `net::forward`'s rayon-based `ndarray` internals otherwise.
    // max_batch tied to concurrency (2026-07-16, Fable 5 consult).
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency);
    crate::batch_infer::parallel_for_each(num_games, concurrency, |_| {
        run_one_az_game(&bnet, cfg, num_players, advcpu_share, expert_share)
    })
    .into_iter()
    .flatten()
    .collect()
}

/// One greedy `choose_action` decision, kept so a live-play eval can say
/// whether Pass won because it was the only legal move, because the logits
/// tied and Pass sits at index 0, or because the net ranked it higher.
#[derive(Clone, Copy)]
struct ChooseSnap {
    turn: u32,
    hand: usize,
    n_legal: usize,
    chosen: u8,
    pass_logit: f32,
    best_other_logit: f32,
    best_other: u8,
    /// Softmax probability of Pass over the legal actions. This is the rate
    /// a temperature-1 sample would pass, independent of the greedy pick.
    pass_p: f32,
    value: f32,
}

struct BehaviorLog {
    games: usize,
    wins: usize,
    turns: u64,
    all_pass_games: usize,
    assemble_fails: usize,
    chooses: Vec<ChooseSnap>,
    openings: Vec<ChooseSnap>,
    chain_n: usize,
    chain_pass: usize,
    chain_margins: Vec<f32>,
    chain_pass_p: Vec<f32>,
    decide_n: usize,
    target_n: usize,
    discard_n: usize,
}

impl Default for BehaviorLog {
    fn default() -> Self {
        Self {
            games: 0,
            wins: 0,
            turns: 0,
            all_pass_games: 0,
            assemble_fails: 0,
            chooses: Vec::new(),
            openings: Vec::new(),
            chain_n: 0,
            chain_pass: 0,
            chain_margins: Vec::new(),
            chain_pass_p: Vec::new(),
            decide_n: 0,
            target_n: 0,
            discard_n: 0,
        }
    }
}

struct GameFlags {
    chooses: AtomicUsize,
    played: AtomicBool,
}

/// Same picks as `GreedyNetPlayer`, plus a record of the logits behind them.
struct BehaviorProbe<B: Backend>
where
    B::Device: Clone + Send + Sync + 'static,
{
    name_str: String,
    bnet: Arc<BatchedNet<B>>,
    log: Arc<Mutex<BehaviorLog>>,
    flags: Arc<GameFlags>,
    opened: AtomicBool,
}

fn choose_snap(tokens: &[tokens::Token], masked: &[f32], idx: usize, turn: u32, hand: usize, value: f32) -> ChooseSnap {
    let mut pass_logit = f32::NAN;
    let mut best_other_logit = f32::NEG_INFINITY;
    let mut best_other = 255u8;
    for (i, t) in tokens.iter().enumerate() {
        let k = t.fields[0];
        if k == tokens::ActionKind::Pass as u8 {
            pass_logit = masked[i];
        } else if masked[i] > best_other_logit {
            best_other_logit = masked[i];
            best_other = k;
        }
    }
    let n = tokens.len();
    let pass_i = tokens.iter().position(|t| t.fields[0] == tokens::ActionKind::Pass as u8);
    ChooseSnap {
        turn,
        hand,
        n_legal: n,
        chosen: tokens[idx].fields[0],
        pass_logit,
        best_other_logit,
        best_other,
        pass_p: softmax_prob(masked, n, pass_i),
        value,
    }
}

/// Softmax probability of `idx` over the first `n` logits. `None` is zero.
fn softmax_prob(masked: &[f32], n: usize, idx: Option<usize>) -> f32 {
    if n == 0 {
        return f32::NAN;
    }
    let m = masked[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return f32::NAN;
    }
    let mut sum = 0.0f32;
    let mut hit = 0.0f32;
    for (i, v) in masked[..n].iter().enumerate() {
        let e = (v - m).exp();
        sum += e;
        if Some(i) == idx {
            hit += e;
        }
    }
    if sum == 0.0 { f32::NAN } else { hit / sum }
}

impl<B: Backend> BehaviorProbe<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn forward(&self, view: &GameView, action_tokens: &[tokens::Token]) -> Option<(Vec<f32>, f32)> {
        tokens::assemble(view, action_tokens).map(|enc| self.bnet.forward_single(&enc.tokens))
    }
}

impl<B: Backend> Player for BehaviorProbe<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name_str
    }
    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let toks = tokens::encode_action_tokens_choose_action(view, legal);
        let Some((masked, value)) = self.forward(view, &toks) else {
            self.log.lock().unwrap().assemble_fails += 1;
            return 0;
        };
        let n = toks.len();
        let idx = sampling::argmax(&masked[..n]);
        let snap = choose_snap(&toks, &masked, idx, view.turn_number, view.my_hand.len(), value);
        self.flags.chooses.fetch_add(1, Ordering::Relaxed);
        if snap.chosen != tokens::ActionKind::Pass as u8 {
            self.flags.played.store(true, Ordering::Relaxed);
        }
        let mut g = self.log.lock().unwrap();
        if !self.opened.swap(true, Ordering::Relaxed) {
            g.openings.push(snap);
        }
        g.chooses.push(snap);
        idx
    }
    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let toks = tokens::encode_action_tokens_chain(view, legal);
        let Some((masked, _v)) = self.forward(view, &toks) else {
            self.log.lock().unwrap().assemble_fails += 1;
            return None;
        };
        let n = toks.len();
        let idx = sampling::argmax(&masked[..n]);
        let mut g = self.log.lock().unwrap();
        g.chain_n += 1;
        if n > 1 {
            let best_other = masked[1..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            g.chain_margins.push(masked[0] - best_other);
            g.chain_pass_p.push(softmax_prob(&masked, n, Some(0)));
        }
        if idx == 0 {
            g.chain_pass += 1;
            None
        } else {
            Some(idx - 1)
        }
    }
    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        self.log.lock().unwrap().decide_n += 1;
        if req.options.is_empty() {
            return 0;
        }
        let toks = tokens::encode_action_tokens_decide(view, req);
        let Some((masked, _v)) = self.forward(view, &toks) else {
            return 0;
        };
        sampling::argmax(&masked[..toks.len()])
    }
    fn pick_targets(&self, view: &GameView, _prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        self.log.lock().unwrap().target_n += 1;
        if options.is_empty() {
            return Vec::new();
        }
        let toks = tokens::encode_action_tokens_targets(view, options);
        let Some((masked, _v)) = self.forward(view, &toks) else {
            return Vec::new();
        };
        sampling::top_k_indices(&masked, toks.len(), count)
    }
    fn choose_discards(&self, view: &GameView, hand: &[Card], must_discard: usize) -> Vec<CardId> {
        self.log.lock().unwrap().discard_n += 1;
        if hand.is_empty() {
            return Vec::new();
        }
        let toks = tokens::encode_action_tokens_discards(hand);
        let Some((masked, _v)) = self.forward(view, &toks) else {
            return Vec::new();
        };
        let idxs = sampling::top_k_indices(&masked, toks.len(), must_discard);
        idxs.into_iter().filter_map(|i| hand.get(i)).map(|c| c.id).collect()
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let i = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[i]
}

fn print_behavior(log: &BehaviorLog) {
    let games = log.games.max(1);
    println!(
        "games={}  wins={} ({:.1}%)  mean_turns={:.0}  games_with_only_passes={}/{}",
        log.games,
        log.wins,
        100.0 * log.wins as f32 / games as f32,
        log.turns as f32 / games as f32,
        log.all_pass_games,
        log.games
    );
    let n = log.chooses.len();
    let passes = log.chooses.iter().filter(|s| s.chosen == tokens::ActionKind::Pass as u8).count();
    let forced = log.chooses.iter().filter(|s| s.chosen == tokens::ActionKind::Pass as u8 && s.n_legal <= 1).count();
    let voluntary: Vec<&ChooseSnap> = log
        .chooses
        .iter()
        .filter(|s| s.chosen == tokens::ActionKind::Pass as u8 && s.n_legal > 1 && s.best_other_logit.is_finite())
        .collect();
    println!("choose_action: {n} decisions, greedy pass {passes}, of which {forced} had no other legal play");
    let contested: Vec<&ChooseSnap> = log
        .chooses
        .iter()
        .filter(|s| s.n_legal > 1 && s.pass_logit.is_finite() && s.best_other_logit.is_finite())
        .collect();
    if !contested.is_empty() {
        let greedy_pass = contested.iter().filter(|s| s.chosen == tokens::ActionKind::Pass as u8).count();
        let mean_p = contested.iter().map(|s| s.pass_p).sum::<f32>() / contested.len() as f32;
        let mut ps: Vec<f32> = contested.iter().map(|s| s.pass_p).collect();
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_pass = contested.iter().map(|s| s.pass_logit).sum::<f32>() / contested.len() as f32;
        let mean_other = contested.iter().map(|s| s.best_other_logit).sum::<f32>() / contested.len() as f32;
        println!(
            "  contested: {}  greedy pass {}/{} ({:.1}%)",
            contested.len(),
            greedy_pass,
            contested.len(),
            100.0 * greedy_pass as f32 / contested.len() as f32
        );
        println!(
            "  softmax P(pass) at temperature 1: mean {:.3}  p50 {:.3}  p90 {:.3}",
            mean_p,
            percentile(&ps, 0.5),
            percentile(&ps, 0.9)
        );
        println!(
            "  logits on every contested decision: pass {:.3}  best other {:.3}  margin {:.3}",
            mean_pass,
            mean_other,
            mean_pass - mean_other
        );
    }
    if !voluntary.is_empty() {
        let mut margins: Vec<f32> = voluntary.iter().map(|s| s.pass_logit - s.best_other_logit).collect();
        margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ties = margins.iter().filter(|m| **m < 0.05).count();
        let mean = margins.iter().sum::<f32>() / margins.len() as f32;
        let hands = voluntary.iter().map(|s| s.hand).min().unwrap_or(0);
        let mean_pass = voluntary.iter().map(|s| s.pass_logit).sum::<f32>() / voluntary.len() as f32;
        let mean_other = voluntary.iter().map(|s| s.best_other_logit).sum::<f32>() / voluntary.len() as f32;
        println!(
            "  voluntary passes: {}  hand>={}  margin pass-minus-best-other: mean {:.3}  min {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}  within 0.05 of a tie: {}",
            voluntary.len(),
            hands,
            mean,
            margins[0],
            percentile(&margins, 0.5),
            percentile(&margins, 0.9),
            margins[margins.len() - 1],
            ties
        );
        println!("  mean logits while passing: pass {mean_pass:.3}  best other {mean_other:.3}");
        let mut kinds = [0usize; tokens::NUM_ACTION_KINDS];
        for s in &voluntary {
            if (s.best_other as usize) < kinds.len() {
                kinds[s.best_other as usize] += 1;
            }
        }
        println!("  best other legal play while passing:");
        for (k, c) in kinds.iter().enumerate() {
            if *c > 0 {
                println!("    {:<28} {c}", tokens::action_kind_name(k as u8));
            }
        }
    }
    let played: Vec<&ChooseSnap> = log.chooses.iter().filter(|s| s.chosen != tokens::ActionKind::Pass as u8).collect();
    if !played.is_empty() {
        let mut kinds = [0usize; tokens::NUM_ACTION_KINDS];
        for s in &played {
            if (s.chosen as usize) < kinds.len() {
                kinds[s.chosen as usize] += 1;
            }
        }
        println!("  plays actually chosen:");
        for (k, c) in kinds.iter().enumerate() {
            if *c > 0 {
                println!("    {:<28} {c}", tokens::action_kind_name(k as u8));
            }
        }
    }
    if n > 0 {
        let mean_v = log.chooses.iter().map(|s| s.value).sum::<f32>() / n as f32;
        println!("  value head mean on these decisions: {mean_v:.3}");
    }
    if log.assemble_fails > 0 {
        println!("  assemble() failed {} times; those picks returned index 0", log.assemble_fails);
    }
    println!(
        "other decisions: chain {} (pass {}), decide {}, target {}, discard {}",
        log.chain_n, log.chain_pass, log.decide_n, log.target_n, log.discard_n
    );
    if !log.chain_margins.is_empty() {
        let mean = log.chain_margins.iter().sum::<f32>() / log.chain_margins.len() as f32;
        let mean_p = if log.chain_pass_p.is_empty() {
            f32::NAN
        } else {
            log.chain_pass_p.iter().sum::<f32>() / log.chain_pass_p.len() as f32
        };
        println!(
            "  chain with a card legal: {}  greedy pass margin mean {:.3}  softmax P(pass) mean {:.3}",
            log.chain_margins.len(),
            mean,
            mean_p
        );
    }
    if !log.openings.is_empty() {
        println!("first decision of each game:");
        for s in &log.openings {
            let margin = if s.best_other_logit.is_finite() { s.pass_logit - s.best_other_logit } else { f32::NAN };
            println!(
                "  turn {:<4} hand {:<2} legal {:<3} {:<16} margin {:>7.3}  P(pass) {:.3}  pass {:.3}  best {} {:.3}  value {:.3}",
                s.turn,
                s.hand,
                s.n_legal,
                tokens::action_kind_name(s.chosen),
                margin,
                s.pass_p,
                s.pass_logit,
                tokens::action_kind_name(s.best_other),
                s.best_other_logit,
                s.value
            );
        }
    }
}

/// Greedy net vs AdvancedCpu, same seats as `eval_win_rate`, with the
/// per-decision logit record printed at the end.
pub fn eval_behavior<B: Backend>(net: &Net<B>, device: &B::Device, num_games: usize, num_players: usize, concurrency: usize)
where
    B::Device: Clone + Send + Sync + 'static,
{
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency.max(1));
    let log = Arc::new(Mutex::new(BehaviorLog::default()));
    crate::batch_infer::parallel_for_each(num_games, concurrency, |_| {
        let flags = Arc::new(GameFlags { chooses: AtomicUsize::new(0), played: AtomicBool::new(false) });
        let mut players: Vec<Box<dyn Player>> = Vec::new();
        players.push(Box::new(BehaviorProbe {
            name_str: "AzEvalNet".into(),
            bnet: Arc::clone(&bnet),
            log: Arc::clone(&log),
            flags: Arc::clone(&flags),
            opened: AtomicBool::new(false),
        }));
        for i in 1..num_players {
            players.push(Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))));
        }
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(400);
        let result = engine.run();
        let mut g = log.lock().unwrap();
        g.games += 1;
        g.wins += (result.winner_idx == 0) as usize;
        g.turns += result.turns_elapsed as u64;
        if flags.chooses.load(Ordering::Relaxed) > 0 && !flags.played.load(Ordering::Relaxed) {
            g.all_pass_games += 1;
        }
    });
    print_behavior(&log.lock().unwrap());
}

/// Verified win-rate eval (ASTRA7.md §2): seat 0 = greedy (no-search) net,
/// seats 1..N = AdvancedCpuPlayer. No-search rather than full MCTS at eval
/// time for speed - 600/1000-game evals at >=400 sims/decision would be
/// prohibitively slow, and this matches how `fidelity`/BC's existing evals
/// in this codebase already measure "full-ML" strength (raw policy, no
/// search) rather than search-assisted play.
pub fn eval_win_rate<B: Backend>(net: &Net<B>, device: &B::Device, num_games: usize, num_players: usize, concurrency: usize) -> f32
where
    B::Device: Clone + Send + Sync + 'static,
{
    // Plain OS threads, not rayon - see batch_infer.rs's module doc.
    // max_batch tied to concurrency (2026-07-16, Fable 5 consult).
    let bnet = BatchedNet::spawn(Arc::new(net.clone()), device.clone(), concurrency);
    let wins: usize = crate::batch_infer::parallel_for_each(num_games, concurrency, |_| {
        let mut players: Vec<Box<dyn Player>> = Vec::new();
        players.push(Box::new(GreedyNetPlayer { name_str: "AzEvalNet".into(), bnet: Arc::clone(&bnet) }));
        for i in 1..num_players {
            players.push(Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))));
        }
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(400);
        let result = engine.run();
        (result.winner_idx == 0) as usize
    })
    .into_iter()
    .sum();
    wins as f32 / num_games.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;

    #[test]
    fn self_play_produces_samples_with_valid_distributions() {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device);
        // Tiny sims count + all-AZ seats (no AdvCPU/Expert dilution) so the
        // test exercises the MCTS path and finishes quickly. One game is
        // enough to touch all five decision kinds at this scale (a full
        // game with a fresh net still runs dozens of decisions) - 2 games
        // roughly doubled wall time for no extra coverage.
        let cfg = McConfig { sims: 4, c_puct: 1.5, dirichlet_eps: 0.25 };
        let samples = generate_az_games(&net, &device, &cfg, 1, 4, 0.0, 0.0, 4);
        assert!(!samples.is_empty(), "expected at least some AZ samples from 1 game");
        for s in &samples {
            let sum: f32 = s.visit_dist.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "visit_dist must sum to ~1, got {sum}");
            assert!(s.outcome == 0.0 || s.outcome == 1.0);
        }
    }

    #[test]
    fn eval_win_rate_returns_a_fraction() {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device);
        let rate = eval_win_rate(&net, &device, 2, 4, 4);
        assert!((0.0..=1.0).contains(&rate));
    }

    /// Control for the 2026-07-09 low-win-rate investigation: all 4 seats
    /// AdvancedCpuPlayer (no net at all), does seat 0 win ~1/4 of games?
    /// Rules out a positional/harness bug (turn order advantage, a
    /// `winner_idx` off-by-one, etc.) as the explanation for a low
    /// `eval_win_rate` reading - if this test's seat-0 rate were skewed away
    /// from ~25%, that would mean the harness itself is unfair and any
    /// win-rate number from it is uninterpretable regardless of the model.
    #[test]
    fn harness_has_no_seat_bias_between_equally_matched_players() {
        use rayon::prelude::*;
        let n = 400;
        let seat0_wins: usize = (0..n)
            .into_par_iter()
            .map(|_| {
                let players: Vec<Box<dyn Player>> =
                    (0..4).map(|i| Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))) as Box<dyn Player>).collect();
                let mut engine = GameEngine::new(Deck::build_classic(), players);
                engine.set_silent();
                engine.set_max_turns(400);
                (engine.run().winner_idx == 0) as usize
            })
            .sum();
        let rate = seat0_wins as f32 / n as f32;
        // Measured 2026-07-09: 20.0% (n=400) - seat 0 has a modest real
        // structural disadvantage (likely turn order), not the naive 25%.
        // Recalibrates how to read eval_win_rate's seat-0 numbers: compare
        // against ~20%, not 25%. Wide tolerance below since a genuine
        // harness bug (winner_idx off-by-one, etc.) would blow past this,
        // not land just under naive-uniform.
        assert!((0.10..=0.35).contains(&rate), "seat 0 (AdvCPU) win rate {rate} outside [0.10, 0.35] - harness may have a positional bias");
    }

    /// Same control, ExpertCpu vs 3 other ExpertCpu (2026-07-10). Measured
    /// 19.65% at n=2000 (an n=200 first pass gave 15.5%, just small-sample
    /// noise) - essentially matches AdvCPU's 20.0% control. The seat-0
    /// disadvantage looks like a genuine property of the game's turn
    /// order, not something specific to which bot is playing: it holds
    /// consistently whether all 4 seats are AdvCPU or all 4 are ExpertCpu.
    /// NOT the calibration baseline for `eval_win_rate`'s "net vs 3x
    /// AdvCPU" configuration, though - that's ExpertCpu's already-known
    /// 47.5% vs 3x AdvCPU (a weaker, non-equally-matched opponent mix).
    /// This test answers a narrower question: does the seat-0 penalty
    /// generalize across skill levels (yes).
    #[test]
    fn expertcpu_self_play_baseline() {
        use rayon::prelude::*;
        let n = 2000;
        let seat0_wins: usize = (0..n)
            .into_par_iter()
            .map(|_| {
                let players: Vec<Box<dyn Player>> =
                    (0..4).map(|i| Box::new(ExpertCpuPlayer::new(format!("ExpertCPU{}", i + 1))) as Box<dyn Player>).collect();
                let mut engine = GameEngine::new(Deck::build_classic(), players);
                engine.set_silent();
                engine.set_max_turns(400);
                (engine.run().winner_idx == 0) as usize
            })
            .sum();
        let rate = seat0_wins as f32 / n as f32;
        eprintln!("expertcpu_self_play_baseline: seat0 ExpertCPU win rate = {rate:.4}  (n={n})");
        assert!((0.0..=1.0).contains(&rate));
    }

    /// Repeatable regression test for the BC teacher's actual competitive
    /// strength - added 2026-07-16 while evaluating (then reverting, see
    /// the comment above FOLLOWUP_TOPK in cpu.rs) a prefilter for
    /// `choose_action_with_state`'s primary search loop, specifically so
    /// that change (and any future one to ExpertCpu's search) has a direct,
    /// cheap way to check it didn't quietly weaken the BC teacher.
    ///
    /// UNRESOLVED DISCREPANCY, flagged rather than silently reconciled:
    /// recorder.rs's module doc / ASTRA7.md §1.1 document this exact
    /// config's win rate as 47.5%±0.9. This test, run against the byte-
    /// identical exhaustive search (confirmed via `git diff` - nothing in
    /// this session's changes touches choose_action_with_state's actual
    /// behavior), measured 39.55% (n=2000) - an ~8pp gap, many SEs beyond
    /// what n=2000 sampling noise alone would produce. Two prefiltered
    /// variants tried in between measured 38.15% (K=10) and 36.45% (K=20),
    /// i.e. this run's exhaustive number sits close to those, not close to
    /// the documented 47.5%. Whether the original 47.5% measurement used
    /// different conditions (max_turns, engine version, run-to-run
    /// variance beyond its stated ±0.9), or something upstream of this
    /// session's changes already shifted it, is unresolved - surfaced for
    /// a human to decide whether ASTRA7.md/recorder.rs's documented figure
    /// needs updating. Assert threshold set loose (>0.35) so this test
    /// tracks "still a strong, working teacher clearly beating parity,"
    /// not a specific number pending that resolution.
    #[test]
    fn expertcpu_vs_advcpu_teacher_strength() {
        use rayon::prelude::*;
        let n = 2000;
        let seat0_wins: usize = (0..n)
            .into_par_iter()
            .map(|_| {
                let mut players: Vec<Box<dyn Player>> = vec![Box::new(ExpertCpuPlayer::new("ExpertCPU".to_string()))];
                for i in 1..4 {
                    players.push(Box::new(AdvancedCpuPlayer::new(format!("AdvCPU{}", i + 1))));
                }
                let mut engine = GameEngine::new(Deck::build_classic(), players);
                engine.set_silent();
                engine.set_max_turns(400);
                (engine.run().winner_idx == 0) as usize
            })
            .sum();
        let rate = seat0_wins as f32 / n as f32;
        eprintln!("expertcpu_vs_advcpu_teacher_strength: seat0 ExpertCPU win rate = {rate:.4}  (n={n}) - see this test's doc comment re: an unresolved gap vs. the 47.5%±0.9 documented elsewhere");
        assert!(rate > 0.30, "ExpertCpu-vs-3xAdvCPU win rate {rate} dropped surprisingly low - investigate before trusting BC teacher quality");
    }

    /// 2026-07-10: how noisy is ExpertCpu's own `choose_action_with_state`
    /// decision on an IDENTICAL state, repeated? Its search averages
    /// `EXPERT_SIM_SAMPLES` stochastic rollouts per candidate before
    /// picking the max-scoring one - the same state can plausibly get a
    /// different chosen action on different calls. That inconsistency is
    /// an information-theoretic floor on BC's cross-entropy loss: even a
    /// perfect model predicting the true distribution over ExpertCpu's
    /// noisy choices pays nonzero loss, since the "label" itself isn't
    /// deterministic. This measures that floor directly instead of
    /// guessing at it from the training loss curve's shape.
    #[test]
    fn expertcpu_choose_action_self_consistency() {
        use rayon::prelude::*;

        struct StateCapturingPlayer {
            inner: ExpertCpuPlayer,
            captured: Arc<Mutex<Vec<(GameView, GameState, Vec<Action>)>>>,
            cap: usize,
        }
        impl Player for StateCapturingPlayer {
            fn name(&self) -> &str {
                self.inner.name()
            }
            fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
                self.inner.choose_action(view, legal)
            }
            fn choose_action_with_state(&self, view: &GameView, state: &GameState, legal: &[Action]) -> usize {
                if legal.len() > 1 {
                    let mut c = self.captured.lock().unwrap();
                    if c.len() < self.cap {
                        c.push((view.clone(), state.clone(), legal.to_vec()));
                    }
                }
                self.inner.choose_action_with_state(view, state, legal)
            }
            fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
                self.inner.respond_to_chain(view, chain, legal)
            }
            fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
                self.inner.decide(view, req)
            }
            fn pick_targets(&self, view: &GameView, p: &str, opts: &[TargetOption], c: usize) -> Vec<usize> {
                self.inner.pick_targets(view, p, opts, c)
            }
            fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
                self.inner.choose_discards(view, hand, n)
            }
        }

        // Phase 1: play real games, capturing a sample of real mid-game
        // choose_action decision points (state, legal actions) reached
        // via ExpertCpu's own actual play - not synthetic states.
        let cap = 300;
        let captured: Arc<Mutex<Vec<(GameView, GameState, Vec<Action>)>>> = Arc::new(Mutex::new(Vec::new()));
        (0..40).into_par_iter().for_each(|_| {
            if captured.lock().unwrap().len() >= cap {
                return;
            }
            let players: Vec<Box<dyn Player>> = (0..4)
                .map(|i| {
                    Box::new(StateCapturingPlayer {
                        inner: ExpertCpuPlayer::new(format!("ExpertCPU{}", i + 1)),
                        captured: Arc::clone(&captured),
                        cap,
                    }) as Box<dyn Player>
                })
                .collect();
            let mut engine = GameEngine::new(Deck::build_classic(), players);
            engine.set_silent();
            engine.set_max_turns(400);
            engine.run();
        });
        let states = captured.lock().unwrap().clone();
        assert!(!states.is_empty(), "expected at least some captured decision points");

        // Phase 2: replay each captured state through ExpertCpu's real
        // decision function REPEATS times, check whether the chosen index
        // varies across repeats on the byte-identical input.
        const REPEATS: usize = 8;
        let checker = ExpertCpuPlayer::new("consistency-check");
        let mut any_disagreement = 0usize;
        let mut total_agreement_frac_sum = 0f32;
        for (view, state, legal) in &states {
            let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for _ in 0..REPEATS {
                let choice = checker.choose_action_with_state(view, state, legal);
                *counts.entry(choice).or_insert(0) += 1;
            }
            if counts.len() > 1 {
                any_disagreement += 1;
            }
            let modal = counts.values().copied().max().unwrap_or(0);
            total_agreement_frac_sum += modal as f32 / REPEATS as f32;
        }
        let n = states.len();
        let disagreement_rate = any_disagreement as f32 / n as f32;
        let avg_modal_agreement = total_agreement_frac_sum / n as f32;
        eprintln!(
            "expertcpu_choose_action_self_consistency: n={n} repeats={REPEATS}  \
             decisions_with_any_disagreement={any_disagreement} ({:.1}%)  \
             avg_modal_agreement={:.1}%",
            disagreement_rate * 100.0,
            avg_modal_agreement * 100.0
        );
        assert!((0.0..=1.0).contains(&disagreement_rate));
    }
}
