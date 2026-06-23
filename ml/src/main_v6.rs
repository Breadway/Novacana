mod action;
mod candidate;
mod feature;
mod feature_v6;
mod mcts;
mod generate;
mod net;
mod net_v6;
mod net_v6_burn;
mod player;
mod self_play;
mod self_play_v6;
mod train;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use novacana_engine::{
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::GameEngine,
    player::{Action, ChainAction, ChoiceRequest, GameView, Player, TargetOption},
    card::{Card, CardId},
    chain::ChainView,
};

use burn::backend::Autodiff;
use burn::tensor::backend::AutodiffBackend;
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "gpu")]
use burn_wgpu::{Wgpu, WgpuDevice};

use candidate::{CAND_DIM, encode_card_candidate, encode_system_candidate, encode_target_candidate};
use feature_v6::{encode_v6, FEATURE_DIM_V6};
use net_v6::{NetV6, WeightsV6, INPUT_DIM};
use net_v6_burn::{
    NetV6Burn, MAX_CANDS, make_optimizer,
    rl_train_step_burn, cand_step_burn,
    pretrain_value_step_burn, pretrain_policy_step_burn,
};
use self_play_v6::{run_self_play_v6, PerBuffer};

// ── OpenBLAS thread count switching ──────────────────────────────────────────
//
// Set to 1 before game generation (rayon owns parallelism across games).
// Set to 16 before training (single-threaded caller, large batches).
// unsafe: env var mutation is inherently racy, but we are always single-threaded
// at the switch point and rayon workers re-read the var on each call.

fn openblas_threads(n: usize) {
    unsafe { std::env::set_var("OPENBLAS_NUM_THREADS", n.to_string()); }
}

// ── Stacked-state wrapper for eval ───────────────────────────────────────────

struct MlPlayerV6 {
    name_str: String,
    net:      Arc<NetV6>,
    stack:    Mutex<StackBuf>,
}

struct StackBuf {
    buf:   [[f32; FEATURE_DIM_V6]; 3],
}

impl StackBuf {
    fn new() -> Self { Self { buf: [[0.0; FEATURE_DIM_V6]; 3] } }
    fn push(&mut self, snap: [f32; FEATURE_DIM_V6]) {
        self.buf[0] = self.buf[1];
        self.buf[1] = self.buf[2];
        self.buf[2] = snap;
    }
    fn stacked(&self) -> [f32; INPUT_DIM] {
        let mut out = [0.0f32; INPUT_DIM];
        out[..FEATURE_DIM_V6].copy_from_slice(&self.buf[0]);
        out[FEATURE_DIM_V6..2 * FEATURE_DIM_V6].copy_from_slice(&self.buf[1]);
        out[2 * FEATURE_DIM_V6..].copy_from_slice(&self.buf[2]);
        out
    }
}

impl MlPlayerV6 {
    fn new(name: impl Into<String>, net: Arc<NetV6>) -> Self {
        Self { name_str: name.into(), net, stack: Mutex::new(StackBuf::new()) }
    }

    fn features(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode_v6(view);
        let mut s = self.stack.lock().unwrap();
        s.push(snap);
        s.stacked()
    }
}

impl Player for MlPlayerV6 {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        let features = self.features(view);
        use action::{action_to_abstract, legal_mask};
        let mask = legal_mask(legal, &view.my_hand);
        let (abs_idx, _lp, _val, body) =
            self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        let hand = &view.my_hand;
        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs_idx))
            .map(|(i, _)| i)
            .collect();
        if matching.is_empty() { return 0; }
        if matching.len() == 1 { return matching[0]; }
        let total = matching.len();
        let cands: Vec<[f32; CAND_DIM]> = matching.iter().enumerate().map(|(slot, &ci)| {
            match &legal[ci] {
                novacana_engine::player::Action::PlayDiscovery { onto_star, .. } => {
                    view.my_star_chart.systems.iter()
                        .find(|s| s.star_id == *onto_star)
                        .map(|s| encode_system_candidate(s, false, slot, total))
                        .unwrap_or([0.0; CAND_DIM])
                }
                novacana_engine::player::Action::PlayStar(id) => {
                    hand.iter().find(|c| c.id == *id)
                        .map(|c| encode_card_candidate(c, slot, total))
                        .unwrap_or([0.0; CAND_DIM])
                }
                _ => { let mut c = [0.0f32; CAND_DIM]; c[22] = slot as f32 / total as f32; c }
            }
        }).collect();
        let scores = self.net.score_candidates(&body, &cands);
        let best = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        matching[best]
    }

    fn respond_to_chain(&self, view: &GameView, _chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        let features = self.features(view);
        use action::{chain_legal_mask, abstract_to_chain};
        let mask = chain_legal_mask(legal, &view.my_hand);
        let (abs_idx, _, _, _) = self.net.infer_stochastic_with_body(&features, mask, 0.1, rand::random());
        if abs_idx == 0 { return None; }
        abstract_to_chain(abs_idx, legal, &view.my_hand)
            .and_then(|r| legal.iter().position(|x| std::ptr::eq(x, r)))
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let cands: Vec<[f32; CAND_DIM]> = (0..total).map(|i| {
            let mut c = [0.0f32; CAND_DIM];
            c[22] = i as f32 / total as f32;
            c[23] = total as f32 / 10.0;
            c
        }).collect();
        let scores = self.net.score_candidates(&body, &cands);
        scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    fn pick_targets(&self, view: &GameView, _: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let total = options.len();
        let cands: Vec<[f32; CAND_DIM]> = options.iter().enumerate()
            .map(|(i, opt)| encode_target_candidate(opt, view, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(count);
        order
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.features(view);
        let body = self.net.body_forward(&features);
        let total = hand.len();
        let cands: Vec<[f32; CAND_DIM]> = hand.iter().enumerate()
            .map(|(i, card)| encode_card_candidate(card, i, total))
            .collect();
        let scores = self.net.score_candidates(&body, &cands);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        order.truncate(n);
        order.iter().map(|&i| hand[i].id).collect()
    }
}

// ── Imitation (AdvCPU) data recording ─────────────────────────────────────────

struct ImPending {
    features:    Vec<f32>,
    legal_mask:  u64,
    action:      u8,
    has_policy:  bool,
    cand_feats:  Vec<f32>,  // flat: n_cands * CAND_DIM; empty if no candidate choice
    cand_chosen: u8,
    has_cand:    bool,
}

pub struct ImitateSample {
    pub features:    Vec<f32>,
    pub legal_mask:  u64,
    pub action:      u8,
    pub outcome:     f32,
    pub has_policy:  bool,
    pub cand_feats:  Vec<f32>,
    pub cand_chosen: u8,
    pub has_cand:    bool,
}

struct AdvCpuRecorder {
    name_str: String,
    fallback: AdvancedCpuPlayer,
    stack:    Mutex<StackBuf>,
    buffer:   Arc<Mutex<Vec<ImPending>>>,
}

impl AdvCpuRecorder {
    fn new(name: impl Into<String>, buffer: Arc<Mutex<Vec<ImPending>>>) -> Self {
        let n = name.into();
        Self {
            fallback: AdvancedCpuPlayer::new(n.clone()),
            name_str: n,
            stack: Mutex::new(StackBuf::new()),
            buffer,
        }
    }

    fn encode_and_stack(&self, view: &GameView) -> [f32; INPUT_DIM] {
        let snap = encode_v6(view);
        let mut s = self.stack.lock().unwrap();
        s.push(snap);
        s.stacked()
    }

    fn push(&self, p: ImPending) {
        self.buffer.lock().unwrap().push(p);
    }
}

impl Player for AdvCpuRecorder {
    fn name(&self) -> &str { &self.name_str }

    fn choose_action(&self, view: &GameView, legal: &[Action]) -> usize {
        use action::{action_to_abstract, legal_mask};
        let features = self.encode_and_stack(view);
        let mask = legal_mask(legal, &view.my_hand);

        let chosen_idx = self.fallback.choose_action(view, legal);
        let hand = &view.my_hand;
        let abs = action_to_abstract(&legal[chosen_idx], hand).unwrap_or(0);

        let matching: Vec<usize> = legal.iter().enumerate()
            .filter(|(_, a)| action_to_abstract(a, hand) == Some(abs))
            .map(|(i, _)| i)
            .collect();

        let (cand_feats, cand_chosen, has_cand) = if matching.len() >= 2 {
            let chosen_slot = matching.iter().position(|&ci| ci == chosen_idx).unwrap_or(0);
            // Only record if chosen fits within MAX_CANDS
            if chosen_slot < MAX_CANDS {
                let nc = matching.len().min(MAX_CANDS);
                let cands: Vec<[f32; CAND_DIM]> = matching[..nc].iter().enumerate().map(|(slot, &ci)| {
                    match &legal[ci] {
                        novacana_engine::player::Action::PlayDiscovery { onto_star, .. } => {
                            view.my_star_chart.systems.iter()
                                .find(|s| s.star_id == *onto_star)
                                .map(|s| encode_system_candidate(s, false, slot, nc))
                                .unwrap_or([0.0; CAND_DIM])
                        }
                        novacana_engine::player::Action::PlayStar(id) => {
                            hand.iter().find(|c| c.id == *id)
                                .map(|c| encode_card_candidate(c, slot, nc))
                                .unwrap_or([0.0; CAND_DIM])
                        }
                        _ => {
                            let mut c = [0.0f32; CAND_DIM];
                            c[22] = slot as f32 / nc as f32;
                            c[23] = nc as f32 / 10.0;
                            c
                        }
                    }
                }).collect();
                let flat = cands.iter().flat_map(|c| c.iter().copied()).collect();
                (flat, chosen_slot as u8, true)
            } else {
                (Vec::new(), 0u8, false)
            }
        } else {
            (Vec::new(), 0u8, false)
        };

        self.push(ImPending {
            features: features.to_vec(),
            legal_mask: mask,
            action: abs as u8,
            has_policy: true,
            cand_feats,
            cand_chosen,
            has_cand,
        });
        chosen_idx
    }

    fn respond_to_chain(&self, view: &GameView, chain: &ChainView, legal: &[ChainAction]) -> Option<usize> {
        use action::{chain_legal_mask, chain_response_to_abstract};
        let features = self.encode_and_stack(view);
        let mask = chain_legal_mask(legal, &view.my_hand);
        let result = self.fallback.respond_to_chain(view, chain, legal);
        let abs = result
            .and_then(|i| chain_response_to_abstract(&legal[i], &view.my_hand))
            .unwrap_or(0);
        self.push(ImPending {
            features: features.to_vec(),
            legal_mask: mask,
            action: abs as u8,
            has_policy: true,
            cand_feats: Vec::new(),
            cand_chosen: 0,
            has_cand: false,
        });
        result
    }

    fn decide(&self, view: &GameView, req: &ChoiceRequest) -> usize {
        let total = req.options.len();
        if total == 0 { return 0; }
        let features = self.encode_and_stack(view);
        let chosen = self.fallback.decide(view, req);
        // Only record if chosen fits within MAX_CANDS (skip decisions with too many options)
        if total >= 2 && chosen < MAX_CANDS {
            let nc = total.min(MAX_CANDS);
            let cands: Vec<[f32; CAND_DIM]> = (0..nc).map(|i| {
                let mut c = [0.0f32; CAND_DIM];
                c[22] = i as f32 / nc as f32;
                c[23] = nc as f32 / 10.0;
                c
            }).collect();
            self.push(ImPending {
                features: features.to_vec(),
                legal_mask: 0,
                action: 0,
                has_policy: false,
                cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: chosen as u8,
                has_cand: true,
            });
        }
        chosen
    }

    fn pick_targets(&self, view: &GameView, prompt: &str, options: &[TargetOption], count: usize) -> Vec<usize> {
        if options.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let chosen = self.fallback.pick_targets(view, prompt, options, count);
        if options.len() >= 2 && !chosen.is_empty() && chosen[0] < MAX_CANDS {
            let nc = options.len().min(MAX_CANDS);
            let cands: Vec<[f32; CAND_DIM]> = options.iter().take(nc).enumerate()
                .map(|(i, opt)| encode_target_candidate(opt, view, i, nc))
                .collect();
            self.push(ImPending {
                features: features.to_vec(),
                legal_mask: 0,
                action: 0,
                has_policy: false,
                cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                cand_chosen: chosen[0] as u8,
                has_cand: true,
            });
        }
        chosen
    }

    fn choose_discards(&self, view: &GameView, hand: &[Card], n: usize) -> Vec<CardId> {
        if hand.is_empty() { return Vec::new(); }
        let features = self.encode_and_stack(view);
        let chosen_ids = self.fallback.choose_discards(view, hand, n);
        if hand.len() >= 2 && !chosen_ids.is_empty() {
            let total = hand.len();
            let first_slot = hand.iter().position(|c| c.id == chosen_ids[0]).unwrap_or(0);
            if first_slot < MAX_CANDS {
                let nc = total.min(MAX_CANDS);
                let cands: Vec<[f32; CAND_DIM]> = hand.iter().take(nc).enumerate()
                    .map(|(i, card)| encode_card_candidate(card, i, nc))
                    .collect();
                self.push(ImPending {
                    features: features.to_vec(),
                    legal_mask: 0,
                    action: 0,
                    has_policy: false,
                    cand_feats: cands.iter().flat_map(|c| c.iter().copied()).collect(),
                    cand_chosen: first_slot as u8,
                    has_cand: true,
                });
            }
        }
        chosen_ids
    }
}

fn run_one_imitate_game(num_players: usize) -> Vec<ImitateSample> {
    let mut bufs: Vec<Arc<Mutex<Vec<ImPending>>>> = Vec::new();
    let mut players: Vec<Box<dyn Player>> = Vec::new();

    for i in 0..num_players {
        let buf = Arc::new(Mutex::new(Vec::new()));
        bufs.push(Arc::clone(&buf));
        players.push(Box::new(AdvCpuRecorder::new(format!("AdvCPU{}", i + 1), buf)));
    }

    let mut engine = GameEngine::new(Deck::build_classic(), players);
    engine.set_silent();
    engine.set_max_turns(1000);
    let result = engine.run();

    let mut out = Vec::new();
    for (player_idx, buf) in bufs.into_iter().enumerate() {
        let outcome = if result.winner_idx == player_idx { 1.0f32 } else { 0.0 };
        let pending = Arc::try_unwrap(buf)
            .unwrap_or_else(|b| std::mem::take(&mut *b.lock().unwrap()).into())
            .into_inner()
            .unwrap();
        for p in pending {
            out.push(ImitateSample {
                features:    p.features,
                legal_mask:  p.legal_mask,
                action:      p.action,
                outcome,
                has_policy:  p.has_policy,
                cand_feats:  p.cand_feats,
                cand_chosen: p.cand_chosen,
                has_cand:    p.has_cand,
            });
        }
    }
    out
}

fn generate_imitate_data(num_games: usize, num_players: usize) -> Vec<ImitateSample> {
    use rayon::prelude::*;
    (0..num_games)
        .into_par_iter()
        .flat_map(|_| run_one_imitate_game(num_players))
        .collect()
}

// ── Weights serialisation ─────────────────────────────────────────────────────

fn save_weights_v6(w: &WeightsV6, path: &str) -> std::io::Result<()> {
    let data = bincode::serialize(w).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, &data)
}

fn load_weights_v6(path: &str) -> std::io::Result<WeightsV6> {
    let data = std::fs::read(path)?;
    bincode::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

// ── Eval ──────────────────────────────────────────────────────────────────────

fn eval_v6(net: &Arc<NetV6>, num_games: usize, num_players: usize) -> f32 {
    let mut wins = 0usize;
    for _ in 0..num_games {
        let net_c = Arc::clone(net);
        let ml_seat = 0usize;
        let mut players: Vec<Box<dyn Player>> = Vec::new();
        players.push(Box::new(MlPlayerV6::new("Astra-6", net_c)));
        for i in 1..num_players {
            players.push(Box::new(AdvancedCpuPlayer::new(format!("CPU{}", i))));
        }
        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        let result = engine.run();
        if result.winner_idx == ml_seat { wins += 1; }
    }
    wins as f32 / num_games as f32
}

// ── CLI helpers ───────────────────────────────────────────────────────────────

fn flag_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn flag_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn flag_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // Game gen phase uses single-threaded BLAS (rayon owns CPU).
    openblas_threads(1);

    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "eval") {
        cmd_eval(&args);
        return;
    }

    if has_flag(&args, "pretrain") {
        cmd_pretrain(&args);
        return;
    }

    if has_flag(&args, "rl") {
        cmd_rl(&args);
        return;
    }

    println!("Usage:");
    println!("  novacana-ml-v6 pretrain [options]");
    println!("  novacana-ml-v6 rl [options]");
    println!("  novacana-ml-v6 eval --model <path> [--games 400] [--players 4]");
    println!();
    println!("Pretrain options (imitation from AdvCPU):");
    println!("  --out <path>          output model path [models/astra6_pretrain.bin]");
    println!("  --games <n>           games to generate [20000]");
    println!("  --players <n>         players per game [4]");
    println!("  --lr <f>              learning rate [0.0003]");
    println!("  --value-epochs <n>    value head epochs [5]");
    println!("  --policy-epochs <n>   policy head epochs [5]");
    println!("  --cand-epochs <n>     candidate head epochs [5]");
    println!("  --batch-size <n>      batch size [512]");
    println!();
    println!("RL options:");
    println!("  --model <path>        starting model (omit to train from scratch)");
    println!("  --out <path>          output model path [models/net_v6.bin]");
    println!("  --iters <n>           RL iterations [300]");
    println!("  --games <n>           games per iter [2000]");
    println!("  --players <n>         players per game [4]");
    println!("  --lr <f>              learning rate [0.0001]");
    println!("  --temp-start <f>      initial temperature [1.0]");
    println!("  --temp-final <f>      final temperature [0.3]");
    println!("  --entropy-start <f>   initial entropy coeff [0.01]");
    println!("  --entropy-final <f>   final entropy coeff [0.001]");
    println!("  --ppo-epochs <n>      PPO gradient passes per iter [4]");
    println!("  --batch-size <n>      batch size [512]");
    println!("  --clip-eps <f>        PPO clip epsilon [0.2]");
    println!("  --snapshot-every <n>  iters between snapshots [20]");
    println!("  --pool-cap <n>        max snapshot pool size [10]");
    println!("  --cand-lr-mult <f>    candidate head LR multiplier [3.0]");
    println!("  --per-alpha <f>       PER priority exponent [0.6]");
    println!("  --per-beta <f>        PER IS weight exponent start [0.4]");
    println!("  --per-size <n>        PER buffer max size [500000]");
    println!("  --eval-every <n>      eval frequency in iters [50]");
    println!("  --eval-games <n>      games per eval [400]");
    println!("  --gamma <f>           discount factor [0.99]");
    println!("  --lambda <f>          GAE lambda [0.95]");
}

// ── Pretraining loop (generic over backend) ───────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_pretrain_burn<B: AutodiffBackend>(
    initial_weights: WeightsV6,
    out_path:        &str,
    num_games:       usize,
    num_players:     usize,
    lr:              f32,
    value_epochs:    usize,
    policy_epochs:   usize,
    cand_epochs:     usize,
    batch_size:      usize,
    device:          &B::Device,
) {
    println!("Generating imitation data: {} games × {} players...", num_games, num_players);
    let t0 = Instant::now();
    openblas_threads(1);
    let samples = generate_imitate_data(num_games, num_players);
    println!("  {} total samples in {:.1}s", samples.len(), t0.elapsed().as_secs_f32());

    let policy_idx: Vec<usize> = samples.iter().enumerate()
        .filter(|(_, s)| s.has_policy).map(|(i, _)| i).collect();
    let cand_idx: Vec<usize> = samples.iter().enumerate()
        .filter(|(_, s)| s.has_cand).map(|(i, _)| i).collect();
    println!("  Value samples: {}  Policy samples: {}  Cand samples: {}",
        samples.len(), policy_idx.len(), cand_idx.len());

    let mut net  = NetV6Burn::<B>::from_weights(&initial_weights, device);
    let mut opt  = make_optimizer::<B>();
    let mut rng  = thread_rng();

    openblas_threads(16);

    // ── Value pretraining ──────────────────────────────────────────────────────
    println!("Pretraining value ({} epochs, lr={})...", value_epochs, lr);
    for epoch in 0..value_epochs {
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);
        let mut total_loss = 0.0f64;
        let mut nb = 0usize;
        for chunk in indices.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:    Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let outcomes: Vec<f32>      = chunk.iter().map(|&i| samples[i].outcome).collect();
            let (nn, loss) = pretrain_value_step_burn(net, &mut opt, lr as f64, &feats, &outcomes, device);
            net = nn; total_loss += loss as f64; nb += 1;
        }
        println!("  value epoch {}/{}: loss={:.4}", epoch + 1, value_epochs, total_loss / nb.max(1) as f64);
    }

    // ── Policy pretraining ─────────────────────────────────────────────────────
    println!("Pretraining policy ({} epochs)...", policy_epochs);
    for epoch in 0..policy_epochs {
        let mut idx = policy_idx.clone();
        idx.shuffle(&mut rng);
        let mut total_loss = 0.0f64;
        let mut nb = 0usize;
        for chunk in idx.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:   Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let masks:   Vec<u64>      = chunk.iter().map(|&i| samples[i].legal_mask).collect();
            let actions: Vec<u8>       = chunk.iter().map(|&i| samples[i].action).collect();
            let (nn, loss) = pretrain_policy_step_burn(net, &mut opt, lr as f64, &feats, &masks, &actions, device);
            net = nn; total_loss += loss as f64; nb += 1;
        }
        println!("  policy epoch {}/{}: loss={:.4}", epoch + 1, policy_epochs, total_loss / nb.max(1) as f64);
    }

    // ── Candidate pretraining ──────────────────────────────────────────────────
    println!("Pretraining candidates ({} epochs)...", cand_epochs);
    for epoch in 0..cand_epochs {
        let mut idx = cand_idx.clone();
        idx.shuffle(&mut rng);
        let mut total_loss = 0.0f64;
        let mut nb = 0usize;
        for chunk in idx.chunks(batch_size) {
            if chunk.len() < 2 { continue; }
            let feats:  Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
            let cands:  Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].cand_feats.clone()).collect();
            let chosen: Vec<u8>       = chunk.iter().map(|&i| samples[i].cand_chosen).collect();
            let (nn, loss) = cand_step_burn(net, &mut opt, lr as f64, &feats, &cands, &chosen, device);
            net = nn; total_loss += loss as f64; nb += 1;
        }
        println!("  cand epoch {}/{}: loss={:.4}", epoch + 1, cand_epochs, total_loss / nb.max(1) as f64);
    }

    openblas_threads(1);
    let weights = net.to_weights();
    save_weights_v6(&weights, out_path)
        .unwrap_or_else(|e| eprintln!("Warning: save failed: {}", e));
    println!("Pretrain complete. Model saved → {}", out_path);
}

fn cmd_pretrain(args: &[String]) {
    let out_path      = flag_str(args, "--out", "models/astra6_pretrain.bin");
    let num_games     = flag_usize(args, "--games", 20_000);
    let num_players   = flag_usize(args, "--players", 4);
    let lr            = flag_f32(args, "--lr", 0.0003);
    let value_epochs  = flag_usize(args, "--value-epochs", 5);
    let policy_epochs = flag_usize(args, "--policy-epochs", 5);
    let cand_epochs   = flag_usize(args, "--cand-epochs", 5);
    let batch_size    = flag_usize(args, "--batch-size", 512);

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();

    let initial_weights = NetV6::new().to_weights();

    #[cfg(feature = "gpu")]
    {
        println!("Backend: WGPU (GPU)");
        let device = WgpuDevice::default();
        run_pretrain_burn::<Autodiff<Wgpu>>(
            initial_weights, &out_path, num_games, num_players, lr,
            value_epochs, policy_epochs, cand_epochs, batch_size, &device,
        );
        return;
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("Backend: NdArray (CPU)");
        let device = NdArrayDevice::Cpu;
        run_pretrain_burn::<Autodiff<NdArray>>(
            initial_weights, &out_path, num_games, num_players, lr,
            value_epochs, policy_epochs, cand_epochs, batch_size, &device,
        );
    }
}

fn cmd_eval(args: &[String]) {
    let model_path = flag_str(args, "--model", "models/net_v6.bin");
    let num_games  = flag_usize(args, "--games", 400);
    let num_players = flag_usize(args, "--players", 4);

    let weights = load_weights_v6(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));
    let net = Arc::new(NetV6::from_weights(&weights));

    let t0 = Instant::now();
    let win_rate = eval_v6(&net, num_games, num_players);
    println!("Eval ({} games, {} players): {:.1}%  ({:.1}s)",
        num_games, num_players, win_rate * 100.0, t0.elapsed().as_secs_f32());
}

// ── Burn training loop (generic over backend) ─────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_rl_burn<B: AutodiffBackend>(
    initial_weights: WeightsV6,
    out_path:        &str,
    num_iters:       usize,
    games_per:       usize,
    num_players:     usize,
    lr:              f32,
    temp_start:      f32,
    temp_final:      f32,
    ent_start:       f32,
    ent_final:       f32,
    ppo_epochs:      usize,
    batch_size:      usize,
    clip_eps:        f32,
    snapshot_every:  usize,
    pool_cap:        usize,
    per_alpha:       f32,
    per_beta:        f32,
    per_size:        usize,
    eval_every:      usize,
    eval_games:      usize,
    gamma:           f32,
    lambda:          f32,
    cand_lr_mult:    f32,
    device:          &B::Device,
) {
    let mut net_burn = NetV6Burn::<B>::from_weights(&initial_weights, device);
    let mut opt      = make_optimizer::<B>();
    // ndarray weights: kept in sync after each iter for game gen, eval, and checkpointing
    let mut ndarray_weights = initial_weights;

    let mut snapshot_pool: Vec<Arc<WeightsV6>> = Vec::new();
    let mut per = PerBuffer::new(per_size, per_alpha, per_beta);
    let mut rng = thread_rng();

    println!(
        "RL v6 (PPO+PER+burn): {} iters × {} games × {} ppo-epochs  \
         T={:.2}→{:.2}  ent={:.4}→{:.4}  lr={}  snapshot_every={}",
        num_iters, games_per, ppo_epochs,
        temp_start, temp_final, ent_start, ent_final,
        lr, snapshot_every
    );

    for iter in 0..num_iters {
        // Save snapshot before training this iter (weights from previous iter)
        if iter > 0 && iter % snapshot_every == 0 {
            snapshot_pool.push(Arc::new(ndarray_weights.clone()));
            if snapshot_pool.len() > pool_cap {
                snapshot_pool.remove(0);
            }
            println!("  [snapshot saved — pool size {}]", snapshot_pool.len());
        }

        let frac          = iter as f32 / num_iters.saturating_sub(1).max(1) as f32;
        let temperature   = temp_start + (temp_final   - temp_start) * frac;
        let entropy_coeff = ent_start  + (ent_final    - ent_start)  * frac;

        let t0 = Instant::now();

        // Game generation: BLAS single-threaded, rayon owns cores
        openblas_threads(1);
        let current_weights = Arc::new(ndarray_weights.clone());
        let new_samples = run_self_play_v6(
            games_per, num_players, current_weights, &snapshot_pool,
            temperature, gamma, lambda,
        );
        let new_count = new_samples.len();
        per.push_batch(new_samples);

        // Training: burn backend (WGPU or NdArray)
        openblas_threads(16);

        let mut total_policy_loss = 0.0f64;
        let mut total_value_loss  = 0.0f64;
        let mut num_batches       = 0usize;

        let per_len = per.len();
        if per_len >= batch_size {
            for _ppo in 0..ppo_epochs {
                let sample_n = per_len.min(batch_size * 32);
                let (indices, is_weights) = per.sample_indices(sample_n);
                let mut order: Vec<usize> = (0..indices.len()).collect();
                order.shuffle(&mut rng);

                for chunk in order.chunks(batch_size) {
                    if chunk.len() < 2 { continue; }

                    let mut feats:    Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
                    let mut masks:    Vec<u64>      = Vec::with_capacity(chunk.len());
                    let mut actions:  Vec<u8>       = Vec::with_capacity(chunk.len());
                    let mut outcomes: Vec<f32>      = Vec::with_capacity(chunk.len());
                    let mut old_lps:  Vec<f32>      = Vec::with_capacity(chunk.len());
                    let mut is_w:     Vec<f32>      = Vec::with_capacity(chunk.len());

                    for &j in chunk {
                        let buf_idx = indices[j];
                        let s = &per.samples[buf_idx];
                        if s.legal_mask & (1u64 << s.action) == 0 { continue; }
                        feats.push(s.features.clone());
                        masks.push(s.legal_mask);
                        actions.push(s.action);
                        outcomes.push(s.outcome);
                        old_lps.push(s.log_prob);
                        is_w.push(is_weights[j]);
                    }
                    if feats.is_empty() { continue; }

                    let (nb, pl, vl) = rl_train_step_burn(
                        net_burn, &mut opt, lr as f64,
                        &feats, &masks, &actions, &outcomes, &old_lps, &is_w,
                        entropy_coeff, clip_eps, device,
                    );
                    net_burn = nb;
                    total_policy_loss += pl as f64;
                    total_value_loss  += vl as f64;
                    num_batches += 1;

                    // Candidate head update: REINFORCE on samples that had multi-candidate choices.
                    // Advantage = outcome - 0.5 (constant baseline for 4-player: E[win]=0.25).
                    let cand_feats_b: Vec<Vec<f32>> = chunk.iter()
                        .map(|&j| &per.samples[indices[j]])
                        .filter(|s| !s.cand_feats.is_empty())
                        .map(|s| s.features.clone())
                        .collect();
                    let cand_data_b: Vec<Vec<f32>> = chunk.iter()
                        .map(|&j| &per.samples[indices[j]])
                        .filter(|s| !s.cand_feats.is_empty())
                        .map(|s| s.cand_feats.clone())
                        .collect();
                    let cand_chosen_b: Vec<u8> = chunk.iter()
                        .map(|&j| &per.samples[indices[j]])
                        .filter(|s| !s.cand_feats.is_empty())
                        .map(|s| s.cand_chosen)
                        .collect();
                    if cand_feats_b.len() >= 2 {
                        let (nb2, _) = cand_step_burn(
                            net_burn, &mut opt, (lr * cand_lr_mult) as f64,
                            &cand_feats_b, &cand_data_b, &cand_chosen_b, device,
                        );
                        net_burn = nb2;
                    }
                }

                let td_errors: Vec<f32> = indices.iter()
                    .map(|&i| per.samples[i].td_error)
                    .collect();
                per.update_priorities(&indices, &td_errors);
            }
        }

        per.anneal_beta(1.0, 1.0 / num_iters as f32);

        // Sync burn → ndarray weights (one GPU→CPU copy per iter)
        ndarray_weights = net_burn.to_weights();

        let elapsed = t0.elapsed().as_secs_f32();
        let avg_pl = if num_batches > 0 { total_policy_loss / num_batches as f64 } else { 0.0 };
        let avg_vl = if num_batches > 0 { total_value_loss  / num_batches as f64 } else { 0.0 };

        println!(
            "Iter {:4}  new={:6}  buf={:7}  policy_loss={:+.4}  value_loss={:.4}  T={:.2}  ent={:.4}  pool={}  ({:.1}s)",
            iter + 1, new_count, per.len(),
            avg_pl, avg_vl, temperature, entropy_coeff,
            snapshot_pool.len(), elapsed,
        );

        openblas_threads(1);
        if let Err(e) = save_weights_v6(&ndarray_weights, out_path) {
            eprintln!("Warning: failed to save {}: {}", out_path, e);
        }

        if eval_every > 0 && (iter + 1) % eval_every == 0 {
            let net_arc = Arc::new(NetV6::from_weights(&ndarray_weights));
            let win_rate = eval_v6(&net_arc, eval_games, num_players);
            println!("  Eval @ iter {}: {:.1}% win rate vs {} AdvCPU  ({} games)",
                iter + 1, win_rate * 100.0, num_players - 1, eval_games);
        }
    }

    println!("Training complete. Model saved → {}", out_path);
    openblas_threads(1);
}

fn cmd_rl(args: &[String]) {
    let model_path     = flag_str(args, "--model", "");
    let out_path       = flag_str(args, "--out", "models/net_v6.bin");
    let num_iters      = flag_usize(args, "--iters", 300);
    let games_per      = flag_usize(args, "--games", 2000);
    let num_players    = flag_usize(args, "--players", 4);
    let lr             = flag_f32(args, "--lr", 0.0001);
    let temp_start     = flag_f32(args, "--temp-start", 1.0);
    let temp_final     = flag_f32(args, "--temp-final", 0.3);
    let ent_start      = flag_f32(args, "--entropy-start", 0.01);
    let ent_final      = flag_f32(args, "--entropy-final", 0.001);
    let ppo_epochs     = flag_usize(args, "--ppo-epochs", 4);
    let batch_size     = flag_usize(args, "--batch-size", 512);
    let clip_eps       = flag_f32(args, "--clip-eps", 0.2);
    let snapshot_every = flag_usize(args, "--snapshot-every", 20);
    let pool_cap       = flag_usize(args, "--pool-cap", 10);
    let per_alpha      = flag_f32(args, "--per-alpha", 0.6);
    let per_beta       = flag_f32(args, "--per-beta", 0.4);
    let per_size       = flag_usize(args, "--per-size", 500_000);
    let cand_lr_mult   = flag_f32(args, "--cand-lr-mult", 3.0);
    let eval_every     = flag_usize(args, "--eval-every", 50);
    let eval_games     = flag_usize(args, "--eval-games", 400);
    let gamma          = flag_f32(args, "--gamma", 0.99);
    let lambda         = flag_f32(args, "--lambda", 0.95);

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();

    let initial_weights = if !model_path.is_empty() {
        match load_weights_v6(&model_path) {
            Ok(w)  => { println!("Starting from {}", model_path); w }
            Err(e) => { eprintln!("Warning: could not load {}: {} — starting fresh", model_path, e); NetV6::new().to_weights() }
        }
    } else {
        println!("Starting from scratch (random weights)");
        NetV6::new().to_weights()
    };

    #[cfg(feature = "gpu")]
    {
        println!("Backend: WGPU (GPU)");
        let device = WgpuDevice::default();
        run_rl_burn::<Autodiff<Wgpu>>(
            initial_weights, &out_path,
            num_iters, games_per, num_players, lr,
            temp_start, temp_final, ent_start, ent_final,
            ppo_epochs, batch_size, clip_eps,
            snapshot_every, pool_cap,
            per_alpha, per_beta, per_size,
            eval_every, eval_games, gamma, lambda,
            cand_lr_mult, &device,
        );
        return;
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("Backend: NdArray (CPU)");
        let device = NdArrayDevice::Cpu;
        run_rl_burn::<Autodiff<NdArray>>(
            initial_weights, &out_path,
            num_iters, games_per, num_players, lr,
            temp_start, temp_final, ent_start, ent_final,
            ppo_epochs, batch_size, clip_eps,
            snapshot_every, pool_cap,
            per_alpha, per_beta, per_size,
            eval_every, eval_games, gamma, lambda,
            cand_lr_mult, &device,
        );
    }
}
