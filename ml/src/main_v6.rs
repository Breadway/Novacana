mod action;
mod candidate;
mod feature;
mod feature_v6;
mod mcts;
mod generate;
mod net;
mod net_v6;
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

use candidate::{CAND_DIM, encode_card_candidate, encode_system_candidate, encode_target_candidate};
use feature_v6::{encode_v6, FEATURE_DIM_V6};
use net_v6::{NetV6, OptimizerV6, WeightsV6, RlBatchV6, rl_train_step_v6, INPUT_DIM};
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

    if has_flag(&args, "rl") {
        cmd_rl(&args);
        return;
    }

    println!("Usage:");
    println!("  novacana-ml-v6 rl [options]");
    println!("  novacana-ml-v6 eval --model <path> [--games 400] [--players 4]");
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
    println!("  --per-size <n>        PER buffer max size [200000]");
    println!("  --eval-every <n>      eval frequency in iters [50]");
    println!("  --eval-games <n>      games per eval [400]");
    println!("  --gamma <f>           discount factor [0.99]");
    println!("  --lambda <f>          GAE lambda [0.95]");
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

fn cmd_rl(args: &[String]) {
    let model_path      = flag_str(args, "--model", "");
    let out_path        = flag_str(args, "--out", "models/net_v6.bin");
    let num_iters       = flag_usize(args, "--iters", 300);
    let games_per       = flag_usize(args, "--games", 2000);
    let num_players     = flag_usize(args, "--players", 4);
    let lr              = flag_f32(args, "--lr", 0.0001);
    let temp_start      = flag_f32(args, "--temp-start", 1.0);
    let temp_final      = flag_f32(args, "--temp-final", 0.3);
    let ent_start       = flag_f32(args, "--entropy-start", 0.01);
    let ent_final       = flag_f32(args, "--entropy-final", 0.001);
    let ppo_epochs      = flag_usize(args, "--ppo-epochs", 4);
    let batch_size      = flag_usize(args, "--batch-size", 512);
    let clip_eps        = flag_f32(args, "--clip-eps", 0.2);
    let snapshot_every  = flag_usize(args, "--snapshot-every", 20);
    let pool_cap        = flag_usize(args, "--pool-cap", 10);
    let cand_lr_mult    = flag_f32(args, "--cand-lr-mult", 3.0);
    let per_alpha       = flag_f32(args, "--per-alpha", 0.6);
    let per_beta        = flag_f32(args, "--per-beta", 0.4);
    let per_size        = flag_usize(args, "--per-size", 200_000);
    let eval_every      = flag_usize(args, "--eval-every", 50);
    let eval_games      = flag_usize(args, "--eval-games", 400);
    let gamma           = flag_f32(args, "--gamma", 0.99);
    let lambda          = flag_f32(args, "--lambda", 0.95);

    std::fs::create_dir_all(
        std::path::Path::new(&out_path).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();

    let mut net = if !model_path.is_empty() {
        match load_weights_v6(&model_path) {
            Ok(w) => {
                println!("Starting from {}", model_path);
                NetV6::from_weights(&w)
            }
            Err(e) => {
                eprintln!("Warning: could not load {}: {} — starting fresh", model_path, e);
                NetV6::new()
            }
        }
    } else {
        println!("Starting from scratch (random weights)");
        NetV6::new()
    };

    let mut opt = OptimizerV6::new(lr, cand_lr_mult as f32);
    let mut snapshot_pool: Vec<Arc<WeightsV6>> = Vec::new();
    let mut per = PerBuffer::new(per_size, per_alpha, per_beta);
    let mut rng = thread_rng();

    println!(
        "RL v6 (PPO+PER): {} iters × {} games × {} ppo-epochs  \
         T={:.2}→{:.2}  ent={:.4}→{:.4}  lr={}  cand_lr_mult={}  snapshot_every={}",
        num_iters, games_per, ppo_epochs,
        temp_start, temp_final, ent_start, ent_final,
        lr, cand_lr_mult, snapshot_every
    );

    for iter in 0..num_iters {
        // Save snapshot before training this iter
        if iter > 0 && iter % snapshot_every == 0 {
            snapshot_pool.push(Arc::new(net.to_weights()));
            if snapshot_pool.len() > pool_cap {
                snapshot_pool.remove(0);
            }
            println!("  [snapshot saved — pool size {}]", snapshot_pool.len());
        }

        // Linear temperature and entropy schedules
        let frac     = iter as f32 / num_iters.saturating_sub(1).max(1) as f32;
        let temperature = temp_start + (temp_final - temp_start) * frac;
        let entropy_coeff = ent_start + (ent_final - ent_start) * frac;

        let t0 = Instant::now();

        // Game generation: BLAS single-threaded, rayon owns cores
        openblas_threads(1);
        let current_weights = Arc::new(net.to_weights());
        let new_samples = run_self_play_v6(
            games_per, num_players, current_weights, &snapshot_pool,
            temperature, gamma, lambda,
        );
        let new_count = new_samples.len();
        per.push_batch(new_samples);

        // Training: BLAS multi-threaded, single-threaded caller
        openblas_threads(16);

        let mut total_policy_loss = 0.0f64;
        let mut total_value_loss  = 0.0f64;
        let mut num_batches       = 0usize;

        let per_len = per.len();
        if per_len >= batch_size {
            for _ppo in 0..ppo_epochs {
                // Sample a large working set proportional to priority, then mini-batch it.
                let sample_n = per_len.min(batch_size * 32);
                let (indices, is_weights) = per.sample_indices(sample_n);
                let mut order: Vec<usize> = (0..indices.len()).collect();
                order.shuffle(&mut rng);

                for chunk in order.chunks(batch_size) {
                    if chunk.len() < 2 { continue; }

                    // Build batch, skipping any sample whose recorded action isn't in its mask
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

                    let batch = RlBatchV6 {
                        features:      &feats,
                        masks:         &masks,
                        actions:       &actions,
                        outcomes:      &outcomes,
                        old_log_probs: &old_lps,
                        is_weights:    &is_w,
                    };
                    let (pl, vl) = rl_train_step_v6(&mut net, &mut opt, &batch, entropy_coeff, clip_eps);
                    total_policy_loss += pl as f64;
                    total_value_loss  += vl as f64;
                    num_batches += 1;
                }

                // Refresh PER priorities using stored td_error (proxy for |advantage|)
                let td_errors: Vec<f32> = indices.iter()
                    .map(|&i| per.samples[i].td_error)
                    .collect();
                per.update_priorities(&indices, &td_errors);
            }
        }

        // Anneal PER beta toward 1.0
        per.anneal_beta(1.0, 1.0 / num_iters as f32);

        let elapsed = t0.elapsed().as_secs_f32();
        let avg_pl = if num_batches > 0 { total_policy_loss / num_batches as f64 } else { 0.0 };
        let avg_vl = if num_batches > 0 { total_value_loss  / num_batches as f64 } else { 0.0 };

        println!(
            "Iter {:4}  new={:6}  buf={:7}  policy_loss={:+.4}  value_loss={:.4}  T={:.2}  ent={:.4}  pool={}  ({:.1}s)",
            iter + 1, new_count, per.len(),
            avg_pl, avg_vl, temperature, entropy_coeff,
            snapshot_pool.len(), elapsed,
        );

        // Save checkpoint
        openblas_threads(1);
        if let Err(e) = save_weights_v6(&net.to_weights(), &out_path) {
            eprintln!("Warning: failed to save {}: {}", out_path, e);
        }

        // Periodic eval
        if eval_every > 0 && (iter + 1) % eval_every == 0 {
            let net_arc = Arc::new(NetV6::from_weights(&net.to_weights()));
            let win_rate = eval_v6(&net_arc, eval_games, num_players);
            println!("  Eval @ iter {}: {:.1}% win rate vs {} AdvCPU  ({} games)",
                iter + 1, win_rate * 100.0, num_players - 1, eval_games);
        }
    }

    println!("Training complete. Model saved → {}", out_path);
    openblas_threads(1);
}
