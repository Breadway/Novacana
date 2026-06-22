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

use std::sync::Arc;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use novacana_engine::{
    cpu::AdvancedCpuPlayer,
    deck::Deck,
    game::{GameEngine, WinCondition},
    player::Player,
};

use generate::{load_candidate_samples, load_samples, run_games_all, save_candidate_samples, save_samples};
use net::{Net, Optimizer, RlBatch, Weights, rl_train_step};
use mcts::MctsPlayer;
use player::MlPlayer;
use self_play::run_self_play;
use train::{load_weights, pretrain_candidates, pretrain_value, save_weights, TrainConfig};

fn main() {
    // Prevent OpenBLAS from spawning per-call thread pools; rayon owns parallelism.
    unsafe { std::env::set_var("OPENBLAS_NUM_THREADS", "1"); }
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "generate" | "gen"    => cmd_generate(&args),
        "train"               => cmd_train(&args),
        "pretrain-candidates" => cmd_pretrain_candidates(&args),
        "pretrain-value"      => cmd_pretrain_value(&args),
        "eval"                => cmd_eval(&args),
        "rl"                  => cmd_rl(&args),
        "pbt"                 => cmd_pbt(&args),
        _                     => print_help(),
    }
}

// ── generate ──────────────────────────────────────────────────────────────────

fn cmd_generate(args: &[String]) {
    let num_games   = flag_usize(args, "--games",         5_000);
    let num_players = flag_usize(args, "--players",       4);
    let out         = flag_str(args, "--out",             "data/train.bin");
    let cand_out    = flag_str(args, "--cand-out",        "data/candidates.bin");

    println!("Generating {} games ({} players)…", num_games, num_players);
    let t0 = Instant::now();

    let (samples, cand_samples) = run_games_all(num_games, num_players);
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "  {} imitation samples  {} candidate samples  in {:.2}s  ({:.0} games/s)",
        samples.len(), cand_samples.len(),
        elapsed,
        num_games as f64 / elapsed,
    );

    for path in &[&out, &cand_out] {
        std::fs::create_dir_all(
            std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."))
        ).unwrap();
    }

    save_samples(&samples, &out).expect("failed to write samples");
    println!("Imitation samples → {}", out);

    save_candidate_samples(&cand_samples, &cand_out).expect("failed to write candidate samples");
    println!("Candidate samples → {}", cand_out);
}

// ── train ─────────────────────────────────────────────────────────────────────

fn cmd_train(args: &[String]) {
    let data_path = flag_str(args, "--data",   "data/train.bin");
    let out_path  = flag_str(args, "--out",    "models/net.bin");
    let epochs    = flag_usize(args, "--epochs", 50);
    let batch     = flag_usize(args, "--batch",  256);
    let lr        = flag_f32(args, "--lr",    1e-3);

    println!("Loading samples from {}…", data_path);
    let samples = load_samples(&data_path).expect("failed to load samples");
    println!("  {} samples loaded", samples.len());

    let cfg = TrainConfig { epochs, batch_size: batch, lr, out_path };
    let t0 = Instant::now();
    train::train(&samples, &cfg).expect("training failed");
    println!("Training complete in {:.1}s", t0.elapsed().as_secs_f64());
}

// ── eval ──────────────────────────────────────────────────────────────────────

fn cmd_eval(args: &[String]) {
    let model_path  = flag_str(args, "--model",   "models/net.bin");
    let num_games   = flag_usize(args, "--games",  200);
    let num_players = flag_usize(args, "--players", 4);
    let mcts_sims   = flag_usize(args, "--mcts",   0); // 0 = disabled

    let weights = Arc::new(load_weights(&model_path).expect("failed to load model"));
    let mode = if mcts_sims > 0 { format!("MCTS({})", mcts_sims) } else { "greedy".into() };
    println!("Evaluating {} [{}] vs {} AdvancedCpu ({}-player, {} games)…",
        model_path, mode, num_players - 1, num_players, num_games);

    let mut ml_wins    = 0usize;
    let mut cpu_wins   = 0usize;
    let mut cataclysm  = 0usize;
    let mut astro      = 0usize;
    let t0 = Instant::now();

    for game_idx in 0..num_games {
        let w = Arc::clone(&weights);
        let ml_seat = game_idx % num_players;

        let players: Vec<Box<dyn Player>> = (0..num_players)
            .map(|i| -> Box<dyn Player> {
                if i == ml_seat {
                    if mcts_sims > 0 {
                        Box::new(MctsPlayer::new(format!("ML-P{}", i + 1), Arc::clone(&w), mcts_sims))
                    } else {
                        Box::new(MlPlayer::new(format!("ML-P{}", i + 1), Arc::clone(&w)))
                    }
                } else {
                    Box::new(AdvancedCpuPlayer::new(format!("CPU-P{}", i + 1)))
                }
            })
            .collect();

        let mut engine = GameEngine::new(Deck::build_classic(), players);
        engine.set_silent();
        engine.set_max_turns(1000);
        let result = engine.run();

        if result.winner_idx == ml_seat { ml_wins += 1; } else { cpu_wins += 1; }
        match result.condition {
            WinCondition::CosmicCataclysm        => cataclysm += 1,
            WinCondition::AstronomicalDomination => astro += 1,
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    println!("  ML wins  : {}/{} ({:.1}%)", ml_wins, num_games, ml_wins as f64 / num_games as f64 * 100.0);
    println!("  CPU wins : {}/{} ({:.1}%)", cpu_wins, num_games, cpu_wins as f64 / num_games as f64 * 100.0);
    println!("  AstroDom : {}  CataClyasm: {}", astro, cataclysm);
    println!("  Expected random: {:.1}%  ({:.0} games/s)",
        100.0 / num_players as f64, num_games as f64 / elapsed);
}

// ── pretrain-candidates ───────────────────────────────────────────────────────

fn cmd_pretrain_candidates(args: &[String]) {
    let data_path = flag_str(args, "--data",   "data/candidates.bin");
    let model     = flag_str(args, "--model",  "models/net.bin");
    let epochs    = flag_usize(args, "--epochs", 30);
    let batch     = flag_usize(args, "--batch",  256);
    let lr        = flag_f32(args, "--lr",     5e-4);

    println!("Loading candidate samples from {}…", data_path);
    let samples = load_candidate_samples(&data_path).expect("failed to load candidate samples");
    println!("  {} candidate samples loaded", samples.len());

    let cfg = TrainConfig { epochs, batch_size: batch, lr, out_path: model };
    pretrain_candidates(&samples, &cfg).expect("candidate pretraining failed");
}

// ── pretrain-value ────────────────────────────────────────────────────────────

fn cmd_pretrain_value(args: &[String]) {
    let data_path = flag_str(args, "--data",   "data/train.bin");
    let model     = flag_str(args, "--model",  "models/net.bin");
    let epochs    = flag_usize(args, "--epochs", 30);
    let batch     = flag_usize(args, "--batch",  256);
    let lr        = flag_f32(args, "--lr",     5e-4);

    println!("Loading samples from {}…", data_path);
    let samples = load_samples(&data_path).expect("failed to load samples");
    println!("  {} samples", samples.len());

    let cfg = TrainConfig { epochs, batch_size: batch, lr, out_path: model };
    pretrain_value(&samples, &cfg).expect("value pretraining failed");
}

// ── rl ────────────────────────────────────────────────────────────────────────

fn cmd_rl(args: &[String]) {
    let in_model       = flag_str(args, "--model",           "models/net.bin");
    let out_model      = flag_str(args, "--out",             "models/rl.bin");
    let num_iters      = flag_usize(args, "--iters",          200);
    let games_per      = flag_usize(args, "--games",          2000);
    let num_players    = flag_usize(args, "--players",        4);
    let batch_size     = flag_usize(args, "--batch",          512);
    let lr             = flag_f32(args, "--lr",             1e-4);
    let temperature    = flag_f32(args, "--temp",           1.0);
    let entropy        = flag_f32(args, "--entropy",        0.01);
    let gamma          = flag_f32(args, "--gamma",          0.99);
    let lambda         = flag_f32(args, "--lambda",         0.95);
    let ppo_epochs     = flag_usize(args, "--ppo-epochs",    4);
    let clip_eps       = flag_f32(args, "--clip-eps",       0.2);
    let snapshot_every = flag_usize(args, "--snapshot-every", 20);
    let pool_cap       = flag_usize(args, "--pool-cap",      10);
    let advcpu_share   = flag_f32(args, "--advcpu-share",  0.3);

    let init = if std::path::Path::new(&in_model).exists() {
        println!("Starting from {}", in_model);
        load_weights(&in_model).expect("failed to load model")
    } else {
        println!("No model at {} — starting from random weights", in_model);
        Net::new().to_weights()
    };

    std::fs::create_dir_all(
        std::path::Path::new(&out_model).parent().unwrap_or(std::path::Path::new("."))
    ).unwrap();

    let mut net = Net::from_weights(&init);
    let mut opt = Optimizer::new(lr);
    let mut rng = thread_rng();
    let mut snapshot_pool: Vec<Arc<Weights>> = Vec::new();

    println!(
        "RL (PPO) self-play: {} iters × {} games × {} ppo-epochs  \
         T={} lr={} entropy={} gamma={} clip_eps={} snapshot_every={}",
        num_iters, games_per, ppo_epochs,
        temperature, lr, entropy, gamma, clip_eps, snapshot_every,
    );

    for iter in 0..num_iters {
        // Save snapshot of current policy to pool before training
        if iter > 0 && iter % snapshot_every == 0 {
            snapshot_pool.push(Arc::new(net.to_weights()));
            if snapshot_pool.len() > pool_cap {
                snapshot_pool.remove(0);
            }
            println!("  [snapshot saved — pool size {}]", snapshot_pool.len());
        }

        let t0 = Instant::now();
        let weights = Arc::new(net.to_weights());
        let samples = run_self_play(games_per, num_players, weights, &snapshot_pool, temperature, gamma, lambda, advcpu_share);

        let mut indices: Vec<usize> = (0..samples.len()).collect();
        let mut total_pl = 0.0f64;
        let mut total_vl = 0.0f64;
        let mut nb = 0usize;

        // Multiple PPO gradient updates over the same batch of experience
        for _ppo in 0..ppo_epochs {
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(batch_size) {
                if chunk.len() < 2 { continue; }
                let feats:         Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
                let masks:         Vec<u64>      = chunk.iter().map(|&i| samples[i].legal_mask).collect();
                let actions:       Vec<u8>       = chunk.iter().map(|&i| samples[i].action).collect();
                let outcomes:      Vec<f32>      = chunk.iter().map(|&i| samples[i].outcome).collect();
                let old_log_probs: Vec<f32>      = chunk.iter().map(|&i| samples[i].log_prob).collect();

                let batch = RlBatch {
                    features:      &feats,
                    masks:         &masks,
                    actions:       &actions,
                    outcomes:      &outcomes,
                    old_log_probs: &old_log_probs,
                };
                let (pl, vl) = rl_train_step(&mut net, &mut opt, &batch, entropy, clip_eps);
                total_pl += pl as f64;
                total_vl += vl as f64;
                nb += 1;
            }
        }

        println!(
            "Iter {:4}  samples={:6}  policy_loss={:+.4}  value_loss={:.4}  pool={}  ({:.1}s)",
            iter + 1, samples.len(),
            if nb > 0 { total_pl / nb as f64 } else { 0.0 },
            if nb > 0 { total_vl / nb as f64 } else { 0.0 },
            snapshot_pool.len(),
            t0.elapsed().as_secs_f64(),
        );

        save_weights(&net.to_weights(), &out_model).expect("failed to save checkpoint");
    }

    println!("RL training complete → {}", out_model);
}

// ── pbt ───────────────────────────────────────────────────────────────────────
//
// Population-Based Training: N agents train in parallel; every `exploit_every`
// iterations the bottom half (by eval win-rate) is replaced with perturbed
// copies of the top half.  Perturbation = multiply each weight by N(1, 0.02).

fn cmd_pbt(args: &[String]) {
    let seed_model    = flag_str(args, "--model",           "models/net.bin");
    let out_dir       = flag_str(args, "--out-dir",         "models/pbt");
    let pop_size      = flag_usize(args, "--pop",            8);
    let num_iters     = flag_usize(args, "--iters",          400);
    let games_per     = flag_usize(args, "--games",          500);
    let eval_games    = flag_usize(args, "--eval-games",     100);
    let num_players   = flag_usize(args, "--players",        4);
    let batch_size    = flag_usize(args, "--batch",          512);
    let lr            = flag_f32(args, "--lr",             1e-4);
    let temperature   = flag_f32(args, "--temp",           1.0);
    let entropy       = flag_f32(args, "--entropy",        0.01);
    let gamma         = flag_f32(args, "--gamma",          0.99);
    let lambda        = flag_f32(args, "--lambda",         0.95);
    let clip_eps      = flag_f32(args, "--clip-eps",       0.2);
    let exploit_every = flag_usize(args, "--exploit-every",  20);
    let advcpu_share  = flag_f32(args, "--advcpu-share",   0.3);
    let perturb_std   = flag_f32(args, "--perturb-std",    0.02);

    std::fs::create_dir_all(&out_dir).unwrap();

    let seed = if std::path::Path::new(&seed_model).exists() {
        load_weights(&seed_model).expect("failed to load seed model")
    } else {
        Net::new().to_weights()
    };

    // Initialise population with small perturbations of the seed
    let mut population: Vec<Weights> = (0..pop_size).map(|_| perturb(&seed, perturb_std)).collect();
    let mut win_rates = vec![0.0f32; pop_size];
    let mut rng = thread_rng();

    println!("PBT: pop={} iters={} exploit_every={}", pop_size, num_iters, exploit_every);

    for iter in 0..num_iters {
        let t0 = Instant::now();

        // Each agent does one RL step against the rest of the population as pool
        for agent_idx in 0..pop_size {
            let current = Arc::new(population[agent_idx].clone());
            let pool: Vec<Arc<Weights>> = population.iter().enumerate()
                .filter(|(i, _)| *i != agent_idx)
                .map(|(_, w)| Arc::new(w.clone()))
                .collect();

            let samples = run_self_play(games_per, num_players, current, &pool,
                                        temperature, gamma, lambda, advcpu_share);
            if samples.is_empty() { continue; }

            let mut net = Net::from_weights(&population[agent_idx]);
            let mut opt = Optimizer::new(lr);
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(batch_size) {
                if chunk.len() < 2 { continue; }
                let feats:    Vec<Vec<f32>> = chunk.iter().map(|&i| samples[i].features.clone()).collect();
                let masks:    Vec<u64>      = chunk.iter().map(|&i| samples[i].legal_mask).collect();
                let acts:     Vec<u8>       = chunk.iter().map(|&i| samples[i].action).collect();
                let outs:     Vec<f32>      = chunk.iter().map(|&i| samples[i].outcome).collect();
                let old_lps:  Vec<f32>      = chunk.iter().map(|&i| samples[i].log_prob).collect();
                let batch = net::RlBatch { features: &feats, masks: &masks, actions: &acts,
                                           outcomes: &outs, old_log_probs: &old_lps };
                net::rl_train_step(&mut net, &mut opt, &batch, entropy, clip_eps);
            }
            population[agent_idx] = net.to_weights();
        }

        // Exploit: every N iters, eval each agent and replace bottom half
        if (iter + 1) % exploit_every == 0 {
            for (agent_idx, w) in population.iter().enumerate() {
                let wts = Arc::new(w.clone());
                let mut wins = 0usize;
                for g in 0..eval_games {
                    let seat = g % num_players;
                    let players: Vec<Box<dyn Player>> = (0..num_players).map(|i| -> Box<dyn Player> {
                        if i == seat {
                            Box::new(MlPlayer::new("eval", Arc::clone(&wts)))
                        } else {
                            Box::new(AdvancedCpuPlayer::new(format!("cpu{}", i)))
                        }
                    }).collect();
                    let mut engine = GameEngine::new(Deck::build_classic(), players);
                    engine.set_silent(); engine.set_max_turns(1000);
                    if engine.run().winner_idx == seat { wins += 1; }
                }
                win_rates[agent_idx] = wins as f32 / eval_games as f32;
            }

            // Sort agents by win rate
            let mut ranked: Vec<usize> = (0..pop_size).collect();
            ranked.sort_by(|&a, &b| win_rates[b].partial_cmp(&win_rates[a]).unwrap());
            let top_half = &ranked[..pop_size / 2];
            let bot_half = &ranked[pop_size / 2..];

            println!("  [exploit iter {}] best={:.1}%  worst={:.1}%",
                iter + 1,
                win_rates[ranked[0]] * 100.0,
                win_rates[ranked[pop_size - 1]] * 100.0);

            // Replace bottom with perturbed copies of top
            for (i, &bot) in bot_half.iter().enumerate() {
                let src = top_half[i % top_half.len()];
                population[bot] = perturb(&population[src], perturb_std);
                win_rates[bot] = win_rates[src]; // inherit parent's rate as prior
            }

            // Save best agent
            let best_path = format!("{}/best.bin", out_dir);
            save_weights(&population[ranked[0]], &best_path).unwrap();
        }

        println!("Iter {:4}  ({:.1}s)", iter + 1, t0.elapsed().as_secs_f64());
    }

    // Save all agents at the end
    for (i, w) in population.iter().enumerate() {
        save_weights(w, &format!("{}/agent_{}.bin", out_dir, i)).unwrap();
    }
    println!("PBT complete → {}/", out_dir);
}

fn perturb(w: &Weights, std: f32) -> Weights {
    fn noise(v: &[f32], s: f32) -> Vec<f32> {
        v.iter().map(|&x| {
            // Multiplicative perturbation: x * (1 + N(0,s))
            let eps: f32 = rand::random::<f32>() * 2.0 - 1.0; // uniform [-1,1] approx
            x * (1.0 + eps * s)
        }).collect()
    }
    Weights {
        w1: noise(&w.w1, std), b1: noise(&w.b1, std),
        w2: noise(&w.w2, std), b2: noise(&w.b2, std),
        w3: noise(&w.w3, std), b3: noise(&w.b3, std),
        w4: noise(&w.w4, std), b4: noise(&w.b4, std),
        wp: noise(&w.wp, std), bp: noise(&w.bp, std),
        wv: noise(&w.wv, std), bv: noise(&w.bv, std),
        wc1: noise(&w.wc1, std), bc1: noise(&w.bc1, std),
        wc2: noise(&w.wc2, std), bc2: noise(&w.bc2, std),
    }
}

// ── CLI helpers ───────────────────────────────────────────────────────────────

fn flag_str<'a>(args: &'a [String], flag: &str, default: &'a str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

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

fn print_help() {
    println!("novacana-ml — ML player pipeline");
    println!();
    println!("COMMANDS");
    println!("  generate             --games N --players N --out PATH --cand-out PATH");
    println!("  train                --data PATH --out PATH --epochs N --batch N --lr F");
    println!("  pretrain-candidates  --data PATH --model PATH --epochs N --batch N --lr F");
    println!("  pretrain-value       --data PATH --model PATH --epochs N --batch N --lr F");
    println!("  eval                 --model PATH --games N --players N");
    println!("  rl                   --model PATH --out PATH --iters N --games N ...");
    println!();
    println!("PIPELINE (v4)");
    println!("  1. generate          → data/train.bin + data/candidates.bin");
    println!("  2. train             --out models/net_v4.bin");
    println!("  3. pretrain-candidates --model models/net_v4.bin");
    println!("  4. pretrain-value    --model models/net_v4.bin");
    println!("  5. rl                --model models/net_v4.bin --out models/rl_v4.bin");
    println!("  6. eval              --model models/rl_v4.bin --games 500");
}
