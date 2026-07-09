mod net;
mod recorder;
mod tokens;

use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
#[cfg(not(feature = "tch"))]
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};

use burn::tensor::backend::Backend;
use net::Net;
use recorder::{generate_samples, Sample};
use tokens::{action_kind_name, TokenSeq, NUM_ACTION_KINDS};

// ── CLI plumbing ─────────────────────────────────────────────────────────

fn flag_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn flag_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn flag_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or_else(|| default.to_string())
}
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "bc") {
        cmd_bc(&args);
        return;
    }
    if has_flag(&args, "fidelity") {
        cmd_fidelity(&args);
        return;
    }

    println!("Usage:");
    println!("  novacana-ml7 bc [options]");
    println!("  novacana-ml7 fidelity --model <path> [options]");
    println!();
    println!("BC (behavior cloning) options:");
    println!("  --out <path>          [models/astra7_bc.bin]");
    println!("  --model <path>        warm-start weights");
    println!("  --games <n>           pregenerate this many games upfront, reuse across epochs [0]");
    println!("  --chunk-games <n>     stream this many fresh games per epoch instead [2000]");
    println!("  --players <n>         [4]");
    println!("  --lr <f>              [0.0003]");
    println!("  --epochs <n>          [15]");
    println!("  --batch-size <n>      [64]");
    println!("  --label-smoothing <f> [0.05]");
    println!("  --val-games <n>       held-out games for best-checkpoint selection [200]");
    println!();
    println!("Fidelity options (per-ActionKind accuracy breakdown, ASTRA7.md §1.1/§10):");
    println!("  --model <path>        checkpoint to evaluate [models/astra7_bc_best.bin]");
    println!("  --val-games <n>       fresh games to evaluate against [200]");
    println!("  --players <n>         [4]");
    println!("  --batch-size <n>      [64]");
}

// ── BC training ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_bc<B: AutodiffBackend>(
    initial: Option<Net<B>>,
    out_path: &str,
    num_games: usize,
    chunk_games: usize,
    num_players: usize,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    label_smoothing: f32,
    val_games: usize,
    device: &B::Device,
) {
    let mut net = initial.unwrap_or_else(|| Net::fresh(device));
    let mut opt = net::make_optimizer::<B>();
    let mut rng = thread_rng();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    // chunk_games == 0: generate all upfront and reuse across epochs (fine
    // for small demo runs). chunk_games > 0: fresh games every epoch —
    // ASTRA7.md §8.3: never pregenerate more than ~8-10K games at once.
    let pregenerated: Option<Vec<Sample>> = if chunk_games == 0 {
        println!("Generating BC data: {} games x {} players...", num_games, num_players);
        let t0 = Instant::now();
        let s = generate_samples(num_games, num_players);
        println!("  {} samples in {:.1}s", s.len(), t0.elapsed().as_secs_f32());
        Some(s)
    } else {
        println!("Chunked BC training: {} games/epoch x {} players", chunk_games, num_players);
        None
    };

    let val_samples: Vec<Sample> = if val_games > 0 {
        println!("Generating validation set: {} games...", val_games);
        let t0 = Instant::now();
        let v = generate_samples(val_games, num_players);
        println!("  {} val samples in {:.1}s", v.len(), t0.elapsed().as_secs_f32());
        v
    } else {
        Vec::new()
    };

    // LR warmup + cosine decay (ASTRA7.md §10: "cheap insurance" for the
    // scaled rented-GPU run). Total step count is an estimate (chunked mode
    // regenerates slightly different sample counts each epoch) using the
    // val-set's samples/game ratio as a stand-in for train data density —
    // exact step parity doesn't matter for a cosine schedule, only the
    // rough shape does. 5% warmup, floor of 50 steps so tiny smoke runs
    // don't warmup forever.
    let samples_per_game_est = if val_games > 0 {
        (val_samples.len() as f32 / val_games as f32).max(1.0)
    } else {
        1.0
    };
    let games_per_epoch_est = if chunk_games == 0 { num_games } else { chunk_games };
    let steps_per_epoch_est =
        ((games_per_epoch_est as f32 * samples_per_game_est) / batch_size as f32).ceil().max(1.0) as usize;
    let total_steps_est = (steps_per_epoch_est * epochs).max(1);
    let warmup_steps = (total_steps_est / 20).max(50);
    let mut global_step: usize = 0;

    let mut best_acc: f32 = -1.0;
    // NOTE the underscore, not a dot: burn's FileRecorder uses
    // PathBuf::set_extension() when writing, which REPLACES an existing
    // extension rather than appending — "{out_path}.best" + set_extension
    // ("bin") collapses to "{out_path}.bin", identical to the plain final
    // save below, silently discarding the entire best-checkpoint mechanism
    // (found 2026-07-09 while building the fidelity diagnostic; the running
    // Atlas BC job has been affected since it started — see ASTRA7.md §10).
    let best_path = format!("{out_path}_best");

    for epoch in 0..epochs {
        let fresh: Vec<Sample>;
        let samples: &[Sample] = if let Some(ref s) = pregenerated {
            s
        } else {
            fresh = generate_samples(chunk_games, num_players);
            &fresh
        };

        let mut idx: Vec<usize> = (0..samples.len()).collect();
        idx.shuffle(&mut rng);

        // Progress heartbeat: per-epoch logging alone is blind at this scale
        // (a 300-game epoch can be thousands of batches) — print elapsed
        // time + batch index every HEARTBEAT_EVERY batches so a stalled vs.
        // slow-but-progressing run is distinguishable without guessing.
        const HEARTBEAT_EVERY: usize = 50;
        let total_batches = samples.len().div_ceil(batch_size).max(1);

        let (mut tpl, mut tvl, mut nb) = (0.0f64, 0.0f64, 0usize);
        let t0 = Instant::now();
        for chunk in idx.chunks(batch_size) {
            if chunk.len() < 2 {
                continue;
            }
            let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| samples[i].tokens).collect();
            let targets: Vec<u16> = chunk.iter().map(|&i| samples[i].target).collect();
            let outcomes: Vec<f32> = chunk.iter().map(|&i| samples[i].outcome).collect();
            let lr_mult = if global_step < warmup_steps {
                (global_step + 1) as f32 / warmup_steps as f32
            } else {
                let progress = ((global_step - warmup_steps) as f32
                    / (total_steps_est.saturating_sub(warmup_steps)).max(1) as f32)
                    .min(1.0);
                0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            };
            let cur_lr = (lr * lr_mult) as f64;
            let (nn, pl, vl) = net::bc_train_step(net, &mut opt, cur_lr, &seqs, &targets, &outcomes, label_smoothing, device);
            net = nn;
            tpl += pl as f64;
            tvl += vl as f64;
            nb += 1;
            global_step += 1;
            if nb % HEARTBEAT_EVERY == 0 {
                let elapsed = t0.elapsed().as_secs_f32();
                println!(
                    "    heartbeat: batch {}/{}  policy={:.4} value={:.4}  lr={:.2e}  {:.2}s elapsed  ({:.3}s/batch)",
                    nb, total_batches, tpl / nb as f64, tvl / nb as f64, cur_lr, elapsed, elapsed / nb as f32
                );
            }
        }

        if !val_samples.is_empty() {
            // net.valid() (inner, non-autodiff backend): forward passes on
            // the autodiff backend accumulate graph state that only
            // backward() releases (ASTRA7.md §8.2).
            let vnet = net.valid();
            let vidx: Vec<usize> = (0..val_samples.len()).collect();
            let (mut correct, mut ce, mut n) = (0usize, 0.0f32, 0usize);
            for chunk in vidx.chunks(batch_size.max(1)) {
                let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| val_samples[i].tokens).collect();
                let targets: Vec<u16> = chunk.iter().map(|&i| val_samples[i].target).collect();
                let (c, cs) = net::policy_eval(&vnet, &seqs, &targets, device);
                correct += c;
                ce += cs;
                n += chunk.len();
            }
            let acc = correct as f32 / n.max(1) as f32;
            let vce = ce / n.max(1) as f32;
            let star = if acc > best_acc { "  *best" } else { "" };
            println!(
                "Iter {}/{}: policy={:.4} value={:.4}  Eval @ val_acc={:.4} val_ce={:.4}  ({:.1}s){}",
                epoch + 1, epochs, tpl / nb.max(1) as f64, tvl / nb.max(1) as f64, acc, vce, t0.elapsed().as_secs_f32(), star
            );
            if acc > best_acc {
                best_acc = acc;
                let _ = std::fs::remove_file(format!("{best_path}.bin"));
                if let Err(e) = net.clone().save_file(best_path.clone(), &recorder) {
                    eprintln!("Warning: best-checkpoint save failed: {e:?}");
                }
            }
        } else {
            println!(
                "Iter {}/{}: policy={:.4} value={:.4}  ({:.1}s)",
                epoch + 1, epochs, tpl / nb.max(1) as f64, tvl / nb.max(1) as f64, t0.elapsed().as_secs_f32()
            );
        }
    }

    if let Err(e) = net.save_file(out_path, &recorder) {
        eprintln!("Warning: final save failed: {e:?}");
    }
    println!("BC training complete -> {out_path}.bin (best checkpoint -> {best_path}.bin, val_acc={:.4})", best_acc.max(0.0));
}

// ── Per-ActionKind fidelity diagnostic (ASTRA7.md §1.1/§10) ────────────────
//
// Answers "is chain-response fidelity lagging other decision kinds" with
// real evidence on a saved checkpoint, instead of an aggregate accuracy
// number that could hide exactly that gap. Free to run (forward-only,
// no training) — the recommended step before trusting a BC checkpoint or
// spending money scaling it up.

fn run_fidelity<B: Backend>(model_path: &str, samples: &[Sample], batch_size: usize, device: &B::Device) {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let net = Net::<B>::fresh(device)
        .load_file(model_path.to_string(), &recorder, device)
        .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));

    let idx: Vec<usize> = (0..samples.len()).collect();
    let mut buckets = [(0usize, 0usize, 0.0f32); NUM_ACTION_KINDS];
    for chunk in idx.chunks(batch_size.max(1)) {
        let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| samples[i].tokens).collect();
        let targets: Vec<u16> = chunk.iter().map(|&i| samples[i].target).collect();
        let b = net::policy_eval_by_kind(&net, &seqs, &targets, device);
        for k in 0..NUM_ACTION_KINDS {
            buckets[k].0 += b[k].0;
            buckets[k].1 += b[k].1;
            buckets[k].2 += b[k].2;
        }
    }

    let (mut tot_c, mut tot_n, mut tot_ce) = (0usize, 0usize, 0.0f32);
    println!("\n{:<32} {:>8} {:>10} {:>10}", "ActionKind", "n", "accuracy", "mean_ce");
    println!("{:-<62}", "");
    for k in 0..NUM_ACTION_KINDS {
        let (c, n, ce) = buckets[k];
        if n == 0 {
            continue;
        }
        tot_c += c;
        tot_n += n;
        tot_ce += ce;
        println!(
            "{:<32} {:>8} {:>9.1}% {:>10.4}",
            action_kind_name(k as u8),
            n,
            100.0 * c as f32 / n as f32,
            ce / n as f32
        );
    }
    println!("{:-<62}", "");
    println!(
        "{:<32} {:>8} {:>9.1}% {:>10.4}",
        "TOTAL",
        tot_n,
        100.0 * tot_c as f32 / tot_n.max(1) as f32,
        tot_ce / tot_n.max(1) as f32
    );
    println!(
        "\nNote: ChainResponsePass/ChainResponseCard rows are the ASTRA7.md §1.1 chain-response\n\
         fidelity check — if their accuracy trails other rows by a wide margin, that's the signal\n\
         to un-defer history token event-type tagging (tokens.rs's reserved fields[0] on History\n\
         tokens) rather than growing the model."
    );
}

fn cmd_fidelity(args: &[String]) {
    let model_path = flag_str(args, "--model", "models/astra7_bc_best.bin");
    let val_games = flag_usize(args, "--val-games", 200);
    let num_players = flag_usize(args, "--players", 4);
    let batch_size = flag_usize(args, "--batch-size", 64);

    println!("Fidelity check: model={model_path}  games={val_games}");
    let t0 = Instant::now();
    let samples = generate_samples(val_games, num_players);
    println!("  {} samples in {:.1}s", samples.len(), t0.elapsed().as_secs_f32());

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        run_fidelity::<LibTorch<f32>>(&model_path, &samples, batch_size, &device);
        return;
    }
    #[cfg(not(feature = "tch"))]
    {
        let device = NdArrayDevice::Cpu;
        run_fidelity::<NdArray>(&model_path, &samples, batch_size, &device);
    }
}

fn cmd_bc(args: &[String]) {
    let out_path = flag_str(args, "--out", "models/astra7_bc.bin");
    let num_games = flag_usize(args, "--games", 0);
    let chunk_games = flag_usize(args, "--chunk-games", 2000);
    let num_players = flag_usize(args, "--players", 4);
    let lr = flag_f32(args, "--lr", 0.0003);
    let epochs = flag_usize(args, "--epochs", 15);
    let batch_size = flag_usize(args, "--batch-size", 64);
    let label_smoothing = flag_f32(args, "--label-smoothing", 0.05);
    let val_games = flag_usize(args, "--val-games", 200);

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let model_path = flag_str(args, "--model", "");

    // Backend selection (ASTRA7.md §3.3/§5): `tch` feature -> LibTorch/CUDA
    // on Atlas (Autodiff<LibTorch<f32>>, device Cuda(0)); default ->
    // burn-ndarray CPU, kept so hermes-side quick sanity runs/tests still
    // work without needing Atlas. Mirrors ml/src/main.rs's gpu/not(gpu) split.
    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        type B = Autodiff<LibTorch<f32>>;

        let initial: Option<Net<B>> = if !model_path.is_empty() {
            println!("Warm start from {model_path}");
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            Some(
                Net::<B>::fresh(&device)
                    .load_file(model_path.clone(), &recorder, &device)
                    .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}")),
            )
        } else {
            None
        };

        run_bc::<B>(
            initial, &out_path, num_games, chunk_games, num_players, lr, epochs, batch_size, label_smoothing,
            val_games, &device,
        );
        return;
    }
    #[cfg(not(feature = "tch"))]
    {
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray>;

        let initial: Option<Net<B>> = if !model_path.is_empty() {
            println!("Warm start from {model_path}");
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            Some(
                Net::<B>::fresh(&device)
                    .load_file(model_path.clone(), &recorder, &device)
                    .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}")),
            )
        } else {
            None
        };

        run_bc::<B>(
            initial, &out_path, num_games, chunk_games, num_players, lr, epochs, batch_size, label_smoothing,
            val_games, &device,
        );
    }
}
