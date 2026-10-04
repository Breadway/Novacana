mod az;
mod batch_infer;
mod evals;
mod mcts;
mod net;
mod recorder;
mod sampling;
mod spectator;
mod td_value;
mod tokens;

use std::collections::VecDeque;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
#[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(all(not(feature = "tch"), feature = "vulkan"))]
use burn_wgpu::{Wgpu, WgpuDevice};

#[cfg(all(feature = "tch", feature = "vulkan"))]
compile_error!("features `tch` and `vulkan` select different training backends; enable only one");

// Arc A380 (Hestia) does FP16 at twice the FP32 rate, and 6GB cannot hold
// the f32 batch this model wants.
#[cfg(all(not(feature = "tch"), feature = "vulkan"))]
type Gpu = Wgpu<half::f16, i32>;

#[cfg(all(not(feature = "tch"), feature = "vulkan"))]
fn vulkan_device() -> WgpuDevice {
    // Hestia has a discrete Arc GPU; Hermes can use its integrated GPU. Keep
    // the discrete adapter as the default and allow either to be selected at
    // runtime without maintaining separate builds.
    let device = match std::env::var("NOVACANA_WGPU_DEVICE").as_deref() {
        Ok("integrated") => WgpuDevice::IntegratedGpu(0),
        Ok("cpu") => WgpuDevice::Cpu,
        _ => WgpuDevice::DiscreteGpu(0),
    };
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());
    device
}

use az::{eval_win_rate, generate_az_games, scan_training_data_for_collisions, scan_live_play_for_collisions, AzSample};
use burn::tensor::backend::Backend;
use mcts::McConfig;
use net::Net;
use recorder::{generate_samples, Sample};
use td_value::{generate_td_samples, TdSample};
use tokens::{action_kind_name, ACTION_MAX, TokenSeq, NUM_ACTION_KINDS};

// ── CLI plumbing ─────────────────────────────────────────────────────────

/// Sort by sequence length, cut batches, then shuffle the batches.
/// A random mix lets one long decision pad every row; similar lengths keep
/// the compact transformer on the shorter shape.
fn length_grouped(mut idx: Vec<usize>, batch_size: usize, span: impl Fn(usize) -> usize, rng: &mut impl rand::Rng) -> Vec<Vec<usize>> {
    idx.sort_unstable_by_key(|i| span(*i));
    let mut batches: Vec<Vec<usize>> = idx.chunks(batch_size).filter(|chunk| chunk.len() >= 2).map(|chunk| chunk.to_vec()).collect();
    batches.shuffle(rng);
    batches
}

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

// burn's FileRecorder uses PathBuf::set_extension() when writing, which
// REPLACES the extension rather than appending. A plain "{out_path}_best"
// suffix (found 2026-07-09, fidelity diagnostic) still collides whenever
// out_path itself already ends in ".bin": "foo.bin_best" has exactly one
// dot, so set_extension("bin") sees "bin_best" as THE extension and
// replaces it wholesale, collapsing back to "foo.bin" (verified via a
// standalone rustc repro, 2026-07-09 - the underscore fix alone wasn't
// enough, an out_path with an extension retriggers the same collision).
// Fix: strip any existing extension via file_stem() before appending
// "_best", so the constructed path never has more than the one dot
// set_extension() itself adds back. Shared by every checkpoint-saving
// command (bc/td-value/az) - do not reintroduce the bug via a fresh
// string-concat in a new command.
fn best_checkpoint_path(out_path: &str) -> String {
    let out_pb = std::path::Path::new(out_path);
    let stem = out_pb.file_stem().and_then(|s| s.to_str()).unwrap_or(out_path);
    let parent = out_pb.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    if parent.is_empty() {
        format!("{stem}_best")
    } else {
        format!("{parent}/{stem}_best")
    }
}

/// The actual on-disk filename burn's `BinFileRecorder` will use for
/// `path` - mirrors its internal `set_extension("bin")` (REPLACES any
/// existing extension, doesn't append - same gotcha `best_checkpoint_path`
/// works around). Every println! that tells the user what file a
/// checkpoint landed at, or what to pass to `--resume`/`--model`, must go
/// through this rather than hand-formatting `"{path}.bin"` - that guess is
/// wrong whenever `path` already ends in `.bin` (produces a misleading
/// `foo.bin.bin` that doesn't exist - found 2026-07-10 testing --resume).
fn checkpoint_display_path(path: &str) -> String {
    let mut pb = std::path::PathBuf::from(path);
    pb.set_extension("bin");
    pb.to_string_lossy().to_string()
}

// ── Pause/resume support (2026-07-10, brother's-3080 setup) ────────────────
//
// `--model <path>` (existing, unchanged) is a WARM START: load these
// weights but treat it as a brand-new run - LR schedule restarts from
// warmup, the Adam optimizer starts cold, `best_acc` resets to -1.0 (so
// epoch 1 post-"resume" always looks like a new best, silently
// overwriting a checkpoint that may have been genuinely better). That's
// the right behavior for bootstrapping a new architecture/dataset off an
// old checkpoint, but wrong for "the same run got interrupted, continue
// it" - every pause/resume cycle would waste real training time re-
// warming the LR and could regress the saved "best" checkpoint. `--resume
// <path>` below is the real continuation path: same weights, but also
// restores `global_step` (so the cosine LR schedule doesn't restart) and
// `best_acc` (so a worse post-resume epoch can't clobber a better
// pre-pause checkpoint). Optimizer momentum (Adam's per-param state) is
// NOT restored - burn's `Optimizer` trait doesn't expose a simple
// serialization path for it, and losing momentum costs at most a few
// batches of slightly-off step sizes, not worth the complexity here.
struct ResumeMeta {
    epoch: usize,
    global_step: usize,
    best_acc: f32,
}

fn resume_meta_path(checkpoint_path: &str) -> String {
    let pb = std::path::Path::new(checkpoint_path);
    let stem = pb.file_stem().and_then(|s| s.to_str()).unwrap_or(checkpoint_path);
    let parent = pb.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    if parent.is_empty() {
        format!("{stem}.resume.txt")
    } else {
        format!("{parent}/{stem}.resume.txt")
    }
}

fn save_resume_meta(checkpoint_path: &str, meta: &ResumeMeta) {
    let contents = format!("epoch={}\nglobal_step={}\nbest_acc={}\n", meta.epoch, meta.global_step, meta.best_acc);
    if let Err(e) = std::fs::write(resume_meta_path(checkpoint_path), contents) {
        eprintln!("Warning: resume-metadata save failed: {e:?}");
    }
}

fn load_resume_meta(checkpoint_path: &str) -> ResumeMeta {
    let path = resume_meta_path(checkpoint_path);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("--resume requires a sidecar metadata file at {path} (saved automatically alongside every checkpoint since 2026-07-10) - none found: {e:?}")
    });
    let mut epoch = 0usize;
    let mut global_step = 0usize;
    let mut best_acc = -1.0f32;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "epoch" => epoch = v.parse().unwrap_or(0),
                "global_step" => global_step = v.parse().unwrap_or(0),
                "best_acc" => best_acc = v.parse().unwrap_or(-1.0),
                _ => {}
            }
        }
    }
    ResumeMeta { epoch, global_step, best_acc }
}

/// Sibling-file paths for the dashboard's spectator-game/card-spectrum
/// outputs (2026-07-15) - same `{parent}/{stem}...` pattern as
/// `resume_meta_path`, rooted at the checkpoint's own directory so they
/// travel with a run instead of landing in an unrelated cwd.
fn spectator_game_path(out_path: &str) -> String {
    let pb = std::path::Path::new(out_path);
    let parent = pb.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    if parent.is_empty() { "spectator_game.json".to_string() } else { format!("{parent}/spectator_game.json") }
}

fn card_spectrum_path(out_path: &str) -> String {
    let pb = std::path::Path::new(out_path);
    let parent = pb.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    if parent.is_empty() { "card_spectrum.jsonl".to_string() } else { format!("{parent}/card_spectrum.jsonl") }
}

/// Writes both dashboard outputs for one spectator game: the full replay
/// trace (overwritten each time - the dashboard always shows the latest)
/// and one appended line to the running card-spectrum log (so the
/// dashboard can chart how the policy's card preferences evolve across
/// epochs). Never fatal - a write failure here is a dashboard-cosmetics
/// problem, not a training one.
fn write_spectator_outputs(out_path: &str, epoch: usize, game: &spectator::SpectatorGame) {
    match serde_json::to_string(game) {
        Ok(json) => {
            if let Err(e) = std::fs::write(spectator_game_path(out_path), json) {
                eprintln!("Warning: spectator-game write failed: {e:?}");
            }
        }
        Err(e) => eprintln!("Warning: spectator-game serialize failed: {e:?}"),
    }
    let line = serde_json::json!({ "epoch": epoch, "spectrum": game.card_spectrum });
    match std::fs::OpenOptions::new().create(true).append(true).open(card_spectrum_path(out_path)) {
        Ok(mut f) => {
            use std::io::Write;
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("Warning: card-spectrum append failed: {e:?}");
            }
        }
        Err(e) => eprintln!("Warning: card-spectrum open failed: {e:?}"),
    }
}

/// Installs a Ctrl+C/SIGTERM handler (works under WSL2 the same as native
/// Linux - this is what `docker stop`/a "stop training" shortcut sends) and
/// returns a flag the training loop polls once per batch. On trigger, the
/// loop finishes its current (already in-flight) step, saves a checkpoint +
/// resume metadata, and exits cleanly instead of losing everything back to
/// the last periodic/epoch checkpoint. Idempotent-safe to call once per
/// process; a second Ctrl+C while already shutting down just re-sets the
/// same flag (harmless).
fn install_shutdown_flag() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag_clone = std::sync::Arc::clone(&flag);
    if let Err(e) = ctrlc::set_handler(move || {
        eprintln!("\nShutdown requested - finishing current step and saving a checkpoint...");
        flag_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    }) {
        eprintln!("Warning: failed to install shutdown handler ({e:?}) - Ctrl+C/stop will hard-kill instead of checkpointing.");
    }
    flag
}

/// Same collision-avoidance as `best_checkpoint_path`, for AZ's numbered
/// snapshot pool (ASTRA7.md §4.3: "snapshot the model pool every 20
/// iterations") - `{stem}_iter{n}`, never `{out_path}_iter{n}` directly.
fn snapshot_path(out_path: &str, iter: usize) -> String {
    let out_pb = std::path::Path::new(out_path);
    let stem = out_pb.file_stem().and_then(|s| s.to_str()).unwrap_or(out_path);
    let parent = out_pb.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    if parent.is_empty() {
        format!("{stem}_iter{iter}")
    } else {
        format!("{parent}/{stem}_iter{iter}")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "bench") {
        let samples = recorder::benchmark_samples(42);
        let batch = flag_usize(&args, "--batch-size", 2);
        let steps = flag_usize(&args, "--steps", 3);
        let rounds = flag_usize(&args, "--rounds", 3);
        let compact_only = flag_str(&args, "--layout", "both") == "compact";
        #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
        net::benchmark::<Autodiff<NdArray>>(&samples, batch, steps, rounds, compact_only, &NdArrayDevice::Cpu);
        #[cfg(feature = "tch")]
        net::benchmark::<Autodiff<LibTorch<f32>>>(&samples, batch, steps, rounds, compact_only, &LibTorchDevice::Cuda(0));
        #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
        net::benchmark::<Autodiff<Gpu>>(&samples, batch, steps, rounds, compact_only, &vulkan_device());
        return;
    }

    if has_flag(&args, "bc") {
        cmd_bc(&args);
        return;
    }
    if has_flag(&args, "fidelity") {
        cmd_fidelity(&args);
        return;
    }
    if has_flag(&args, "winrate") {
        cmd_winrate(&args);
        return;
    }
    if has_flag(&args, "behavior") {
        cmd_behavior(&args);
        return;
    }
    if has_flag(&args, "token-scan") {
        cmd_token_scan(&args);
        return;
    }
    if has_flag(&args, "td-value") {
        cmd_td_value(&args);
        return;
    }
    if has_flag(&args, "value-rank") {
        cmd_value_rank(&args);
        return;
    }
    if has_flag(&args, "az") {
        cmd_az(&args);
        return;
    }

    println!("Usage:");
    println!("  novacana-ml7 bench [--batch-size 2] [--steps 3] [--rounds 3] [--layout compact]");
    println!("  novacana-ml7 bc [options]");
    println!("  novacana-ml7 fidelity --model <path> [options]");
    println!("  novacana-ml7 winrate --model <path> [options]");
    println!("  novacana-ml7 behavior --model <path> [options]");
    println!("  novacana-ml7 token-scan --model <path> [options]");
    println!("  novacana-ml7 td-value --model <path> [options]");
    println!("  novacana-ml7 value-rank --model <path> [options]");
    println!("  novacana-ml7 az --model <path> [options]");
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
    println!("  --winrate-every <n>   epochs between win-rate gate checks, 0=off [1]");
    println!("  --winrate-games <n>   games per win-rate check [150]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
    println!();
    println!("Fidelity options (per-ActionKind accuracy breakdown, ASTRA7.md §1.1/§10):");
    println!("  --model <path>        checkpoint to evaluate [models/astra7_bc_best.bin]");
    println!("  --val-games <n>       fresh games to evaluate against [200]");
    println!("  --players <n>         [4]");
    println!("  --batch-size <n>      [64]");
    println!();
    println!("Winrate options (ASTRA7.md §4.1/§2 milestone eval - greedy net vs AdvCPU, no search):");
    println!("  --model <path>        checkpoint to evaluate [models/astra7_bc_best.bin]");
    println!("  --games <n>           [200] (use >=1000 for a trustworthy milestone-gate reading)");
    println!("  --players <n>         [4]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
    println!();
    println!("Token-scan options (real live-play, catches drift-only collisions fidelity can't see):");
    println!("  --model <path>        checkpoint to evaluate [models/astra7_bc_best.bin]");
    println!("  --games <n>           [50]");
    println!("  --players <n>         [4]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
    println!();
    println!("TD-value options (ASTRA7.md §4.2 - frozen-trunk value-head training):");
    println!("  --model <path>        REQUIRED: BC checkpoint to warm-start the trunk from");
    println!("  --out <path>          [models/astra7_td.bin]");
    println!("  --chunk-games <n>     fresh self-play games generated per pass [1500]");
    println!("  --passes <n>          §4.2 says 2-3; each regenerates data fresh [3]");
    println!("  --players <n>         [4]");
    println!("  --temperature <f>     self-play sampling temperature [0.8]");
    println!("  --lr <f>              [0.001 frozen trunk / 0.0001 with --unfreeze-trunk]");
    println!("  --batch-size <n>      [64]");
    println!("  --val-games <n>       held-out games for best-checkpoint selection [200]");
    println!("  --unfreeze-trunk      full-net fine-tune (2026-07-10 postmortem fix), not just the value head");
    println!("  --distill-weight <f>  policy-distillation anchor weight, only used with --unfreeze-trunk [0.75]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
    println!();
    println!("value-rank options (ASTRA7.md §4.2 diagnostic, gate >=65%):");
    println!("  --model <path>        checkpoint to evaluate");
    println!("  --games <n>           AdvCPU self-play games to source decision points from [400]");
    println!("  --points <n>          decision points to score (subsampled) [1000]");
    println!("  --players <n>         [4]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
    println!();
    println!("AZ league options (ASTRA7.md §4.3):");
    println!("  --model <path>        REQUIRED: TD-value checkpoint to start the league from");
    println!("  --out <path>          [models/astra7_az.bin]");
    println!("  --iters <n>           [150]");
    println!("  --games-per-iter <n>  self-play games per iteration [150]");
    println!("  --players <n>         [4]");
    println!("  --sims <n>            MCTS sims/decision, floor 400 per §4.3/§8.5 [400]");
    println!("  --c-puct <f>          [1.5]");
    println!("  --dirichlet-eps <f>   [0.25]");
    println!("  --advcpu-share <f>    opponent-mix AdvCPU share [0.3]");
    println!("  --expert-share <f>    opponent-mix ExpertCpuPlayer share [0.05]");
    println!("  --window <n>          sliding replay window, samples [150000]");
    println!("  --batch-size <n>      [128]");
    println!("  --lr-start <f>        [0.00005]");
    println!("  --lr-end <f>          [0.00002]");
    println!("  --snapshot-every <n>  iterations between numbered snapshots [20]");
    println!("  --eval-every <n>      iterations between verified evals [10]");
    println!("  --eval-games <n>      in-loop eval games, >=600 per §2 [600]");
    println!("  --gen-concurrency <n> concurrent self-play/eval workers [2x cores, min 8]");
}

// ── BC training ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_bc<B: AutodiffBackend>(
    initial: Option<Net<B>>,
    resume: Option<ResumeMeta>,
    out_path: &str,
    num_games: usize,
    chunk_games: usize,
    num_players: usize,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    label_smoothing: f32,
    val_games: usize,
    winrate_every: usize,
    winrate_games: usize,
    checkpoint_every: usize,
    concurrency: usize,
    device: &B::Device,
) where
    B::Device: Clone + Sync,
{
    let mut net = initial.unwrap_or_else(|| Net::fresh(device));
    let mut opt = net::make_optimizer::<B>();
    let mut rng = thread_rng();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let shutdown = install_shutdown_flag();

    // chunk_games == 0: generate all upfront and reuse across epochs (fine
    // for small demo runs). chunk_games > 0: fresh games every epoch -
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
    // val-set's samples/game ratio as a stand-in for train data density -
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
    let mut global_step: usize = resume.as_ref().map(|r| r.global_step).unwrap_or(0);

    let mut best_acc: f32 = resume.as_ref().map(|r| r.best_acc).unwrap_or(-1.0);
    let best_path = best_checkpoint_path(out_path);
    let start_epoch = resume.as_ref().map(|r| r.epoch).unwrap_or(0);
    if let Some(r) = &resume {
        println!(
            "Resuming from epoch {} (global_step={}, best_acc={:.4}) - LR schedule and best-checkpoint tracking continue, not restarted.",
            r.epoch + 1, r.global_step, r.best_acc
        );
    }
    let mut shut_down_early = false;

    'epochs: for epoch in start_epoch..epochs {
        let fresh: Vec<Sample>;
        let samples: &[Sample] = if let Some(ref s) = pregenerated {
            s
        } else {
            fresh = recorder::generate_samples_abortable(chunk_games, num_players, Some(std::sync::Arc::clone(&shutdown)));
            &fresh
        };
        // Abort mid-generation (see generate_samples_abortable's doc): don't
        // train on a partial/empty chunk, just stop - no weights changed
        // this epoch, so there's nothing new to checkpoint, just exit.
        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            println!("Shutdown requested during self-play generation - nothing new to checkpoint this epoch, exiting.");
            return;
        }

        let idx: Vec<usize> = (0..samples.len()).collect();
        let batches = length_grouped(idx, batch_size, |i| net::sequence_span(&samples[i].tokens), &mut rng);

        // Progress heartbeat: per-epoch logging alone is blind at this scale
        // (a 300-game epoch can be thousands of batches) - print elapsed
        // time + batch index every HEARTBEAT_EVERY batches so a stalled vs.
        // slow-but-progressing run is distinguishable without guessing.
        const HEARTBEAT_EVERY: usize = 50;
        let total_batches = batches.len().max(1);

        let (mut tpl, mut tvl, mut metric_batches, mut nb) = (0.0f64, 0.0f64, 0usize, 0usize);
        let (mut last_pl, mut last_vl) = (0.0f32, 0.0f32);
        let t0 = Instant::now();
        for chunk in &batches {
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
            let report_metrics = (nb + 1) % HEARTBEAT_EVERY == 0 || nb + 1 == total_batches;
            let (nn, pl, vl) = net::bc_train_step(
                net, &mut opt, cur_lr, &seqs, &targets, &outcomes, label_smoothing, report_metrics, device,
            );
            net = nn;
            if let (Some(pl), Some(vl)) = (pl, vl) {
                tpl += pl as f64;
                tvl += vl as f64;
                metric_batches += 1;
                last_pl = pl;
                last_vl = vl;
            }
            nb += 1;
            global_step += 1;
            if nb % HEARTBEAT_EVERY == 0 {
                let elapsed = t0.elapsed().as_secs_f32();
                println!(
                    "    heartbeat: batch {}/{}  sampled_policy={:.4} (mean={:.4})  sampled_value={:.4} (mean={:.4})  lr={:.2e}  {:.2}s elapsed  ({:.3}s/batch)",
                    nb, total_batches, last_pl, tpl / metric_batches.max(1) as f64, last_vl, tvl / metric_batches.max(1) as f64, cur_lr, elapsed, elapsed / nb as f32
                );
            }

            // Mid-epoch checkpointing + graceful shutdown (2026-07-10):
            // without this, a kill mid-epoch loses everything back to the
            // last epoch boundary - potentially the whole epoch's training
            // on a machine that gets paused/reclaimed often. `checkpoint_every`
            // bounds the max loss to that many batches; the shutdown flag
            // (Ctrl+C/SIGTERM, e.g. from a "stop training" shortcut) triggers
            // an immediate save-and-exit instead of a hard kill.
            if checkpoint_every > 0 && nb % checkpoint_every == 0 {
                if let Err(e) = net.clone().save_file(out_path, &recorder) {
                    eprintln!("Warning: periodic checkpoint save failed: {e:?}");
                } else {
                    save_resume_meta(out_path, &ResumeMeta { epoch, global_step, best_acc });
                }
            }
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                if let Err(e) = net.clone().save_file(out_path, &recorder) {
                    eprintln!("Warning: shutdown checkpoint save failed: {e:?}");
                } else {
                    save_resume_meta(out_path, &ResumeMeta { epoch, global_step, best_acc });
                    let real_path = checkpoint_display_path(out_path);
                    println!(
                        "Saved shutdown checkpoint -> {real_path} (epoch {}, batch {}/{}). Resume with --resume {real_path}",
                        epoch + 1, nb, total_batches
                    );
                }
                shut_down_early = true;
                break 'epochs;
            }
        }

        if !val_samples.is_empty() {
            // net.valid() (inner, non-autodiff backend): forward passes on
            // the autodiff backend accumulate graph state that only
            // backward() releases (ASTRA7.md §8.2).
            let vnet = net.valid();
            let vidx: Vec<usize> = (0..val_samples.len()).collect();
            let mut diag = net::ValDiag::default();
            for chunk in vidx.chunks(batch_size.max(1)) {
                let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| val_samples[i].tokens).collect();
                let targets: Vec<u16> = chunk.iter().map(|&i| val_samples[i].target).collect();
                let outcomes: Vec<f32> = chunk.iter().map(|&i| val_samples[i].outcome).collect();
                diag.absorb(net::val_diag(&vnet, &seqs, &targets, &outcomes, device));
            }
            let n = diag.len().max(1);
            let acc = diag.top1_count() as f32 / n as f32;
            let vce = diag.ce_sum() / n as f32;
            let star = if acc > best_acc { "  *best" } else { "" };
            println!(
                "Iter {}/{}: sampled_policy={:.4} sampled_value={:.4}  Eval @ val_acc={:.4} val_ce={:.4}  ({:.1}s){}",
                epoch + 1, epochs, tpl / metric_batches.max(1) as f64, tvl / metric_batches.max(1) as f64, acc, vce, t0.elapsed().as_secs_f32(), star
            );
            println!("  diag@epoch{}: {}", epoch + 1, diag.to_json());
            // Win-rate eval every N epochs (ASTRA7.md §4.1's actual gate -
            // 2026-07-09: a mid-training checkpoint hit 92.8% val_acc but
            // only 14% win_rate, well below both the >=25% floor and v6's
            // own 16% BC baseline. Token-accuracy and win-rate can diverge
            // sharply (compounding-error distribution shift is the expected
            // mechanism, but only measuring it catches it - that gap sat
            // undetected for 5+ hours before a manual check surfaced it).
            // Wired in here, not as a separate manual step, so a bad run
            // gets caught within one epoch instead of at the very end.
            if winrate_every > 0 && (epoch + 1) % winrate_every == 0 {
                let wt0 = Instant::now();
                let wr = eval_win_rate(&vnet, device, winrate_games, num_players, concurrency);
                println!(
                    "  winrate@epoch{}: {:.4}  (n={}, {:.1}s)  [gate: >=0.25 floor, 0.30-0.40 target]",
                    epoch + 1, wr, winrate_games, wt0.elapsed().as_secs_f32()
                );
                // Spectator game for the dashboard's replay theater + card-
                // spectrum chart (2026-07-15, dashboard "make it fun" pass -
                // see STATE.md). One extra game, same seat setup as the
                // win-rate eval above - cheap relative to winrate_games
                // games. Never fatal: a write failure here shouldn't take
                // down training, just warns and moves on.
                let spectator_game = spectator::record_spectator_game(&vnet, device, num_players);
                write_spectator_outputs(out_path, epoch + 1, &spectator_game);
            }
            if acc > best_acc {
                best_acc = acc;
                let _ = std::fs::remove_file(format!("{best_path}.bin"));
                if let Err(e) = net.clone().save_file(best_path.clone(), &recorder) {
                    eprintln!("Warning: best-checkpoint save failed: {e:?}");
                }
            }
        } else {
            println!(
                "Iter {}/{}: sampled_policy={:.4} sampled_value={:.4}  ({:.1}s)",
                epoch + 1, epochs, tpl / metric_batches.max(1) as f64, tvl / metric_batches.max(1) as f64, t0.elapsed().as_secs_f32()
            );
        }
    }

    if shut_down_early {
        // Already saved (weights + resume metadata) at the shutdown point
        // above - re-saving here would just be redundant work on the way
        // out, and "training complete" would be a misleading thing to print
        // for a run that was deliberately interrupted, not finished.
        return;
    }
    if let Err(e) = net.save_file(out_path, &recorder) {
        eprintln!("Warning: final save failed: {e:?}");
    }
    println!(
        "BC training complete -> {} (best checkpoint -> {best_path}.bin, val_acc={:.4})",
        checkpoint_display_path(out_path), best_acc.max(0.0)
    );
}

// ── Per-ActionKind fidelity diagnostic (ASTRA7.md §1.1/§10) ────────────────
//
// Answers "is chain-response fidelity lagging other decision kinds" with
// real evidence on a saved checkpoint, instead of an aggregate accuracy
// number that could hide exactly that gap. Free to run (forward-only,
// no training) - the recommended step before trusting a BC checkpoint or
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
         fidelity check - if their accuracy trails other rows by a wide margin, that's the signal\n\
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
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        run_fidelity::<Gpu>(&model_path, &samples, batch_size, &device);
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        run_fidelity::<NdArray>(&model_path, &samples, batch_size, &device);
    }
}

/// Scans real live-play games (net vs AdvCPU, same setup as `winrate`) for
/// token collisions at whatever state the net's OWN drifted play actually
/// reaches - the failure mode a fidelity check (which only ever samples
/// AdvCPU-typical states) structurally cannot catch. See az.rs's module doc.
fn cmd_token_scan(args: &[String]) {
    let games = flag_usize(args, "--games", 50);
    let num_players = flag_usize(args, "--players", 4);
    // --advcpu: scan the actual BC training distribution (all seats the
    // real BC teacher, no net) instead of live net-vs-AdvCPU drifted play -
    // answers "how noisy was the training signal itself", independent of
    // whether the net has drifted off-distribution (2026-07-09 discussion).
    // Flag name predates the 2026-07-10 AdvCPU->ExpertCPU teacher switch;
    // kept as-is since it's already muscle memory, but it now scans
    // ExpertCPU (see scan_training_data_for_collisions's doc).
    let advcpu_mode = has_flag(args, "--advcpu");
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    let t0 = Instant::now();
    let report = if advcpu_mode {
        println!("Token-collision scan (BC training distribution, ExpertCPU): games={games}  players={num_players}");
        scan_training_data_for_collisions(games, num_players)
    } else {
        let model_path = flag_str(args, "--model", "models/astra7_bc_best.bin");
        println!("Token-collision scan (live net-vs-AdvCPU play): model={model_path}  games={games}  players={num_players}");
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        #[cfg(feature = "tch")]
        {
            let device = LibTorchDevice::Cuda(0);
            let net = Net::<LibTorch<f32>>::fresh(&device)
                .load_file(model_path.to_string(), &recorder, &device)
                .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
            scan_live_play_for_collisions(&net, &device, games, num_players, concurrency)
        }
        #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
        {
            let device = vulkan_device();
            let net = Net::<Gpu>::fresh(&device)
                .load_file(model_path.to_string(), &recorder, &device)
                .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
            scan_live_play_for_collisions(&net, &device, games, num_players, concurrency)
        }
        #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
        {
            let device = NdArrayDevice::Cpu;
            let net = Net::<NdArray>::fresh(&device)
                .load_file(model_path.to_string(), &recorder, &device)
                .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
            scan_live_play_for_collisions(&net, &device, games, num_players, concurrency)
        }
    };

    let pct = 100.0 * report.collisions as f32 / report.decisions.max(1) as f32;
    println!(
        "decisions={}  collisions={}  ({:.2}%)  ({:.1}s)",
        report.decisions, report.collisions, pct, t0.elapsed().as_secs_f32()
    );
    if report.collisions > 0 {
        println!("\nCollisions by ActionKind:");
        for k in 0..NUM_ACTION_KINDS {
            if report.by_kind[k] > 0 {
                println!("  {:<32} {}", action_kind_name(k as u8), report.by_kind[k]);
            }
        }
        println!("\nExample collisions (up to 20):");
        for ex in &report.examples {
            println!("  {ex}");
        }
    } else {
        println!("No token collisions found.");
    }
}

/// Standalone CLI wrapper around `az::eval_win_rate` (ASTRA7.md §4.1's
/// "milestone eval": greedy net vs AdvCPU, no search) - the function
/// already existed for AZ's own in-loop evals, this just exposes it against
/// any checkpoint (BC, TD-value, or AZ) without needing a full AZ run.
/// Live-play readout: same greedy-vs-AdvCPU games as `winrate`, plus the
/// logit gap between Pass and the best other legal play.
fn cmd_behavior(args: &[String]) {
    let model_path = flag_str(args, "--model", "models/astra7_bc_best.bin");
    let games = flag_usize(args, "--games", 16);
    let num_players = flag_usize(args, "--players", 4);
    let concurrency = flag_usize(args, "--gen-concurrency", 4);

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    println!("Behavior eval: model={model_path}  games={games}  players={num_players}");
    let t0 = Instant::now();

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        let net = Net::<LibTorch<f32>>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        az::eval_behavior(&net, &device, games, num_players, concurrency);
        println!("({:.1}s)", t0.elapsed().as_secs_f32());
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        let net = Net::<Gpu>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        az::eval_behavior(&net, &device, games, num_players, concurrency);
        println!("({:.1}s)", t0.elapsed().as_secs_f32());
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        az::eval_behavior(&net, &device, games, num_players, concurrency);
        println!("({:.1}s)", t0.elapsed().as_secs_f32());
    }
}

fn cmd_winrate(args: &[String]) {
    let model_path = flag_str(args, "--model", "models/astra7_bc_best.bin");
    let games = flag_usize(args, "--games", 200);
    let num_players = flag_usize(args, "--players", 4);
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    println!("Winrate eval (ASTRA7.md §4.1/§2): model={model_path}  games={games}  players={num_players}");
    let t0 = Instant::now();

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        let net = Net::<LibTorch<f32>>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let rate = eval_win_rate(&net, &device, games, num_players, concurrency);
        println!("win_rate={:.4}  ({:.1}s)", rate, t0.elapsed().as_secs_f32());
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        let net = Net::<Gpu>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let rate = eval_win_rate(&net, &device, games, num_players, concurrency);
        println!("win_rate={:.4}  ({:.1}s)", rate, t0.elapsed().as_secs_f32());
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        let net = Net::<NdArray>::fresh(&device)
            .load_file(model_path.to_string(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let rate = eval_win_rate(&net, &device, games, num_players, concurrency);
        println!("win_rate={:.4}  ({:.1}s)", rate, t0.elapsed().as_secs_f32());
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
    // Real win-rate gate (ASTRA7.md §4.1), not just token-accuracy - found
    // 2026-07-09 that these can diverge sharply (92.8% val_acc, 14% win_rate
    // on the same checkpoint). every=1 by default: the eval itself is cheap
    // (~130s/200 games) against a multi-hour epoch; every=0 disables it.
    let winrate_every = flag_usize(args, "--winrate-every", 1);
    let winrate_games = flag_usize(args, "--winrate-games", 150);
    // Pause/resume (2026-07-10, see run_bc's module doc): 0 disables
    // mid-epoch checkpointing (epoch-boundary-only, the old behavior).
    let checkpoint_every = flag_usize(args, "--checkpoint-every", 300);
    // Self-play/eval concurrency (2026-07-16, Fable 5 consult): adapts to
    // this machine's core count by default (batch_infer.rs's
    // default_concurrency) instead of a value hardcoded for the old pod's
    // 13-14 vCPUs - override for a box where that guess is wrong.
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let model_path = flag_str(args, "--model", "");
    let resume_path = flag_str(args, "--resume", "");
    if !model_path.is_empty() && !resume_path.is_empty() {
        eprintln!("--model and --resume are mutually exclusive (--model is a fresh-schedule warm start, --resume continues the same run) - pick one.");
        std::process::exit(1);
    }
    let load_path = if !resume_path.is_empty() { resume_path.clone() } else { model_path.clone() };
    let resume_meta = if !resume_path.is_empty() { Some(load_resume_meta(&resume_path)) } else { None };

    // Backend selection (ASTRA7.md §3.3/§5): `tch` feature -> LibTorch/CUDA
    // on Atlas (Autodiff<LibTorch<f32>>, device Cuda(0)); default ->
    // burn-ndarray CPU, kept so hermes-side quick sanity runs/tests still
    // work without needing Atlas. Mirrors ml/src/main.rs's gpu/not(gpu) split.
    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        type B = Autodiff<LibTorch<f32>>;

        let initial: Option<Net<B>> = if !load_path.is_empty() {
            if resume_meta.is_none() {
                println!("Warm start from {load_path}");
            }
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            Some(
                Net::<B>::fresh(&device)
                    .load_file(load_path.clone(), &recorder, &device)
                    .unwrap_or_else(|e| panic!("Failed to load {load_path}: {e:?}")),
            )
        } else {
            None
        };

        run_bc::<B>(
            initial, resume_meta, &out_path, num_games, chunk_games, num_players, lr, epochs, batch_size, label_smoothing,
            val_games, winrate_every, winrate_games, checkpoint_every, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        type B = Autodiff<Gpu>;
        let initial: Option<Net<B>> = if !load_path.is_empty() {
            if resume_meta.is_none() {
                println!("Warm start from {load_path}");
            }
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            Some(Net::<B>::fresh(&device)
                .load_file(load_path.clone(), &recorder, &device)
                .unwrap_or_else(|e| panic!("Failed to load {load_path}: {e:?}")))
        } else {
            None
        };
        run_bc::<B>(
            initial, resume_meta, &out_path, num_games, chunk_games, num_players, lr, epochs, batch_size, label_smoothing,
            val_games, winrate_every, winrate_games, checkpoint_every, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray>;

        let initial: Option<Net<B>> = if !load_path.is_empty() {
            if resume_meta.is_none() {
                println!("Warm start from {load_path}");
            }
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            Some(
                Net::<B>::fresh(&device)
                    .load_file(load_path.clone(), &recorder, &device)
                    .unwrap_or_else(|e| panic!("Failed to load {load_path}: {e:?}")),
            )
        } else {
            None
        };

        run_bc::<B>(
            initial, resume_meta, &out_path, num_games, chunk_games, num_players, lr, epochs, batch_size, label_smoothing,
            val_games, winrate_every, winrate_games, checkpoint_every, concurrency, &device,
        );
    }
}

// ── TD-value training (ASTRA7.md §4.2) ──────────────────────────────────
//
// Each "pass" regenerates a fresh self-play chunk (net's own policy,
// temperature-sampled - see td_value.rs) and does ONE frozen-trunk training
// sweep over it. "2-3 passes" (§4.2) means 2-3 such regenerate+train
// rounds, not 2-3 epochs over one fixed dataset - see td_value.rs's module
// doc for why TD targets are computed once per chunk rather than
// re-bootstrapped mid-pass.

fn value_mse<B: Backend>(net: &Net<B>, seqs: &[TokenSeq], targets: &[f32], device: &B::Device) -> f32 {
    if seqs.is_empty() {
        return 0.0;
    }
    let (_, values) = net::forward(net, seqs, device);
    let preds = net::download_f32(values);
    preds.iter().zip(targets.iter()).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / preds.len().max(1) as f32
}

#[allow(clippy::too_many_arguments)]
fn run_td_value<B: AutodiffBackend>(
    initial: Net<B>,
    out_path: &str,
    chunk_games: usize,
    passes: usize,
    num_players: usize,
    temperature: f32,
    lr: f32,
    batch_size: usize,
    val_games: usize,
    advcpu_share: f32,
    unfreeze_trunk: bool,
    distill_weight: f32,
    concurrency: usize,
    device: &B::Device,
) where
    B::Device: Clone + Send + Sync + 'static,
{
    let mut net = initial;
    let mut opt = net::make_optimizer::<B>();
    let mut rng = thread_rng();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let best_path = best_checkpoint_path(out_path);
    let mut best_val_mse = f32::MAX;
    let shutdown = install_shutdown_flag();
    // Frozen teacher snapshot for the unfrozen-trunk policy-distillation
    // anchor (2026-07-10, net.rs::td_unfrozen_train_step's doc) - taken
    // ONCE from the BC checkpoint this run started from and held fixed for
    // the whole run, so it anchors to "what BC produced," not a moving
    // target. Built either way (cheap forward-only clone) since it's only
    // ever read from when unfreeze_trunk=true.
    let teacher = net.clone().valid();
    let mut shut_down_early = false;

    'passes: for pass in 0..passes {
        let t0 = Instant::now();
        let vnet = net.valid(); // ASTRA7.md §8.2: self-play generation is forward-only
        println!(
            "TD-value pass {}/{}: generating {} self-play games (T={temperature}, advcpu_share={advcpu_share})...",
            pass + 1, passes, chunk_games
        );
        let samples: Vec<TdSample> = generate_td_samples(&vnet, device, chunk_games, num_players, temperature, advcpu_share, concurrency);
        println!("  {} samples in {:.1}s", samples.len(), t0.elapsed().as_secs_f32());
        if samples.len() < 2 {
            continue;
        }

        let idx: Vec<usize> = (0..samples.len()).collect();
        let batches = length_grouped(idx, batch_size, |i| net::sequence_span(&samples[i].tokens), &mut rng);

        const HEARTBEAT_EVERY: usize = 100;
        let total_batches = batches.len().max(1);
        let (mut tvl, mut tdl, mut metric_batches, mut nb) = (0.0f64, 0.0f64, 0usize, 0usize);
        let t1 = Instant::now();
        for chunk in &batches {
            if chunk.len() < 2 {
                continue;
            }
            let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| samples[i].tokens).collect();
            let targets: Vec<f32> = chunk.iter().map(|&i| samples[i].td_target).collect();
            let report_metrics = (nb + 1) % HEARTBEAT_EVERY == 0 || nb + 1 == total_batches;
            if unfreeze_trunk {
                let (nn, vl, dl) = net::td_unfrozen_train_step(
                    net, &teacher, &mut opt, lr as f64, &seqs, &targets, distill_weight, report_metrics, device,
                );
                net = nn;
                if let (Some(vl), Some(dl)) = (vl, dl) {
                    tvl += vl as f64;
                    tdl += dl as f64;
                    metric_batches += 1;
                }
            } else {
                let (nn, vl) = net::td_train_step(net, &mut opt, lr as f64, &seqs, &targets, report_metrics, device);
                net = nn;
                if let Some(vl) = vl {
                    tvl += vl as f64;
                    metric_batches += 1;
                }
            }
            nb += 1;
            if nb % HEARTBEAT_EVERY == 0 {
                if unfreeze_trunk {
                    println!(
                        "    heartbeat: batch {}/{}  sampled_value_mse={:.4}  sampled_distill_ce={:.4}  {:.2}s elapsed",
                        nb, total_batches, tvl / metric_batches.max(1) as f64, tdl / metric_batches.max(1) as f64, t1.elapsed().as_secs_f32()
                    );
                } else {
                    println!(
                        "    heartbeat: batch {}/{}  sampled_value_mse={:.4}  {:.2}s elapsed",
                        nb, total_batches, tvl / metric_batches.max(1) as f64, t1.elapsed().as_secs_f32()
                    );
                }
            }
            // Graceful shutdown (2026-07-10, same rationale as run_bc): save
            // whatever's trained so far and stop, instead of a hard kill
            // losing this pass's training entirely. No mid-pass resume
            // metadata (unlike BC) - TD-value's passes are few (1-3) and the
            // expensive part per pass is self-play generation, which can't
            // be checkpointed mid-way anyway (no model weights change until
            // training starts), so the loss window this bounds is already
            // small relative to BC's many-epoch runs.
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                if let Err(e) = net.clone().save_file(out_path, &recorder) {
                    eprintln!("Warning: shutdown checkpoint save failed: {e:?}");
                } else {
                    println!("Saved shutdown checkpoint -> {} (pass {}, batch {}/{}).", checkpoint_display_path(out_path), pass + 1, nb, total_batches);
                }
                shut_down_early = true;
                break 'passes;
            }
        }

        if val_games > 0 {
            let vnet2 = net.valid();
            let val_samples = generate_td_samples(&vnet2, device, val_games, num_players, temperature, advcpu_share, concurrency);
            let (mut mse_sum, mut n) = (0.0f64, 0usize);
            for chunk in val_samples.chunks(batch_size.max(1)) {
                let seqs: Vec<TokenSeq> = chunk.iter().map(|s| s.tokens).collect();
                let targets: Vec<f32> = chunk.iter().map(|s| s.td_target).collect();
                mse_sum += value_mse(&vnet2, &seqs, &targets, device) as f64 * chunk.len() as f64;
                n += chunk.len();
            }
            let val_mse = if n > 0 { (mse_sum / n as f64) as f32 } else { 0.0 };
            let star = if val_mse < best_val_mse { "  *best" } else { "" };
            if unfreeze_trunk {
                println!(
                    "Pass {}/{}: sampled_train_value_mse={:.4}  sampled_train_distill_ce={:.4}  val_value_mse={:.4}  ({:.1}s){}",
                    pass + 1, passes, tvl / metric_batches.max(1) as f64, tdl / metric_batches.max(1) as f64, val_mse, t0.elapsed().as_secs_f32(), star
                );
            } else {
                println!(
                    "Pass {}/{}: sampled_train_value_mse={:.4}  val_value_mse={:.4}  ({:.1}s){}",
                    pass + 1, passes, tvl / metric_batches.max(1) as f64, val_mse, t0.elapsed().as_secs_f32(), star
                );
            }
            if val_mse < best_val_mse {
                best_val_mse = val_mse;
                let _ = std::fs::remove_file(format!("{best_path}.bin"));
                if let Err(e) = net.clone().save_file(best_path.clone(), &recorder) {
                    eprintln!("Warning: best-checkpoint save failed: {e:?}");
                }
            }
        } else {
            println!(
                "Pass {}/{}: sampled_train_value_mse={:.4}  ({:.1}s)",
                pass + 1, passes, tvl / metric_batches.max(1) as f64, t0.elapsed().as_secs_f32()
            );
        }
    }

    if shut_down_early {
        // Already saved at the shutdown point above.
        return;
    }
    if let Err(e) = net.save_file(out_path, &recorder) {
        eprintln!("Warning: final save failed: {e:?}");
    }
    let real_path = checkpoint_display_path(out_path);
    if best_val_mse == f32::MAX {
        // val_games=0 (or every pass produced <2 samples): no best-checkpoint
        // save ever ran, so best_path.bin doesn't exist - don't claim it does.
        println!("TD-value training complete -> {real_path} (no --val-games, no best checkpoint saved)");
    } else {
        println!("TD-value training complete -> {real_path} (best checkpoint -> {best_path}.bin, val_value_mse={best_val_mse:.4})");
    }
}

fn cmd_td_value(args: &[String]) {
    let out_path = flag_str(args, "--out", "models/astra7_td.bin");
    let model_path = flag_str(args, "--model", "");
    if model_path.is_empty() {
        eprintln!(
            "td-value requires --model <BC checkpoint> (ASTRA7.md §4.2: TD-value fine-tunes \
             a BC-trained value head - trunk frozen by default, --unfreeze-trunk for the \
             2026-07-10 postmortem's full-net fine-tune; either way it doesn't train from scratch)"
        );
        std::process::exit(1);
    }
    let chunk_games = flag_usize(args, "--chunk-games", 1500);
    let passes = flag_usize(args, "--passes", 3);
    let num_players = flag_usize(args, "--players", 4);
    let temperature = flag_f32(args, "--temperature", 0.8);
    let batch_size = flag_usize(args, "--batch-size", 64);
    let val_games = flag_usize(args, "--val-games", 200);
    // 2026-07-10 postmortem (STATE.md "TD-value stage"): frozen-trunk
    // head-only training plateaued well under the value-rank gate at both
    // λ=0.9 (50.8%) and λ=1.0 (44.8%, worse - chance level). Two Fable 5
    // consults converged on the frozen trunk's policy-shaped features being
    // the binding constraint, not the TD target formula.
    let advcpu_share = flag_f32(args, "--advcpu-share", 0.5);
    let unfreeze_trunk = has_flag(args, "--unfreeze-trunk");
    // The default --lr depends on the mode: 0.001 was tuned for the OLD
    // frozen-trunk (head-only) path. The postmortem's unfrozen-trunk fix
    // (net.rs::td_unfrozen_train_step's doc comment) explicitly calls for
    // ~1e-4 - a full-trunk Adam step at the head-only rate would blow past
    // what the 0.75-weighted distillation anchor can hold onto, wrecking the
    // BC-trained trunk before the value head ever gets a useful signal.
    // Caught before this mode's first real run (Fable 5 consult,
    // 2026-07-16) - --unfreeze-trunk had never actually been exercised
    // against real training data, so this bug was purely latent until now.
    let lr_default = if unfreeze_trunk { 0.0001 } else { 0.001 };
    let lr = flag_f32(args, "--lr", lr_default);
    println!("td-value: unfreeze_trunk={unfreeze_trunk} lr={lr} (default for this mode: {lr_default})");
    let distill_weight = flag_f32(args, "--distill-weight", 0.75);
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        type B = Autodiff<LibTorch<f32>>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_td_value::<B>(
            net, &out_path, chunk_games, passes, num_players, temperature, lr, batch_size, val_games, advcpu_share, unfreeze_trunk,
            distill_weight, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        type B = Autodiff<Gpu>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_td_value::<B>(
            net, &out_path, chunk_games, passes, num_players, temperature, lr, batch_size, val_games, advcpu_share, unfreeze_trunk,
            distill_weight, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_td_value::<B>(
            net, &out_path, chunk_games, passes, num_players, temperature, lr, batch_size, val_games, advcpu_share, unfreeze_trunk,
            distill_weight, concurrency, &device,
        );
    }
}

// ── value-rank diagnostic (ASTRA7.md §4.2) ──────────────────────────────

fn report_value_rank(rate: f32, n: usize, t0: Instant) {
    println!("value-rank: {n} decision points scored in {:.1}s", t0.elapsed().as_secs_f32());
    println!("  top-2 hit rate: {:.1}%  (gate: >=65.0%, ASTRA7.md §4.2)", rate * 100.0);
    if rate >= 0.65 {
        println!("  PASS - value head ranks successors well enough to trust MCTS leaf evals.");
    } else {
        println!("  FAIL - do not proceed to the AZ league until this clears the gate.");
    }
}

fn cmd_value_rank(args: &[String]) {
    let model_path = flag_str(args, "--model", "");
    if model_path.is_empty() {
        eprintln!("value-rank requires --model <path>");
        std::process::exit(1);
    }
    let num_games = flag_usize(args, "--games", 400);
    let target_points = flag_usize(args, "--points", 1000);
    let num_players = flag_usize(args, "--players", 4);
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    println!("value-rank: model={model_path}  games={num_games}  target_points={target_points}");
    let t0 = Instant::now();

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<LibTorch<f32>>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let (rate, n) = evals::run_value_rank(&net, &device, num_games, num_players, target_points, concurrency);
        report_value_rank(rate, n, t0);
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<Gpu>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let (rate, n) = evals::run_value_rank(&net, &device, num_games, num_players, target_points, concurrency);
        report_value_rank(rate, n, t0);
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<NdArray>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        let (rate, n) = evals::run_value_rank(&net, &device, num_games, num_players, target_points, concurrency);
        report_value_rank(rate, n, t0);
    }
}

// ── AZ league (ASTRA7.md §4.3) ───────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_az<B: AutodiffBackend>(
    initial: Net<B>,
    out_path: &str,
    iters: usize,
    games_per_iter: usize,
    num_players: usize,
    sims: usize,
    c_puct: f32,
    dirichlet_eps: f32,
    advcpu_share: f32,
    expert_share: f32,
    window: usize,
    batch_size: usize,
    lr_start: f32,
    lr_end: f32,
    snapshot_every: usize,
    eval_every: usize,
    eval_games: usize,
    concurrency: usize,
    device: &B::Device,
) where
    B::Device: Clone + Sync,
{
    let mut net = initial;
    let mut opt = net::make_optimizer::<B>();
    let mut rng = thread_rng();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let best_path = best_checkpoint_path(out_path);

    let cfg = McConfig { sims, c_puct, dirichlet_eps };
    // VecDeque, not Vec: the sliding-window trim below drops from the front
    // every iteration once the window fills, and a Vec's drain(0..excess)
    // memmoves every surviving element (up to ~150K AzSamples, ~3.1KB each)
    // on every single iteration. VecDeque still supports the random-access
    // indexing the shuffled-batch loop below needs (Index<usize>), so this
    // is a drop-in swap (Fable 5 consult, 2026-07-16).
    let mut replay: VecDeque<AzSample> = VecDeque::new();
    let mut best_win_rate: f32 = -1.0;
    let mut eval_history: Vec<f32> = Vec::new();

    for iter in 0..iters {
        let t0 = Instant::now();
        let vnet = net.valid(); // ASTRA7.md §8.2: self-play generation is forward-only
        let fresh = generate_az_games(&vnet, device, &cfg, games_per_iter, num_players, advcpu_share, expert_share, concurrency);
        println!(
            "Iter {}/{}: generated {} AZ samples from {} games ({:.1}s)",
            iter + 1, iters, fresh.len(), games_per_iter, t0.elapsed().as_secs_f32()
        );
        replay.extend(fresh);
        if replay.len() > window {
            // Sliding FIFO window (§4.3: 300K samples) - oldest self-play
            // drops off first.
            let excess = replay.len() - window;
            replay.drain(0..excess);
        }
        if replay.len() < 2 {
            continue;
        }

        // Cosine lr_start -> lr_end over the whole run (§4.3).
        let progress = iter as f32 / (iters.max(2) - 1) as f32;
        let lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + (std::f32::consts::PI * progress).cos());

        // az-epochs=1 (§4.3, §8.4: "4 epochs oscillates") - ONE pass over
        // the CURRENT sliding window, not just this iteration's fresh batch.
        let idx: Vec<usize> = (0..replay.len()).collect();
        let batches = length_grouped(idx, batch_size, |i| net::sequence_span(&replay[i].tokens), &mut rng);
        const METRIC_EVERY: usize = 50;
        let total_batches = batches.len().max(1);
        let (mut tpl, mut tvl, mut metric_batches, mut nb) = (0.0f64, 0.0f64, 0usize, 0usize);
        let t1 = Instant::now();
        for chunk in &batches {
            if chunk.len() < 2 {
                continue;
            }
            let seqs: Vec<TokenSeq> = chunk.iter().map(|&i| replay[i].tokens).collect();
            let visit_dists: Vec<[f32; ACTION_MAX]> = chunk.iter().map(|&i| replay[i].visit_dist).collect();
            let outcomes: Vec<f32> = chunk.iter().map(|&i| replay[i].outcome).collect();
            let report_metrics = (nb + 1) % METRIC_EVERY == 0 || nb + 1 == total_batches;
            let (nn, pl, vl) = net::az_train_step(
                net, &mut opt, lr as f64, &seqs, &visit_dists, &outcomes, report_metrics, device,
            );
            net = nn;
            if let (Some(pl), Some(vl)) = (pl, vl) {
                tpl += pl as f64;
                tvl += vl as f64;
                metric_batches += 1;
            }
            nb += 1;
        }
        println!(
            "  sampled_train: policy={:.4} value={:.4}  lr={:.2e}  window={}  ({:.1}s)",
            tpl / metric_batches.max(1) as f64, tvl / metric_batches.max(1) as f64, lr, replay.len(), t1.elapsed().as_secs_f32()
        );

        if (iter + 1) % snapshot_every.max(1) == 0 {
            let snap = snapshot_path(out_path, iter + 1);
            match net.clone().save_file(snap.clone(), &recorder) {
                Ok(_) => println!("  snapshot -> {snap}.bin"),
                Err(e) => eprintln!("Warning: snapshot save failed: {e:?}"),
            }
        }

        if (iter + 1) % eval_every.max(1) == 0 || iter + 1 == iters {
            let t2 = Instant::now();
            let vnet2 = net.valid();
            let win_rate = eval_win_rate(&vnet2, device, eval_games, num_players, concurrency);
            eval_history.push(win_rate);
            let star = if win_rate > best_win_rate { "  *best" } else { "" };
            println!(
                "  Eval @ iter {}: win_rate={:.1}% ({} games, {:.1}s){}",
                iter + 1, win_rate * 100.0, eval_games, t2.elapsed().as_secs_f32(), star
            );

            if win_rate > best_win_rate {
                best_win_rate = win_rate;
                let _ = std::fs::remove_file(format!("{best_path}.bin"));
                if let Err(e) = net.clone().save_file(best_path.clone(), &recorder) {
                    eprintln!("Warning: best-checkpoint save failed: {e:?}");
                }
            }

            // Restart-from-best with a fresh optimizer on two consecutive
            // declines (ASTRA7.md §2/§4.3 - "the v6 leg pattern, it works").
            let n = eval_history.len();
            if n >= 3 && eval_history[n - 1] < eval_history[n - 2] && eval_history[n - 2] < eval_history[n - 3] {
                println!("  Two consecutive eval declines -> restarting from best checkpoint with a fresh optimizer.");
                match Net::<B>::fresh(device).load_file(best_path.clone(), &recorder, device) {
                    Ok(reloaded) => {
                        net = reloaded;
                        opt = net::make_optimizer::<B>();
                        eval_history.clear();
                    }
                    Err(e) => eprintln!("  Warning: restart-from-best failed to load {best_path}.bin: {e:?} - continuing without restart."),
                }
            }
        }
    }

    if let Err(e) = net.save_file(out_path, &recorder) {
        eprintln!("Warning: final save failed: {e:?}");
    }
    let real_path = checkpoint_display_path(out_path);
    if best_win_rate < 0.0 {
        // No eval ever ran (e.g. iters=0, or the replay window never reached
        // 2 samples) - best_path.bin was never written, don't claim it was.
        println!("AZ league complete -> {real_path} (no eval ran, no best checkpoint saved)");
        return;
    }
    println!(
        "AZ league complete -> {real_path} (best checkpoint -> {best_path}.bin, best_win_rate={:.1}%)",
        best_win_rate * 100.0
    );
}

fn cmd_az(args: &[String]) {
    let out_path = flag_str(args, "--out", "models/astra7_az.bin");
    let model_path = flag_str(args, "--model", "");
    if model_path.is_empty() {
        eprintln!(
            "az requires --model <TD-value checkpoint> (ASTRA7.md §4.3: the league continues \
             from a value-rank-gated checkpoint, it doesn't start from scratch)"
        );
        std::process::exit(1);
    }
    let iters = flag_usize(args, "--iters", 150);
    let games_per_iter = flag_usize(args, "--games-per-iter", 150);
    let num_players = flag_usize(args, "--players", 4);
    let sims = flag_usize(args, "--sims", 400);
    if sims < 400 {
        eprintln!(
            "Warning: --sims {sims} is below ASTRA7.md §4.3's floor of 400 - Dirichlet noise at \
             low sim counts crashed v6 (§8.5). Proceeding anyway (assuming a deliberate smoke test)."
        );
    }
    let c_puct = flag_f32(args, "--c-puct", 1.5);
    let dirichlet_eps = flag_f32(args, "--dirichlet-eps", 0.25);
    let advcpu_share = flag_f32(args, "--advcpu-share", 0.3);
    // 0.05, not the original 0.15: ExpertCpu seats run a ~325x-more-expensive
    // search than AdvCPU (recorder.rs's module doc) inside the SAME bounded
    // thread pool as every other concurrent MCTS seat (batch_infer.rs's
    // parallel_for_each) - on a 13-14 vCPU pod that's tolerable slack, but on
    // the brother's PC's 6 WSL2 vCPUs one Expert seat can pin a worker for
    // minutes and stall the whole iteration behind it (Fable 5 consult,
    // 2026-07-16). Diversity value is real (§4.3) but not load-bearing;
    // override with --expert-share on a box with more cores to spare.
    let expert_share = flag_f32(args, "--expert-share", 0.05);
    // 150_000, not the original 300_000: AzSample is ~3.1KB (tokens + visit
    // dist), so 300K samples is ~0.95GB resident - fine on the pod's 58GB
    // cgroup, tight against the brother's PC's 12GB WSL2 cap (dot-wslconfig)
    // once LibTorch/CUDA context and each iteration's fresh samples are
    // accounted for (Fable 5 consult, 2026-07-16). Override with --window on
    // a box with more RAM to spare.
    let window = flag_usize(args, "--window", 150_000);
    // 128, not the original 256: STATE.md's measured 13.5GB peak VRAM at
    // bs=256 on a 16GB A4000 (roughly linear in batch size, same trunk+heads
    // as bc_train_step) would OOM the brother's PC's 10GB 3080 outright -
    // this default was tuned for the old pod and never revisited for the new
    // one (Fable 5 consult, 2026-07-16). Matches launch-bc.sh's bs=128.
    let batch_size = flag_usize(args, "--batch-size", 128);
    let lr_start = flag_f32(args, "--lr-start", 0.00005);
    let lr_end = flag_f32(args, "--lr-end", 0.00002);
    let snapshot_every = flag_usize(args, "--snapshot-every", 20);
    let eval_every = flag_usize(args, "--eval-every", 10);
    let eval_games = flag_usize(args, "--eval-games", 600);
    let concurrency = flag_usize(args, "--gen-concurrency", batch_infer::default_concurrency());

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    #[cfg(feature = "tch")]
    {
        let device = LibTorchDevice::Cuda(0);
        type B = Autodiff<LibTorch<f32>>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_az::<B>(
            net, &out_path, iters, games_per_iter, num_players, sims, c_puct, dirichlet_eps, advcpu_share,
            expert_share, window, batch_size, lr_start, lr_end, snapshot_every, eval_every, eval_games, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), feature = "vulkan"))]
    {
        let device = vulkan_device();
        type B = Autodiff<Gpu>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_az::<B>(
            net, &out_path, iters, games_per_iter, num_players, sims, c_puct, dirichlet_eps, advcpu_share,
            expert_share, window, batch_size, lr_start, lr_end, snapshot_every, eval_every, eval_games, concurrency, &device,
        );
        return;
    }
    #[cfg(all(not(feature = "tch"), not(feature = "vulkan")))]
    {
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray>;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let net = Net::<B>::fresh(&device)
            .load_file(model_path.clone(), &recorder, &device)
            .unwrap_or_else(|e| panic!("Failed to load {model_path}: {e:?}"));
        run_az::<B>(
            net, &out_path, iters, games_per_iter, num_players, sims, c_puct, dirichlet_eps, advcpu_share,
            expert_share, window, batch_size, lr_start, lr_end, snapshot_every, eval_every, eval_games, concurrency, &device,
        );
    }
}
