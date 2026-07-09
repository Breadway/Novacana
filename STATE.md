# Astra-7 — Current State

Snapshot as of 2026-07-09. Full design/history in `ASTRA7.md`; this is the
training-pipeline-focused status doc.

**In progress: migrating from the A4000 to a rented 4090.** An independent
review (Opus 4.8, reading this doc + ASTRA7.md + GPU_RENTAL.md) pointed out
that the live BC telemetry (100% GPU util, wall time = batches × s/batch,
game-gen regen is seconds against a 4hr epoch) contradicts §10's "VRAM
problem not compute problem, GPU tier doesn't matter" conclusion for this
stage — it's compute-bound, and a 4090 should cut the ~60hr/~$10.30 BC run
to ~15-18hr/~$5-7. Decided: switch. Waiting on a new pod + SSH access.

## Infrastructure

**Rented pod**: RunPod, RTX A4000, SSH alias `astra7` (`root@193.183.22.56 -p 1094`,
key `~/.ssh/id_ed25519_astra7`).

| | |
|---|---|
| GPU | RTX A4000, 16GB VRAM, driver 550.144.03 (CUDA 12.4 max) |
| RAM | cgroup-capped at **~58GB** (host has 503GB shared across tenants — do not assume full host RAM is available) |
| Disk | 35GB overlay, ~27GB free |
| Rate | $0.17/hr |
| vCPUs | 14 (game generation is CPU/rayon-bound, this matters) |

**Software stack**: Rust 1.96.1 (rustup stable), LibTorch **2.9.0+cu126**
(matched to `torch-sys 0.22.0`'s pinned `TORCH_VERSION`; cu126 works fine on
the cu124-capable driver via CUDA minor-version compatibility — cu121/cu124
builds no longer exist for LibTorch 2.9.0), burn 0.21 + burn-tch, built via
`ml7/build.rs`'s CUDA-linking fix (`--no-as-needed` around `-ltorch_cuda`,
see `tch-rs-cuda-linking-gotcha` memory).

Repo synced to `~/novacana-engine` on the pod via `rsync` (not git — repo
has substantial uncommitted local changes). Legacy `data/*.bin` (v5/v6
pretraining corpora, ~7.9GB) deliberately **not** synced — Astra-7 generates
its own self-play data on the fly, no static corpus dependency.

## Pipeline status

- Token schema (`tokens.rs`), transformer net (`net.rs`, D_MODEL=256,
  N_HEADS=8, N_LAYERS=6), BC recorder (`recorder.rs`) — built, tested (11/11
  unit tests passing).
- `novacana-ml7 bc` — chunked-streaming BC trainer with warmup+cosine LR
  (added this session), best-checkpoint tracking (bug-fixed this session —
  see below), heartbeat logging.
- `novacana-ml7 fidelity` — per-`ActionKind` accuracy/CE diagnostic tool.
- **Not built** (deliberately, per independent review): `server.rs` /
  `client.rs` — the hermes+Atlas LAN-split architecture was scrapped in
  favor of running BC → TD-value → AZ league all single-box on one rented
  instance.

### Bugs found + fixed this session

1. **Best-checkpoint silent collision** — `format!("{out_path}.best")`
   collapsed onto the final save path because burn's `FileRecorder::save`
   uses `PathBuf::set_extension()`, which *replaces* rather than appends an
   extension. Fixed via `format!("{out_path}_best")` (underscore avoids the
   collision). The earlier Atlas 7-epoch run got lucky (its true best epoch
   was also its last), but the checkpointing was non-functional until fixed.
2. **`DecideOption` encoding gap** — `encode_action_tokens_decide` had no
   positional field, causing token collisions between different options
   sharing the same keyword-flag bucket. Fidelity diagnostic caught it
   (60.0% accuracy on 43% of all samples). Fixed by adding slot-index +
   option-count fields (backward-compatible, no shape change). Directional
   validation: 72.0% vs 60.0% on a quick fresh checkpoint.
3. **Chain-response fidelity — investigated, NOT a bug.** Original concern
   (§1.1) was that the model can't distinguish past chain responses from
   primary plays. Fidelity diagnostic showed `ChainResponsePass` 100.0%,
   `ChainResponseCard` 95.4% — among the best-performing kinds. No fix
   needed, question resolved negatively.
4. **`PickTargetOption` collision (19.7% accuracy) — root-caused and fixed.**
   Initially deferred as "all 7 slots in use, no room to fix" — wrong, only
   checked the field list, not the actual generator. Opus 4.8's review
   (reading this doc) pointed out the 19.7% number looked exactly like
   another token collision, not data starvation, and was worth an hour to
   check before spending a 60hr run + TD-value + AZ on top of it. Traced to
   `disc_target_options`/`chart_target_options` (`engine/src/game.rs`):
   they emit one `TargetOption` per discovery in a star system (plus the
   star itself), but `encode_one_target` only encoded system-level info
   (star name/class/fill count) — every discovery in the same system, and
   the star alongside them, collided into an identical token. Fixed using
   the one free field in each branch: a star/discovery position index for
   `StarChartCard`, a hand-slot index for `HandCard` (duplicate card types
   had the same gap). Not yet re-measured via `fidelity` at scale — do that
   once a checkpoint trained on the fixed encoding exists.

### Fixed this session, not yet stress-tested at scale

- **LR warmup + cosine decay** added to `run_bc` (`main.rs`) — 5% linear
  warmup (floor 50 steps) then cosine decay to 0, replacing the previous
  flat constant LR. Confirmed working correctly against the live run (LR
  ramping smoothly on schedule).
- **Value-loss down-weight (0.3)** — `bc_train_step` (`net.rs`) summed
  policy + value loss 1:1 despite §10 flagging this; now weighted, per that
  same recommendation (BC's value target is only a terminal-outcome warm
  start, TD-value/AZ own the real value signal later).

### Investigated and deliberately NOT done (this session)

- **TF32**: not reachable without writing custom torch-sys FFI. tch-rs
  0.22's generated bindings only expose per-op `allow_tf32` on cuDNN conv
  ops, not the global `setAllowTF32CuBLAS` matmul flag transformers
  actually need.
- **bf16 full-backend swap**: type-compatible (`burn_backend::bf16` impls
  `TchElement`, so `LibTorch<bf16>` compiles) but that's full-precision
  swap including Adam's moment estimates, not true mixed-precision
  autocast — real risk of a silent quality regression, not a crash.
  Deferred pending an isolated correctness test; not worth gambling on the
  live campaign.

## Current training run — MIGRATING TO A RENTED 4090

`astra7_bc_scaled.bin` was relaunched twice on the A4000 this session (see
launch history below), most recently with the `PickTargetOption` +
value-loss-weight fixes applied. **Decision (2026-07-09, after Opus 4.8's
review): move to a 4090.** The live telemetry (100% GPU util the entire
time, wall-clock ≈ batches × s/batch, game-gen chunk regen is seconds
against a ~4hr epoch) shows this stage is GPU-*compute*-bound, not the
"VRAM problem, GPU tier doesn't matter" conclusion ASTRA7.md §10 reached —
that conclusion came from the old 4GB Atlas card's numbers and didn't hold
up against real batch-256-on-A4000 data. A 4090 should cut ~60hr/~$10.30
down to ~15-18hr/~$5-7. **Waiting on a new pod + SSH access before
relaunching there** — the A4000 pod is still up and still billing in the
meantime (RunPod bills pod uptime, not process activity, so there's no
cost reason to keep the training process running vs. idle — will decide
whether to keep the A4000 pod around as a cheap smoke-test box, matching
the old Atlas role, or terminate it once the 4090 takes over).

```
novacana-ml7 bc --out models/astra7_bc_scaled.bin \
  --chunk-games 8000 --epochs 15 --batch-size 256 --val-games 200 --players 4
```

**Launch history** (all three issues diagnosed and fixed before/during this session):
- `--chunk-games 20000` → OOM-killed by the pod's 58GB RAM cgroup cap
  during game generation. ASTRA7.md §8.3 already documented an 8-10K/chunk
  ceiling — missed it on the first launch.
- `--batch-size 512` → CUDA OOM (~15.55GB attention-activation footprint
  for this architecture at that batch size, doesn't fit in 16GB with
  headroom). Original ASTRA7.md estimate (~3GB at batch-512, from the old
  4GB Atlas card) was roughly 5x too low for the real number.
- `--batch-size 256` → stable on the A4000, 13.5GB/16.3GB VRAM, 100% GPU
  util — but see the migration decision above.

**Live dashboard**: `http://localhost:8420` on hermes (local stdlib Python
server, polls the `astra7` SSH alias every 6s, not a static snapshot —
script at `/tmp/.../scratchpad/astra7_dash/server.py`, must be running for
the page to load). **Will need repointing to the new pod's SSH alias once
the 4090 migration happens.**

## Budget

$25 USD loaded on the RunPod account. Revised estimate for this run on a
4090: ~$5-7 (down from ~$10.30 on the A4000). Opus 4.8 flagged that the
$15-25 total-campaign estimate (ASTRA7.md §10) inherited the same "compute
is cheap" premise the BC telemetry just broke — the AZ league stage in
particular (100-200 iterations × 150-200 games × ≥400 MCTS sims each) is
GPU-inference-bound self-play, not free, and is where AZ campaigns
typically spend most of their money. **Treat the $15-25 figure as
optimistic; budget for a top-up before starting the AZ league rather than
mid-run.**

## Operating notes (from the 2026-07-09 review, apply going forward)

- **Don't over-grind BC.** Its ceiling is roughly the teacher's (AdvCPU)
  level — the 55-65% target is reached in TD-value + AZ, not BC. Watch
  per-epoch val_acc; stop once it flattens or clears the §4.1 gate rather
  than forcing all 15 epochs. The old Atlas run plateaued by epoch 3-4
  even data-starved.
- **200 val-games is fine for the in-run token-accuracy metric** — it's
  not a §2 win-rate violation, that gate only applies to the §4.1
  milestone's 600/1000-game win-rate eval. Don't let a cheap in-run number
  substitute for that later.
- **Watch s/batch trend over the full run, not just a temperature
  snapshot.** 93°C at 8 minutes in isn't a throttle signal by itself, but
  thermal drift on a shared pod over many hours could quietly inflate both
  time and cost — the tell would show up as rising s/batch, not just a
  temperature number.
- **rsync-not-git is a single point of failure.** The repo's substantial
  local changes exist only on hermes's working tree + an ephemeral rented
  pod. Worth a throwaway WIP-branch commit as a safety net (not a "commit
  to main" §8.6-gated action) — ask before doing this, per this project's
  commit policy.

## Next steps

1. Get SSH access to the new 4090 pod, migrate the build + relaunch the BC
   run there.
2. Let it finish; run `novacana-ml7 fidelity` on the resulting best
   checkpoint before trusting it — including a fresh look at
   `PickTargetOption` now that its encoding is fixed.
3. Hit the real §4.1 gate (≥25% floor, 30-40% target, full-ML 1000-game
   eval).
4. Continue TD-value and the AZ league on the **same** rented instance —
   no hermes+Atlas split, no `server.rs`/`client.rs`. Re-run the
   value-rank diagnostic first (the `hands_dealt` engine bug window may
   have contaminated any earlier reading of it). Budget a top-up before
   this stage, not during it.
