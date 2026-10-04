# Astra-7 - Current State

Snapshot as of 2026-07-10, repo relocated 2026-07-15, optimization pass
2026-07-16. Full design/history in `ASTRA7.md`; this is the training-
pipeline-focused status doc.

**OPTIMIZATION PASS (2026-07-16)** - consulted Fable 5 for engine/training
optimizations given the pod→brother's-PC compute downgrade (13-14 vCPUs/
16GB VRAM/58GB RAM → 6 WSL2 vCPUs/10GB VRAM/12GB RAM cap), then implemented
its findings directly. All changes below verified: `cargo check` clean on
both `novacana-engine` and `novacana-ml7` (default features - no LibTorch
on hermes, `--features tch` still unverifiable here as before), full
26-test `novacana-ml7` suite green (was 25; added one), 6-test
`novacana-engine` suite green including both card-conservation invariant
checks.

Implemented (safe/mechanical):
- **AZ `--batch-size` default 256→128** (`main.rs::cmd_az`) - 256 was sized
  for a 16GB A4000 (measured 13.5GB peak), would OOM the 3080's 10GB.
- **`td-value --lr` now mode-dependent**: 0.001 (frozen trunk, unchanged)
  vs **0.0001 with `--unfreeze-trunk`** (`main.rs::cmd_td_value`) - the
  unfrozen-trunk path (2026-07-10 postmortem fix) had never actually run
  against real data; it would have inherited the frozen-path's 10x-too-hot
  default and likely re-failed the gate for a config reason indistinguishable
  from an architecture one. Effective lr now printed at startup.
- **`[profile.release] lto = "thin", codegen-units = 1`** added to the
  workspace `Cargo.toml` - free compile-time-for-runtime-speed trade,
  relevant since the brother's PC rebuilds from source.
- **Self-play/eval concurrency is now a CLI flag**, not a hardcoded 32:
  `--gen-concurrency` on `bc`/`winrate`/`token-scan`/`td-value`/
  `value-rank`/`az`, threaded through `generate_td_samples`,
  `generate_az_games`, `eval_win_rate`, `run_value_rank`,
  `scan_live_play_for_collisions` (all gained a `concurrency: usize` param;
  public call sites all updated). Default is
  `batch_infer::default_concurrency()` = `max(8, 2x cores)`, adaptive per
  machine instead of assuming ~13-14 vCPUs. `BatchedNet`'s `max_batch` is
  now tied to the same value at every call site (a batch can never exceed
  concurrent-caller count anyway).
- **Card registry is now `Arc<HashMap<CardId, CardKind>>`**
  (`engine/src/deck.rs`, `engine/src/game.rs`) instead of a plain
  `HashMap` cloned on every `GameState::clone()` - this happens on the
  order of millions of times per AZ iteration (every MCTS leaf eval clones
  state). Now a refcount bump instead of a 90-entry deep-clone+rehash.
  Write-once (only `Deck::add_many`, only during `build_classic`, before
  any Arc clone exists) via `Arc::get_mut`.
- **AZ replay window**: `Vec<AzSample>` → `VecDeque<AzSample>`
  (`main.rs::run_az`) - the sliding-window trim was memmoving up to ~150K
  survivors every iteration; VecDeque's front-drain is O(1) amortized per
  element. Default window also cut 300K→150K samples (~0.95GB→~0.475GB) to
  fit the brother's PC's 12GB WSL2 cap alongside LibTorch/CUDA context.
- **AZ `--expert-share` default 0.15→0.05** - ExpertCpu seats cost ~325x
  AdvCPU inside the same bounded thread pool as every other concurrent MCTS
  seat; on 6 vCPUs one Expert seat can stall a whole iteration behind it.
  Diversity value is real (§4.3) but not load-bearing; override on a box
  with cores to spare.
- **`BatchedNet::forward_single` reuses a thread-local channel** instead of
  allocating a fresh `mpsc::channel()` per call - only matters at AZ scale
  (~6M calls/iteration) but cheap and safe to do (each call is a strictly
  synchronous send-then-recv round trip, never reentrant per-thread).

Evaluated and deliberately NOT changed:
- **`Deck::draw`'s `remove(0)`** and **`play_history`'s `remove(0)`**: both
  bounded (90 cards / 64 entries) - real cost is a few-KB memmove, not
  worth the risk of inverting "top of deck" semantics across 8 methods
  (card-conservation-sensitive) or breaking `GameView::recent_plays`'
  slice-indexing consumer in `tokens.rs`.
- **PUCT sim-budget scaling** (`sims.min(n_legal x K)`) and **discovery-
  action dedup** (`enumerate_solar_system_actions`, would shrink C(n,k)
  blowups by treating same-type-multiset discovery combos as one action):
  both flagged by Fable as needing live A/B verification / a token-scan +
  fidelity recheck before adopting, neither of which is possible without
  real training-scale runs. Left as documented follow-ups, not implemented
  blind.

**Tried, measured, reverted**: a `FOLLOWUP_TOPK`-style AdvCPU-score
prefilter on `ExpertCpuPlayer::choose_action_with_state`'s primary search
loop (`engine/src/cpu.rs`), meant to cut full-sim cost on large legal sets.
Added a new permanent regression test,
`az::tests::expertcpu_vs_advcpu_teacher_strength` (1 ExpertCpu seat vs 3
AdvCPU, n=2000 - the exact "47.5%±0.9" config from recorder.rs's module
doc), specifically to measure this before trusting it. Result: K=10 →
38.15%, K=20 → 36.45% - a real ~9-11pp regression, not noise, and NOT
recovering as K widened (ruling out "K too small" as the explanation).
Reverted to the original exhaustive search; kept the new test since it's
useful regardless.

**Separately surfaced, UNRESOLVED - flagging for a human**: even the fully
reverted code (confirmed via `git diff`: this session's `cpu.rs` changes
are comments-only) measured **39.55%** on that same test, not the
documented 47.5%±0.9. That's an ~8pp gap, well beyond n=2000's ~1.1%
sampling noise. Not caused by anything in this optimization pass. Possible
explanations, none confirmed: the original 47.5% measurement used
different conditions than this test's `max_turns(400)`/methodology; the
figure was measured at a different point in the codebase's history and
something upstream of this session already shifted it; or genuine
run-to-run variance beyond the originally-stated ±0.9 tolerance. This
matters because 47.5% vs ~40% is a meaningfully different BC teacher
quality/win-rate-ceiling story - worth deciding whether ASTRA7.md §1.1 and
recorder.rs's module doc need their documented figure updated, or whether
this needs a deeper investigation before the next BC run. Left as-is,
not silently reconciled either direction.

**REPO MOVED 2026-07-15**: from `~/Nextcloud/Documents/Creative/Novacana/novacana-engine`
(Nextcloud-synced, was uploading `target/`, `data/`, checkpoints etc. to the
cloud sync) to `~/Projects/novacana` on hermes - plain local path, not
synced. Same git history/remote, uncommitted work carried over via a plain
`mv` (same btrfs subvolume, instant). Claude Code's project session history
was moved too (`~/.claude/projects/-home-breadway-Projects-novacana`). If
any doc/script below still references the old Nextcloud path, treat
`~/Projects/novacana` as current.

**CURRENT FOCUS (2026-07-15): brother's PC (RTX 3080 10GB, i7-10700K, 16GB
RAM, Windows) as free training compute**, not the RunPod pod below - the pod
work is paused (see "pod lost, budget break" below) until the user has
budget to resume it.

**Install bundle DONE, in `brothers-pc/`** (repo root) - everything needed
to stand the brother's PC up as a training node, ready to walk through
today: `README.md` is the master doc. `windows/setup-wsl2.ps1` +
`windows/dot-wslconfig` get WSL2/Ubuntu running with sane RAM/CPU caps.
`linux/bootstrap.sh` installs Rust + LibTorch 2.9.0+cu126 (same version
already proven on Atlas) and builds `--features tch`; ran the equivalent
test suite locally first (24/24 passing, ~192s, exercises the batched-
inference fix under real concurrent self-play) before shipping it as the
sanity-check step. `linux/launch-bc.sh` is the tuned command:
`--batch-size 128` (extrapolated from the A4000's measured 13.5GB@bs256,
roughly linear scaling, target ~6.75GB to leave headroom under the 3080's
10GB + Windows WDDM overhead), `--chunk-games 1500` (RAM-sized for 16GB
total system RAM vs. the pod's 58GB cgroup - this is a guess extrapolated
from byte-level Sample-struct math, not a live-measured number, flagged as
tunable in the script's comments), `--checkpoint-every 200`. Auto-resumes
if a checkpoint + resume-metadata sidecar already exist.
`windows/start-training.bat`/`stop-training.bat` wrap it in tmux (survives
closing the window) and wire to the verified graceful-shutdown mechanism.
`package-source.sh` (run on hermes) produces the transfer tarball -
already built once, 216KB, verified it extracts and resolves as a valid
Cargo workspace. First transfer has to be USB/file-share (Tailscale isn't
joined on that machine yet); the README documents the rsync-based re-sync
path for once it is.

**2026-07-16 additions to the bundle**: (1) tarball was stale (missing
`az.rs`/`spectator.rs`/`main.rs`/`Cargo.toml` changes) - rebuilt, now
240KB, re-verified it extracts to a valid workspace with all current
`ml7/src` files. (2) `linux/join-tailscale.sh` - non-interactive Tailscale
join using a pre-generated auth key (expires ~2026-07-23, reusable until
then; revoke from https://login.tailscale.com/admin/settings/keys once
joined), registers as `brotherspc`. This file has a live secret baked in -
**not committed to git**, `brothers-pc/` stays untracked. (3) Dashboard
now supports **local mode**: `tools/dashboard_server.py` gained
`ASTRA7_LOCAL=1` (probes the local filesystem/processes via a shell
instead of SSH) and binds `0.0.0.0` by default in that mode instead of
`127.0.0.1`. Lets the dashboard run directly on the brother's PC (inside
WSL2) so both people can watch - him via `http://localhost:8420`
(WSL2's automatic port forwarding), the user remotely via
`http://<brotherspc-tailscale-ip>:8420` once Tailscale's joined, no SSH
tunnel needed since Tailscale runs inside the same WSL2 instance as the
server. Wired up via `windows/start-dashboard.bat` /
`stop-dashboard.bat` (own tmux session, independent of training's).
`tools/dashboard_server.py` is now included in the packaged tarball
(previously excluded - `package-source.sh` updated). All new files
syntax/extraction-verified but **not yet live-tested against a real
WSL2 + Tailscale combination** - flagged in the README as a possible
first-try snag, with an SSH-tunnel fallback documented.

**Verified this session**: the pause/resume mechanism end-to-end (not just
the save half, which was already proven pre-pod-loss) - a real `--resume`
run correctly restored `epoch`/`global_step`/`best_acc` from the sidecar
file and continued training. Minor cosmetic bug noticed, not fixed (low
priority): the "Shutdown requested" message from `install_shutdown_flag`'s
ctrlc handler prints twice on a single SIGTERM in some runs - cosmetic
only, doesn't affect the actual save/exit behavior.

**Dashboard redesign + repoint DONE** (`tools/dashboard_server.py`) -
"mission control" theme: starfield canvas background, MET (mission
elapsed time) clock, radial GPU-utilization gauge (color logic
deliberately inverted vs. a typical meter - for GPU util, high is nominal
and low is the actual problem, see the code comment referencing the
batched-inference bottleneck fix earlier this session), VRAM meter,
telemetry-styled panel names, flight-log/comms-log framing. Repointed off
the RunPod `astra7` SSH alias onto `ASTRA7_SSH_HOST` (default
`brotherspc`, override via env var) + `ASTRA7_LOG_PATH` (matches
launch-bc.sh's `training.log`); removed RunPod-specific cost tracking
(not rented compute anymore). Verified end-to-end with a fake SSH backend
feeding fixture log data through all three status states (standby/
training/anomaly) - screenshotted in a real browser via desktop-control,
not just eyeballed as HTML. Caught and fixed two real bugs in the
process: (1) the heartbeat regex required exactly one space between
`policy=...)` and `value=...` but the actual log format has two - was
silently matching zero heartbeats; (2) initial GPU gauge color logic
copied a generic high-value-is-bad meter pattern, which is backwards for
utilization specifically.

USB transfer done (brothers-pc/ copied to a new exFAT partition on the
`BOS_202606` install stick - added in unallocated space at the end of the
disk, did not touch the existing ISO9660/EFI content, verified before
formatting). Stick was later fully wiped by the user and re-provisioned
from scratch (fresh partition table, single exFAT partition spanning the
whole disk) - same file set copied back over.

**Dashboard "make it fun" pass, part 1 (2026-07-15)** - consulted Fable 5 for
ideas (per user's explicit ask), picked the front-end-only subset that
needed zero Rust-side instrumentation (deferred: card-usage-spectrum
stats dump and full game-replay theater, both needed new Rust
instrumentation - built in part 2 below, same day, once asked for). Added to
`tools/dashboard_server.py`: an **epoch constellation** (each completed
epoch becomes a permanent star on the starfield canvas - brightness =
val accuracy, hue = win rate, best-ever epoch gets a gold pulse - layered
into the existing ambient-twinkle canvas/RAF loop, not a new layer),
**milestone banner + starburst** on a new best val-acc, **mission
patches** (four win-rate threshold badges: First Contact/Established/
Competitive/Dominant at 10/25/40/55%, lighting up + flashing once each as
crossed), a **propulsion/thruster panel** (small ship + particle exhaust
whose color/steadiness reacts to the loss trend - teal steady burn when
improving, amber flicker when flat, red sputter when regressing), a
**T-minus orbital-insertion countdown** (replaces the plain ETA stat
tile, ticks client-side like the MET clock), and **procedural comms
chatter** (dim/italic flavor lines seeded from real metrics, interleaved
into the log tail, clearly `[COMMS]`-tagged so they're never mistaken for
real log output). All verified end-to-end with a fake-SSH fixture feeding
a progressing epoch sequence through a real browser (screenshotted via
desktop-control) - confirmed patches light up in order as win rate
crosses each threshold, constellation/thruster/countdown all update live,
no console errors.

**Dashboard "make it fun" pass, part 2 (2026-07-15)** - the two Rust-
instrumentation items from part 1, plus real-time push:

- **`ml7/src/spectator.rs`** (new module, 25 tests → 25 tests, all
  passing): `record_spectator_game()` plays one extra game per win-rate
  eval (seat 0 = greedy net, seats 1..N = AdvCPU - mirrors
  `az::eval_win_rate`'s own setup, reusing its `GreedyNetPlayer` - made
  `pub(crate)` for this), recording every `choose_action`-level decision
  (the ones that place/attach a real card - chain-response/decide/target
  sub-decisions skipped, same scoping `evals.rs`'s `ValueRankRecorder`
  already established) as a `Frame { player, action (human-readable,
  built from `Action` + the game's own `CardId -> CardKind` registry),
  board (full per-player `StarChart` + hand sizes + dark-matter-void,
  straight off `GameState` - no reconstruction needed, the engine already
  tracks all of this) }`. `card_spectrum` comes directly off
  `GameState::played_counts` (already exists, already exactly
  `CardKind::type_idx`-indexed - no new counting logic needed). Wired into
  `run_bc`'s existing win-rate-eval call site in `main.rs`
  (`write_spectator_outputs`) - writes `models/spectator_game.json`
  (overwritten each cycle) and appends to `models/card_spectrum.jsonl`.
  Never fatal on write failure (dashboard cosmetics, not training).
- **Dashboard**: **Replay Theater** panel - auto-advances through the
  latest spectator game's frames (~2.2s/frame, loops), rendering each
  player's Star Chart (star colored/sized by name, discovery slots as
  filled/empty dots) with the active player highlighted and the current
  action captioned above. **Stellar Spectrum** panel - stacked bar of the
  5 `CardKind` categories (Star/Discovery/Flare/Omen/Fracture, aggregated
  from the 29-slot array using `CardKind::type_idx`'s documented slot
  ranges) per epoch, from `card_spectrum.jsonl`.
- **Real-time polling → real-time push**: previously the browser polled
  `/api/data` every 6s, and `get_data()` made 3 separate SSH subprocess
  calls per poll. Now: (1) one SSH round-trip per cycle (log + nvidia-smi
  + ps + spectator-game + card-spectrum, all in one delimited script) with
  SSH `ControlMaster`/`ControlPath` multiplexing so repeat round-trips
  reuse a warm connection instead of a fresh handshake; (2) a background
  thread does that fetch on a 2s cadence (`ASTRA7_POLL_INTERVAL_S`) and
  caches the result; (3) the browser gets it pushed via Server-Sent
  Events (`/api/stream`, `EventSource`) the instant the cache updates -
  no client-side polling delay, and `EventSource` auto-reconnects on its
  own if a flaky Tailscale link drops. `/api/data` stays as an instant
  (cache-only, non-blocking) fallback/curl-friendly endpoint.
- Verified end-to-end with a fake-SSH harness that actually executes the
  combined fetch script locally against hand-built fixture JSON matching
  spectator.rs's exact serde field names (chose this over waiting on a
  real CPU-backend training run for fixture data - confirmed via `ps`
  that a real 30-game/1-epoch run was genuinely still computing, not
  stuck, after 10+ minutes on CPU/ndarray; CPU training is documented
  elsewhere in this file as "cheap sanity check only," not meant for even
  modest runs) - screenshotted via desktop-control, confirmed the theater
  correctly loops through all frames with accurate board state and
  active-player highlighting at each step, and the spectrum chart renders
  the growing stacked bars correctly.
- Housekeeping note: mid-session, Nextcloud's desktop client was found
  running and had recreated an empty stub (`.git/`, `.claude/
  scheduled_tasks.lock`) at the OLD pre-move repo path
  (`~/Nextcloud/Documents/Creative/Novacana/novacana-engine`) - harmless
  clutter, not data loss (the real repo at `~/Projects/novacana` was
  unaffected throughout), but worth knowing: a plain `mv` doesn't tell
  Nextcloud "this was intentionally relocated," so it may keep syncing
  stray content down to that old path unless the user removes it from
  Nextcloud's sync selection. Flagged to the user, not acted on
  unilaterally.

**Repo moved 2026-07-15**: see the banner at the very top of this file.

**SESSION PAUSED 2026-07-10 - pod lost, budget break. Read this first.**
The 4090 pod that ran everything tonight (BC + both TD-value attempts) was
stopped to save money, then failed to restart, then RunPod's "migrate to a
new pod" option produced an empty pod (verified: `/workspace` and `~` both
empty, no trace of `novacana-engine` anywhere), and a second CPU-only
recovery attempt on the original pod also came back empty. **Both trained
checkpoints from tonight are gone** - `astra7_bc_scaled_best.bin`,
`astra7_bc_epoch6_final.bin`, `astra7_td.bin`, `astra7_td_best.bin` only
ever existed on that pod's disk and were never rsynced back to hermes.
Root lesson, apply going forward: **rsync checkpoints back to hermes after
every major milestone, don't leave them solely on rented compute.**

**What survived, unaffected**: all source code (hermes is the source of
truth, pod was always an rsync target) - including tonight's full set of
fixes, all verified locally (see "Self-play/search batched inference"
below for the batching + deadlock fix, and "TD-value stage" for the
unfrozen-trunk training code). The older v5/v6/`astra6_*` checkpoints and
one early interim `astra7_bc_scaled.bin` are also confirmed present
locally. Nothing about the CODE needs redoing - only the training compute.

**A fresh, working replacement pod is confirmed available**: the
"migrated" pod (`hccjlrky6182nm-64412378@ssh.runpod.io`, RunPod SSH proxy)
has a real RTX 4090 (24GB), 32 vCPUs, 125GB RAM - empty disk, but
functional. **Known gotcha hit while setting it up**: RunPod's
`ssh.runpod.io` proxy requires a PTY and injects a decorative banner into
the stream, which breaks `rsync`'s protocol handshake ("protocol version
mismatch -- is your shell clean?") even with `-e "ssh ..."` and both with
and without `RequestTTY=force`. **Not yet resolved** - next session needs
to find a workaround (a direct IP:port from the RunPod dashboard instead
of the ssh.runpod.io proxy is the most likely fix, since that's what
worked cleanly for the previous pod all night; or tar+ssh-cat instead of
rsync; or check if RunPod exposes a non-proxied port for this pod) before
the repo can be synced over and BC training restarted.

**Next steps on resume**: (1) solve the rsync-over-proxy problem or get a
direct connection string, (2) rsync the repo to the new pod, (3) install
Rust + LibTorch matching versions used before (`ml7/build.rs`'s CUDA-
linking fix, see `tch-rs-cuda-linking-gotcha` memory), (4) build with
`--features tch`, (5) run the LOCAL test suite equivalent on the pod too
(cheap sanity check before spending real GPU time), (6) restart BC
training fresh (`Net::fresh()`, ExpertCpu teacher, the exact command is in
the "Run history" table below, row #10) - the whole night's BC→TD-value
arc needs to be redone from scratch, but with tonight's fixes already
in place so it should go smoother (batched inference is faster + no
deadlock risk once concurrency is high; TD-value has the unfrozen-trunk
option ready to go immediately after BC finishes, skipping the two failed
frozen-trunk attempts and one dead end from earlier tonight).

---

**Historical status note (pre-pod-loss, 2026-07-10): BC stage was DONE.
TD-value had failed its §4.2 gate TWICE** - λ=0.9 (3 passes): 50.8%.
λ=1.0 (1 pass, the first attempted fix): 44.8%, actually worse,
statistically indistinguishable from an untrained random net. Required
gate: ≥65%. **Root cause had moved from "hyperparameter" to
"architecture"**: two Fable 5 consults (full reasoning trail below)
converged on the frozen BC trunk's pooled features being the binding
constraint - the value head, whether bootstrapped or pure-MC, whether one
pass or three, couldn't get past the mid-50s ranking signal available in
policy-shaped features. The fix implemented (code-complete, verified
locally, never got to run on real data before the pod was lost): unfreeze
the trunk for TD-value training (new train step, low lr ~1e-4, a policy-
distillation anchor to protect the pointer head BC already trained) plus
mixing AdvCPU seats into TD self-play generation (the value head
previously never trained on the distribution the gate scores it against).
Full detail in the "TD-value stage" section below - all still accurate as
a plan, just needs re-running against fresh training data.

Separately, tonight also produced a **batched-inference refactor**
(`ml7/src/batch_infer.rs`, new) fixing a real measured inefficiency: the
TD-value run showed only ~3.6/13-14 usable vCPUs and ~30% GPU util active
at once, with most OS threads parked on `futex_wait_queue` - self-play's
per-decision GPU calls were serializing one-at-a-time through the CUDA
driver instead of batching. Fixed by coalescing concurrent games'
single-decision forward calls through a shared micro-batcher. Threaded
through `mcts.rs`, `az.rs`, `td_value.rs`, `evals.rs` - public API
surfaces (`generate_td_samples`, `generate_az_games`, `eval_win_rate`,
`run_value_rank`, `scan_live_play_for_collisions`) kept their existing
signatures unchanged, so `main.rs` needed no changes. This matters most
for the upcoming AZ stage, where MCTS does >=400 leaf evals per real
decision (vs TD-value's 1) - see "Self-play/search batched inference"
section below for the full writeup and verification status.

*(Historical note, migration already complete):* An independent
review (Opus 4.8, reading this doc + ASTRA7.md + GPU_RENTAL.md) pointed out
that the live BC telemetry (100% GPU util, wall time = batches × s/batch,
game-gen regen is seconds against a 4hr epoch) contradicts §10's "VRAM
problem not compute problem, GPU tier doesn't matter" conclusion for this
stage - it's compute-bound, and a 4090 should cut the ~60hr/~$10.30 BC run
to ~15-18hr/~$5-7. Decided: switch. Waiting on a new pod + SSH access.

## Infrastructure

**Rented pod**: RunPod, RTX A4000, SSH alias `astra7` (`root@193.183.22.56 -p 1094`,
key `~/.ssh/id_ed25519_astra7`).

| | |
|---|---|
| GPU | RTX A4000, 16GB VRAM, driver 550.144.03 (CUDA 12.4 max) |
| RAM | cgroup-capped at **~58GB** (host has 503GB shared across tenants - do not assume full host RAM is available) |
| Disk | 35GB overlay, ~27GB free |
| Rate | $0.17/hr |
| vCPUs | 14 (game generation is CPU/rayon-bound, this matters) |

**Software stack**: Rust 1.96.1 (rustup stable), LibTorch **2.9.0+cu126**
(matched to `torch-sys 0.22.0`'s pinned `TORCH_VERSION`; cu126 works fine on
the cu124-capable driver via CUDA minor-version compatibility - cu121/cu124
builds no longer exist for LibTorch 2.9.0), burn 0.21 + burn-tch, built via
`ml7/build.rs`'s CUDA-linking fix (`--no-as-needed` around `-ltorch_cuda`,
see `tch-rs-cuda-linking-gotcha` memory).

Repo synced to `~/novacana-engine` on the pod via `rsync` (not git - repo
has substantial uncommitted local changes). Legacy `data/*.bin` (v5/v6
pretraining corpora, ~7.9GB) deliberately **not** synced - Astra-7 generates
its own self-play data on the fly, no static corpus dependency.

## Pipeline status

- Token schema (`tokens.rs`), transformer net (`net.rs`, D_MODEL=256,
  N_HEADS=8, N_LAYERS=6), BC recorder (`recorder.rs`) - built, tested (11/11
  unit tests passing).
- `novacana-ml7 bc` - chunked-streaming BC trainer with warmup+cosine LR
  (added this session), best-checkpoint tracking (bug-fixed this session -
  see below), heartbeat logging.
- `novacana-ml7 fidelity` - per-`ActionKind` accuracy/CE diagnostic tool.
- **Not built** (deliberately, per independent review): `server.rs` /
  `client.rs` - the hermes+Atlas LAN-split architecture was scrapped in
  favor of running BC → TD-value → AZ league all single-box on one rented
  instance.
- **TD-value stage + AZ league - code DONE, tested, NOT yet run for real
  (2026-07-09).** `novacana-ml7 td-value` (frozen-trunk value-head training
  off TD(λ=0.9) targets), `novacana-ml7 value-rank` (the §4.2 gate
  diagnostic), and `novacana-ml7 az` (MCTS self-play league, restart-from-
  best) are all implemented, unit/integration tested (21/21 tests pass,
  `cargo clippy` clean), and CLI-smoke-tested end-to-end against a
  throwaway untrained checkpoint - all three ran correctly (self-play,
  training, snapshotting, eval, best-checkpoint saving all executed; the
  gate correctly FAILed on the undertrained checkpoint, confirming the gate
  logic itself, not just the plumbing, works). Blocked on an actual BC
  checkpoint existing (see the migration status above) before either
  stage's real gate can be measured. Full design-decision writeup
  (TD(λ) formulation, MCTS's one-level-deep design, trunk-freezing
  mechanism, opponent-mix shares, restart-from-best trigger) is in
  ASTRA7.md §11. `cargo check -p novacana-ml7 --features tch` could not be
  verified in this dev environment - no LibTorch install on hermes, and
  `torch-sys`'s build script needs real headers even just to type-check
  (pre-existing constraint, same as `bc`/`fidelity` already had); none of
  the new code references `tch` types directly, so it should compile fine
  once run somewhere LibTorch exists, but this is unverified, not assumed.

### Bugs found + fixed this session

1. **Best-checkpoint silent collision - fixed, then found still partially
   broken, now properly fixed (2026-07-09).** `format!("{out_path}.best")`
   collapsed onto the final save path because burn's `FileRecorder::save`
   uses `PathBuf::set_extension()`, which *replaces* rather than appends an
   extension. The first fix, `format!("{out_path}_best")`, only works when
   `out_path` has no extension - an `out_path` ending in `.bin` (exactly
   how the 4090 run was launched: `--out models/astra7_bc_scaled.bin`)
   retriggers the identical collision, because `"foo.bin_best"` still has
   exactly one dot, which `set_extension("bin")` treats as *the* extension
   and replaces wholesale, collapsing back to `"foo.bin"`. Verified via a
   standalone `rustc` repro, not just reasoning. Real fix: derive
   `best_path` from `out_path`'s `file_stem()` (stripping any extension
   first) before appending `_best`, so the constructed path never has more
   than the one dot `set_extension()` itself adds back. Confirmed correct
   for both extension and no-extension `out_path` inputs. The earlier
   Atlas 7-epoch run got lucky (its true best epoch was also its last) -
   twice now, checkpointing has been silently non-functional without
   affecting that particular run's outcome.
   **The 4090 run currently in progress was launched with the
   still-broken first fix** - harmless so far (the single file on disk is
   correctly the running best-checkpoint, since every best-save collides
   onto the same path the final save will also use), but would have
   silently lost the true best checkpoint at the very end if epoch 15
   isn't the best epoch. Mitigation: not restarting the live run (cheap
   progress already made); watching per-epoch val_acc and stopping
   manually once it plateaus rather than letting the unconditional final
   save run - sidesteps the bug for this run entirely. Source fix synced
   to the pod for the next relaunch (TD-value stage or a BC restart).
2. **`DecideOption` encoding gap** - `encode_action_tokens_decide` had no
   positional field, causing token collisions between different options
   sharing the same keyword-flag bucket. Fidelity diagnostic caught it
   (60.0% accuracy on 43% of all samples). Fixed by adding slot-index +
   option-count fields (backward-compatible, no shape change). Directional
   validation: 72.0% vs 60.0% on a quick fresh checkpoint.
3. **Chain-response fidelity - investigated, NOT a bug.** Original concern
   (§1.1) was that the model can't distinguish past chain responses from
   primary plays. Fidelity diagnostic showed `ChainResponsePass` 100.0%,
   `ChainResponseCard` 95.4% - among the best-performing kinds. No fix
   needed, question resolved negatively.
4. **`PickTargetOption` collision (19.7% accuracy) - root-caused and fixed.**
   Initially deferred as "all 7 slots in use, no room to fix" - wrong, only
   checked the field list, not the actual generator. Opus 4.8's review
   (reading this doc) pointed out the 19.7% number looked exactly like
   another token collision, not data starvation, and was worth an hour to
   check before spending a 60hr run + TD-value + AZ on top of it. Traced to
   `disc_target_options`/`chart_target_options` (`engine/src/game.rs`):
   they emit one `TargetOption` per discovery in a star system (plus the
   star itself), but `encode_one_target` only encoded system-level info
   (star name/class/fill count) - every discovery in the same system, and
   the star alongside them, collided into an identical token. Fixed using
   the one free field in each branch: a star/discovery position index for
   `StarChartCard`, a hand-slot index for `HandCard` (duplicate card types
   had the same gap). Not yet re-measured via `fidelity` at scale - do that
   once a checkpoint trained on the fixed encoding exists.
5. **`encode_one_action` - the `choose_action` encoder - had the same
   missing-identity gap, systemically, across nearly every `Action` variant
   (found 2026-07-09).** Fidelity's aggregate accuracy looked healthy
   (93.1% total) because it only ever samples AdvCPU-typical states; the
   bug only shows up once a policy plays its own decisions and the game
   state drifts. Built a live-play collision scanner (`novacana-ml7
   token-scan`, `az::scan_live_play_for_collisions` /
   `scan_advcpu_training_data_for_collisions`) that checks every single
   decision's legal-action tokens for exact duplicates - found via a
   30-game live-play scan that **10.4% of all decisions** (361/3476)
   collided. Root causes, three distinct mechanisms:
   - `PlayStar`/`PlayFlare`/`PlayFracture`/`PlayOmen*`/`PlayProtostar*`/
     `DiscardCandidate`/`ChainResponseCard`: card *type* encoded, never card
     *identity* - two legal actions playing different physical copies of
     the same card type collided. Fixed: hand-slot index in the one free
     field per variant.
   - `PlaySolarSystem`: `Action::PlaySolarSystem { star, discoveries: Vec<CardId> }`
     - `discoveries` was **never read at all**. Playing the same star with
     different simultaneously-attached discoveries is a real difference in
     game outcome, not a tie, but collided into one token. **Only
     partially fixed** - added `discoveries.len()` as a field, which
     disambiguates different-*count* combos but not same-count-different-
     *specific*-discoveries combos (confirmed still the dominant remaining
     collision source, ~439/568 in the post-fix AdvCPU-distribution scan).
     A full fix needs real per-discovery identity, which needs a
     field-budget redesign - deliberately not rushed, same judgment call
     as the original `PickTargetOption` deferral.
   - `PlayDiscovery`/`PlayOmen*`: target *system* only encoded by its
     bucket (star name/class/fill), not which specific system - two legal
     actions targeting different systems that happen to bucket identically
     collided. Fixed: a stable target-system position index (same pattern
     as `find_system_containing`, extended to return position).
   - `PlaySupernovas` ignored `gravity`/`gas` entirely (only read `fusion`)
     - fully fixed (all three are unique singleton cards per the game's own
     rules, so this was silently-wrong but likely never actually
     triggered - matches its 100% fidelity accuracy pre-fix).
   Quantified impact: AdvCPU-training-distribution collision rate went
   655/43,795 (1.50%) → 568/40,172 (1.41%) after the partial fix - real but
   modest, because `PlaySolarSystem`'s residual gap dominates.
6. **`PlaySolarSystem`'s residual gap - improved with a discovery-set hash
   signature (2026-07-10).** Same-count-different-specific-discoveries
   combos were still colliding after the count-only fix (bugs #5). Added an
   order-independent hash over the discoveries' card types into `fields[4]`
   (previously a redundant duplicate of `fields[3]` - "fill == class" for a
   simultaneous-complete play, carried no real signal - safe to repurpose).
   Lossy (two genuinely different combos can theoretically still hash to
   the same bucket) but drastically narrows the collision surface versus
   count alone. A fully exact fix would need real per-discovery identity
   fields, which the current 7-field budget has no room for - deliberately
   not pursued further, same judgment call as `PickTargetOption`'s
   deferral. Quantified on the real ExpertCpu training distribution (see
   bug #7 below - this measurement is post-teacher-swap): 0.92% collision
   rate (115/12,456, 30 games), down from the pre-hash-fix AdvCPU-
   distribution reading of 1.41%. Not a like-for-like before/after (both
   the teacher and the encoder changed between readings) but directionally
   confirms the hash is doing real work - `fields[4]` visibly varies
   across examples now instead of being constant.
7. **BC teacher switch to ExpertCpu was silently a no-op on the first
   attempt - found and fixed same day (2026-07-10).** Motivation: BC's
   win-rate ceiling is mathematically bounded by its teacher's own skill (a
   perfect clone of a policy IS that policy in that seat) - AdvCPU's own
   seat-0 baseline measures 20.0% (bug #4's control test), and a BC
   checkpoint 2 epochs into a fresh AdvCPU-imitation run had already
   converged to 19.6% (n=1000) with 13 epochs of budget left and nothing
   left to gain. ExpertCpu measures 47.5%±0.9 vs 3 AdvCPU - switching the
   teacher raises the ceiling by >2x. Swapped `recorder.rs`'s `BcRecorder`
   from wrapping `AdvancedCpuPlayer` to `ExpertCpuPlayer`, relaunched.
   **The swap compiled, ran, and produced plausible-looking numbers, but
   was actually still training against AdvCPU-quality decisions** -
   `ExpertCpuPlayer`'s real search only lives in `choose_action_with_state`
   and `respond_to_chain_with_state` (`engine/src/cpu.rs`); its plain
   `choose_action`/`respond_to_chain` always delegate straight to an
   internal `AdvancedCpuPlayer` fallback (`self.adv`), by design ("Without
   a GameState there is nothing to simulate"). `GameEngine::run()` always
   calls the `_with_state` variants when driving real games - but
   `BcRecorder` only overrode the plain, stateless methods, so the
   `Player` trait's default `choose_action_with_state` implementation
   (which just calls `self.choose_action(...)`) silently routed every
   decision through `ExpertCpuPlayer`'s cheap fallback path, never its
   real search. Caught via a Fable 5 consult flagging that a "0.2-0.3s for
   a 200-game ExpertCpu val-set" timing was arithmetically impossible
   given ExpertCpu's own benchmarked cost (~2.8 games/s single-threaded,
   ~909 games/s for AdvCPU - a ~325x difference) - confirmed by reading
   `ExpertCpuPlayer`'s implementation directly, not just re-measuring.
   Fixed by having `BcRecorder` override `choose_action_with_state` and
   `respond_to_chain_with_state` too, proxying to the teacher's real
   state-aware methods. Verified post-fix: 200-game val-set generation
   went from 0.2-0.3s (impossible) to 5.9s (~34 games/s, consistent with
   real search now running) in the actual production run. **The entire
   first "ExpertCpu-teacher" run (which reached epoch 1-3 before being
   stopped for the PlaySolarSystem fix, unrelated) never actually trained
   against ExpertCpu - discard any conclusions drawn from that run's
   numbers about what an ExpertCpu-taught BC checkpoint looks like.**

### Fixed this session, not yet stress-tested at scale

- **LR warmup + cosine decay** added to `run_bc` (`main.rs`) - 5% linear
  warmup (floor 50 steps) then cosine decay to 0, replacing the previous
  flat constant LR. Confirmed working correctly against the live run (LR
  ramping smoothly on schedule).
- **Value-loss down-weight (0.3)** - `bc_train_step` (`net.rs`) summed
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
  autocast - real risk of a silent quality regression, not a crash.
  Deferred pending an isolated correctness test; not worth gambling on the
  live campaign.

## Current training run - on the 4090, clean run in progress

Migrated A4000 → 4090 (Opus 4.8's review: this stage is GPU-*compute*-bound,
not the "VRAM problem, GPU tier doesn't matter" read ASTRA7.md §10
originally reached). Confirmed 3.3x speedup (0.221s/batch vs 0.73s/batch).
The A4000 pod was terminated.

**Raw log files do NOT survive relaunches** - `~/bc_scaled.log` on the pod
is truncated (`>`, not `>>`) on every launch, so only the currently-running
attempt's log exists on disk. The numbers below are preserved from this
document/conversation history since the original logs are gone.

### Run history (all attempts, in order)

| # | Config | Outcome |
|---|---|---|
| 1 | A4000, `--chunk-games 20000` | OOM-killed by the pod's 58GB RAM cgroup cap during game generation (ASTRA7.md §8.3 already documented the 8-10K ceiling - missed on first launch) |
| 2 | A4000, `--batch-size 512` | CUDA OOM (~15.55GB attention-activation footprint at that batch size - the old estimate of ~3GB, from the 4GB Atlas card, was ~5x too low) |
| 3 | A4000, `--batch-size 256` | Stable (13.5/16.3GB, 100% util) - killed after ~20min once the `PickTargetOption`/value-loss-weight fixes landed, to avoid training on stale encoding |
| 4 | A4000 → migrated to 4090, `--batch-size 512` | Also OOM'd on the 4090 (24GB) - crashed later in the step (backward/optimizer, not forward), confirming true full-step peak memory at batch-512 exceeds even 24GB for this architecture |
| 5 | 4090, `--batch-size 256`, fresh start | **Stable, real run.** Epoch 1: val_acc 92.19%. Epoch 2: 92.80%. Epoch 3: 93.35%. Epoch 4: 93.76% (all new-best). Killed mid-epoch-4-plus for the winrate instrumentation addition (see below) - but see finding below, this run's checkpoint was later found to have won only **14%** at a live win-rate eval (epoch 4 checkpoint, n=200, `eval_win_rate` - greedy net vs 3×AdvCPU), well under the 25% floor gate. Investigation into this number is what found the `choose_action` token-collision bug (bugs list, #5 above). |
| 6 | 4090, warm-started from run 5's epoch-4 checkpoint, `--epochs 11`, `--winrate-every 1` | Added per-epoch win-rate tracking + relaunched with the (at-that-point-partially-applied) collision fix. Only completed a partial epoch before being killed - user correctly flagged that warm-starting through a tokenizer change isn't a fully clean measurement (the model would need to unlearn a "this field is always 0" prior for a large fraction of tokens, not just adapt to a narrow patch) - see conversation reasoning, not in this doc's earlier form. |
| 7 | 4090, fully fresh (`Net::fresh()`, no `--model`), `--epochs 15`, current encoding fixes applied, AdvCPU teacher | Epoch 1: val_acc **94.73%**, policy loss 0.2485, win_rate 12.67% (n=150). Epoch 2: val_acc **95.52%**, policy loss 0.1496, win_rate 14.67% (n=150). Re-measured epoch 2's checkpoint at n=1000 for a trustworthy read: **19.60%** - essentially at the ~20% AdvCPU-equally-matched ceiling already (Fable 5's read: BC vs AdvCPU had converged, more epochs would buy ~nothing). Killed mid-epoch-3 to switch strategy per that consult. |
| 8 | 4090, warm-started from run 7's checkpoint attempt, then **killed and restarted fully fresh** | User correctly pushed back on warm-starting through yet another encoding change - same reasoning as run 6. Restarted `Net::fresh()`. |
| 9 | 4090, fresh start, **teacher switched to `ExpertCpuPlayer`** (bug #7 above), PlaySolarSystem hash-signature fix (bug #6) | **Looked normal, silently broken.** Compiled, ran, val-set generation reported 0.2-0.3s for 200 games - not questioned in the moment. Reached epoch 1-3 before a Fable 5 consult's arithmetic sanity-check flagged the timing as impossible for real ExpertCpu search. Root cause: bug #7 (`BcRecorder` only overrode the stateless `Player` trait methods; the engine always calls the `_with_state` variants, which silently fell through to `ExpertCpuPlayer`'s cheap `AdvancedCpuPlayer`-delegate path). **This run never actually trained against ExpertCpu - its numbers are meaningless for that question, discard them.** Killed once the bug was found. |
| 10 | **4090, fresh start, `BcRecorder` properly proxying `ExpertCpuPlayer`'s state-aware methods (bug #7 fixed)** | **Current run, in progress.** Verified pre-launch: 200-game val-set generation takes 5.9s (~34 games/s) - real search confirmed running. Trajectory (n=1000 win_rate reads, the reliable ones - n=150 in-loop reads are noisier, see below): Epoch 1: val_acc 65.71%, win_rate **14.20%**. Epoch 2: val_acc 66.92%, win_rate **21.70%** (n=150 in-loop read was 23.33%, ran high). Epoch 3: val_acc 67.42%, win_rate **19.20%** (n=1000; n=150 in-loop read was 21.33%). Fable 5 confirmed the epoch2→3 dip is statistically indistinguishable from noise (z≈1.4) - val_acc rose monotonically all 3 epochs, no overfitting signature. Decision rule set: tripwire through epoch 6, stop-and-move-to-TD-value if still 18-23% there, continue to 15 epochs if any read clearly exceeds 24% first (see "Next steps" below). Epoch 4: val_acc 67.83%, win_rate **23.10%** (n=1000; n=150 in-loop read was 27.33%, overestimated - same pattern as epoch 2's n=150 overshoot). Full n=1000 trajectory now: 14.20% → 21.70% → 19.20% → 23.10% - still consistent with fluctuation around a ~20-23% plateau, not a clear breakout past the 24% tripwire threshold. Epoch 5: val_acc 68.32%, win_rate **23.00%** (n=1000). Epoch 6: val_acc 68.70%, win_rate **22.90%** (n=1000) - the decision checkpoint. **Full 6-epoch win_rate trajectory: 14.20% → 21.70% → 19.20% → 23.10% → 23.00% → 22.90%.** Epochs 3-6 locked in a tight 19.20-23.10% band, never once breaking past 24% - about as clean a plateau confirmation as the data could give. **STOPPED BC HERE (2026-07-10, overnight autonomous run, user pre-authorized).** A second Fable 5 consult, given the complete 6-epoch picture plus the standing tripwire rule, confirmed decisively: "stop now and transition to TD-value... this isn't a borderline call... BC has extracted what it can; the gap to the 25% gate and the 47.5% ExpertCpu ceiling belongs to TD-value." Reasoning: val_acc kept climbing slowly (65.71→68.70%) while win_rate showed zero trend across 4 epochs at n=1000 (SE ~1.3%, so the epoch4-6 spread is pure noise) - the val_acc gains were fitting ExpertCpu's stochastic tail (the measured ~76.5% self-agreement noise floor), not transferring to actual play. Checkpoint safety-copied to `astra7_bc_epoch6_final.bin` before stopping (Fable 5's explicit operational note - confirm flush before pkill). val_acc is far lower than the AdvCPU-teacher run (65-67% vs 92-95%) - expected, ExpertCpu's search-based decisions are a much harder function to imitate (see bugs list - measured directly via a self-consistency test: 53.7% of real decision points show disagreement across 8 repeated queries on the identical state, avg modal agreement only 76.5%, implying a real ~0.3-0.6 nat irreducible cross-entropy floor in the labels themselves, not just "harder but learnable"). A Fable 5 consult (2026-07-10) caught that an apparent policy-loss "plateau" late in epoch 3 was a measurement artifact (cumulative epoch-average is structurally near-flat late in a large epoch) - the fix for this (windowed heartbeat figure) is in the codebase, staged for the next relaunch, not applied retroactively to this run. |

**Harness fairness control** (`az::tests::harness_has_no_seat_bias_between_equally_matched_players`,
kept as a permanent regression test): seat 0 as AdvCPU vs 3 other AdvCPU
wins **20.0%** (n=400), not the naive 25% - a real, modest structural
disadvantage (likely turn order), not a harness bug. Recalibrates how to
read any `eval_win_rate` seat-0 number: compare against ~20%, not 25%.

```
novacana-ml7 bc --out models/astra7_bc_scaled.bin \
  --chunk-games 8000 --epochs 15 --batch-size 256 --val-games 200 --players 4 \
  --winrate-every 1 --winrate-games 150
```

**New diagnostic commands built this session**: `novacana-ml7 winrate`
(standalone CLI wrapper around the pre-existing `az::eval_win_rate`, the
real ASTRA7.md §4.1/§2 milestone-eval function that only the AZ code used
internally before), `novacana-ml7 token-scan [--advcpu]` (the collision
scanner described in bugs #5 above).

**Live dashboard**: `http://localhost:8420` on hermes (local stdlib Python
server, polls the `astra7` SSH alias every 6s, not a static snapshot -
script at `tools/dashboard_server.py` in this repo, relocated off `/tmp`
since that's tmpfs and was losing the file across restarts; must be running
for the page to load). GPU name and hourly rate are now pulled live rather
than hardcoded, so it survives future pod migrations without code changes.

### TD-value stage - first real run (2026-07-10)

Launched against `astra7_bc_scaled_best.bin` (the epoch-6 BC checkpoint):

```
novacana-ml7 td-value --model models/astra7_bc_scaled_best.bin \
  --out models/astra7_td.bin --chunk-games 1500 --passes 3 \
  --temperature 0.8 --lr 0.001 --batch-size 64 --val-games 200
```

Pass 1/3 self-play generation: 1500 games → 633,529 samples in 1410.1s
(~1.06 games/s - net-driven self-play does a real GPU forward pass per
decision, so it's expected to be far slower than AdvCPU/ExpertCpu's
CPU-heuristic self-play; not a red flag). Training sweep over those samples
was fast (9899 batches in ~95s) and `value_mse` was already flat at
~0.009 from the very first heartbeat (batch 100) through the last.

Checked this wasn't a bug before concluding it's fine: traced
`net::td_train_step` (net.rs) - the trunk runs on `net.valid()`'s inner
backend and is re-imported via `Tensor::from_inner` (not `detach()`, which
would leak - see [[burn-autodiff-leak]] pattern), so gradients genuinely
only reach `value_hidden`/`value_out`, confirmed by the doc comment and the
code structure (`GradientsParams::from_grads` has no trunk entries because
the trunk forward pass never touched the live autodiff `net`). A fast, flat
convergence is the expected shape here: the value head already got a BC-
stage warm start against terminal outcomes (`bc_train_step`'s
`VALUE_LOSS_WEIGHT=0.3` term), and TD(λ=0.9) targets are smoother than raw
0/1 outcomes since most of them blend in a bootstrapped `V(s_next)` rather
than the raw terminal result. **The training-loss curve is not the metric
that matters here** - the real check is the post-training value-rank gate
(`novacana-ml7 value-rank`, §4.2, ≥65% required), which hasn't run yet
since pass 1 isn't complete. Treat this section as provisional until that
gate result is in.

No crashes, no NaNs, GPU/CPU healthy throughout (29-30% GPU util, 342-366%
CPU, ~950MiB VRAM). Will update this section as passes 2-3 complete and
once the value-rank gate runs.

**Pass 1/3 complete**: `train_value_mse=0.0092  val_value_mse=0.0095
(1697.5s)  *best`. Val tracks train closely (0.0095 vs 0.0092) - no
value-head overfitting signature. Checkpoint saved to
`astra7_td_best.bin`. Pass 1 total wall time ~28.3min (mostly the 1500-game
self-play generation, ~23.5min of that). Pass 2/3 self-play generation
started immediately after, no gap, no crash. At this rate all 3 passes
should finish in roughly 80-90min total from launch (~01:20 UTC), i.e.
around 02:45-03:00 UTC.

**Pass 2/3 complete**: `train_value_mse=0.0094  val_value_mse=0.0091
(1665.7s)  *best` - new best (down from pass 1's 0.0095), consistent
timing (~27.8min). Pass 3/3 self-play generation started immediately, no
crash. Full run should finish around 02:50-02:55 UTC.

**Pass 3/3 complete, run finished cleanly**: `train_value_mse=0.0091
val_value_mse=0.0091 (1714.7s)  *best`. Process exited (code 0), both
`astra7_td.bin` and `astra7_td_best.bin` written (19,952,515 bytes, Jul 10
02:45). Final val_value_mse trajectory across passes: 0.0095 → 0.0091 →
0.0091 - essentially flat the entire run, never moved much even with fresh
self-play data regenerated each pass. (Minor cosmetic bug, not functional:
the final console line prints `models/astra7_td.bin.bin` - double `.bin`
- because the print template appends `.bin` to an `--out` value that
already had it; the actual saved files are correctly named, single
extension, confirmed via `ls`.)

**Gate check - FAILED on first attempt**: `novacana-ml7 value-rank
--model models/astra7_td_best.bin --games 400 --points 1000 --players 4`
→ **50.8%** top-2 hit rate (n=1000) vs the required ≥65% (ASTRA7.md §4.2).
Not noise - roughly 9 SE below gate at this sample size. Did not proceed
to AZ (per standing instruction: never skip this gate).

**Root-cause investigation (Fable 5 consult, with independently-run
measurements, not just code-reading)**: Fable 5 built a temporary local
test harness (reverted after) and measured:

| Configuration | value-rank top-2 |
|---|---|
| Pure-chance floor (E[2/k] over legal-set sizes, mode 5-6) | 38.8% |
| Untrained random net, full protocol | 44.3% (±2.9 SE) |
| Pre-TD BC checkpoint (before any TD-value training) | 38.7% - **exactly chance** |
| This run's TD checkpoint | 50.8% |

So TD training did add real signal (+12pp over a chance-level warm start),
but nowhere near enough. Cause: `compute_td_targets`'s λ=0.9 recursion
weights the terminal outcome at step *t* by `0.9^(steps-from-end)` - over
this game's long trajectories (measured: 633,529 samples / 1500 games / 4
seats ≈ **105 decisions per seat-trajectory**), that's ~0.4% by the
midpoint. For roughly two-thirds of every trajectory, the TD target is
>99% a smoothed function of the net's own current value prediction -
near-zero true gradient by construction, which is exactly why
`val_value_mse` looked flat from the first heartbeat (see above) - that
flatness was never a sign of healthy convergence, it was a sign the
targets barely differed from what the net already predicted. Compounding
factor: `td_train_step` only updates `value_hidden`/`value_out`, so the
self-play policy is identical across all 3 passes - passes only buy
iterated bootstrapping, propagating real terminal signal backward a few
turns each time. 3 passes ≈ signal reaching roughly the last third of the
game, consistent with the 50.8% result (late-game decision points, which
also get partial ground-truth rollout in the gate's own scoring
mechanism, likely rank fine; mid/early-game points sit near chance).
Fable 5's explicit verdict: **not a bug, not a "gate too hard at this
stage" situation - an underpowered training signal, quantifiable and
fixable.** No evidence of an architectural fault in the frozen-trunk
mechanism itself (it demonstrably added signal); that's the *next*
suspect only if the fix below doesn't clear the gate.

**Fix applied**: `LAMBDA` in `ml7/src/td_value.rs` changed from `0.9` to
**`1.0`** - collapses the λ-return recursion to `G_t = R` for every step
(pure Monte Carlo regression against the actual terminal outcome, no
bootstrapping at all). Higher per-sample variance, zero bias, and doesn't
depend on the value head's own still-forming estimates - the right
tradeoff given this stage generates 600K+ samples per chunk into a
tiny 2-layer head with the trunk frozen (ample data to absorb the extra
variance). Doc comment in `td_value.rs` updated with the full reasoning
and measurements. Existing test `td_targets_collapse_to_outcome_at_final_step`
updated: at λ=1.0 every step's target now collapses to the outcome exactly
(previously asserted intermediate steps were strictly between 0 and 1,
which was λ=0.9-specific behavior).

**Relaunched** with fewer/larger-chunk passes since at λ=1.0 additional
passes over the same self-play distribution add nothing but sample count
(no re-bootstrapping to iterate on):

```
novacana-ml7 td-value --model models/astra7_bc_scaled_best.bin \
  --out models/astra7_td.bin --chunk-games 2500 --passes 1 \
  --temperature 0.8 --lr 0.001 --batch-size 64 --val-games 200
```

**Decision rule for the re-run's value-rank result** (Fable 5's explicit
guidance, apply automatically, no need to re-consult unless the result is
ambiguous or surprising):
- **≥65%**: gate passed, proceed to the fidelity check + token-scan, then
  AZ league (budget/readiness consult first, see "Next steps").
- **~55-62%**: the frozen BC trunk's pooled features are now the binding
  constraint, not the target formulation. Options: one low-lr (~1e-4) pass
  with the trunk unfrozen (caution: a value-only loss could distort the
  trunk under the policy head - re-check `fidelity` and a quick win-rate
  read afterward if this is tried), or bring this specific number back to
  Fable 5/Opus for a judgment call rather than guessing.
- **Still near 50%, barely moved**: this would point to (c) - dig into the
  trunk representation and `mcts.rs::value_at_state`'s off-distribution
  issue (training states are always "my_idx is deciding," but gate/leaf
  successor states usually aren't) - this is a real architectural
  possibility, not yet ruled out, just deprioritized until the cheap
  λ=1.0 experiment is ruled out first.

Also flagged as worthwhile but not done yet: a turn-number-stratified
value-rank readout (bucket decision points into early/mid/late terciles -
`Captured` already carries the full `GameState`) would directly confirm or
refute the "signal only reached the last third of the game" story on the
existing (λ=0.9) checkpoint, and separately, replacing the tautological
TD-target MSE with a value-vs-actual-terminal-outcome metric (MSE or AUC
on held-out self-play) would give a training-time signal that actually
means something for future runs. Neither implemented yet - worth ~20min if
the λ=1.0 result comes back ambiguous and more diagnosis is needed.

**λ=1.0 re-run result - WORSE, not better: 44.8%** (n=1000), below even
the failed λ=0.9 attempt (50.8%) and statistically indistinguishable from
an untrained random net (44.3%±2.9). `train_value_mse=val_value_mse=0.1828`
- suspiciously close to `p(1-p)` for `p≈0.21-0.25` (the 4-player base win
rate), consistent with the head converging toward predicting roughly the
marginal win probability rather than anything state-dependent. Did not
match any of the three anticipated decision-rule buckets above, so went
back to Fable 5 rather than mechanically picking one.

**Fable 5's revised diagnosis (2026-07-10, second consult)** - this one
explicitly walked back part of the first consult's advice rather than
rationalizing the miss: the λ=0.9→1.0 switch wasn't a clean "same
experiment, better target formula" test as advertised - dropping from 3
passes to 1 pass removed the actual mechanism that produced 50.8%. Each
λ=0.9 pass re-records `V(s_next)` live from the just-updated head, so 3
passes function as 3 rounds of value iteration (pass 1 fixes up late-game
values where the terminal signal survives 0.9^k decay, pass 2's recorded
bootstraps carry that signal into mid-game targets, etc.) - that iterated
propagation, not the λ formula itself, is what got 50.8%. Switching to
pure-MC λ=1.0 *and* cutting to one pass meant ~1.08M samples seen once
(16,920 batches) against high-variance binary targets, vs ~1.9M samples
across three iterated rounds previously - an underpowered, apples-to-
oranges comparison, not a fair test of λ=1.0 on its own terms.

That said, the diagnosis isn't "just run λ=1.0 longer and it'll be fine":
Fable 5's read is the binding constraint is now the **frozen trunk's
representation**, not the target formula. Supporting evidence: (a) the
pre-TD BC checkpoint already sat at the chance floor (38.7%) - BC's value
loss (weight 0.3, trunk unfrozen at that stage) had already collapsed the
head toward a near-constant output, since the trunk there was shaped
almost entirely by the policy loss; (b) λ=0.9's +6.5-point lift over the
untrained baseline (50.8 vs 44.3, ~4 SE, a real effect) shows the frozen
features carry *some* value signal, but likely cap out well below the 65%
gate even with more iterated passes.

**Next step (not yet started)**: unfreeze the trunk for TD-value training,
per Fable 5's specifics:
- New train step needed (`td_train_step`'s whole design deliberately
  freezes the trunk via `net.valid()`/`Tensor::from_inner` to dodge the
  autodiff-leak bug - [[burn-autodiff-leak]] - so this requires an
  unfrozen variant shaped more like `bc_train_step`/`az_train_step`, full
  autodiff graph, more VRAM).
- lr ~1e-4 (10x below the head-only run's 1e-3).
- **Must protect the policy head**: pure value-MSE fine-tuning through a
  shared trunk will distort the pointer-policy features BC paid for.
  Fable 5's suggested fix: joint loss = value MSE + a policy-distillation
  anchor (CE against a frozen pre-update snapshot's masked softmax on the
  same states, weight ~0.5-1.0) - reuses the already-recorded full token
  sequences, no new data needed.
- Keep λ=1.0 (fine once the trunk can create dedicated value features
  instead of the head fishing in policy-shaped ones) with 2-3 fresh passes
  for sample volume (not bootstrapping this time).
- **Also add AdvCPU seat mixing to TD self-play generation**: currently
  `generate_td_samples` self-plays 100% net-vs-net at T=0.8, but
  `run_value_rank`'s gate scores exclusively on AdvCPU-vs-AdvCPU game
  states - the value head has never trained on the distribution it's
  gated against. Same AdvCPU-mixing precedent as the BC stage (commit
  69bd20a). Flagged as possibly worth several points on its own.
- Caveat worth remembering: the gate's own ceiling is unknown (it measures
  agreement with AdvCPU, not an oracle, via one-step off-policy successor
  values) - if a properly unfrozen run plateaus around 60-62% with
  genuinely state-dependent value curves, the next question becomes
  whether the gate itself needs interrogating, not just the model.

Sequencing: the batched-inference fix (see "Self-play/search batched
inference" section below) and this trunk-unfreezing work are independent
- finish and verify batching first (it speeds up TD data regeneration too,
useful for whatever comes next), then do the unfreeze work. **Not
launched yet - paused here to check in with the user, since this is a
bigger commitment (new training code path, more GPU-hours) than the
hyperparameter tweaks tried so far.**

### Self-play/search batched inference (2026-07-10)

Prompted by the user noticing (correctly) that the pod's `nproc`=64 is the
*host's* core count, not the cgroup allocation (~13-14 usable vCPUs) - the
real finding stands regardless: during the TD-value run, only ~3.6 of the
usable cores and ~30% of the GPU were ever active at once. Checked
per-thread state directly (`ps -T -p <pid>`): of 48 OS threads, most sat
on `futex_wait_queue`, with dedicated `cuda-EvtHandlr`/CUDA-context
threads visible - consistent with CUDA-driver-level serialization, not
CPU contention. Root cause: every self-play/search/eval path called
`net::forward_single` (batch-of-1) directly from its own rayon worker
thread, so concurrent threads on the same CUDA context serialized one tiny
GPU round-trip at a time instead of batching.

**Why this mattered enough to fix before AZ specifically**: TD-value's
self-play makes exactly ONE forward call per real decision. AZ's MCTS
(`mcts.rs`) spends `sims>=400` PUCT visits per real decision, and per its
own module doc, most visits fall through to a value-head forward call
(`leaf_value_choose_action`'s `value_at_state` fallback) - roughly 2-3
orders of magnitude more GPU calls per decision than TD-value. The same
serialization bug would have been far more costly there, in both wall
time and rented-GPU dollars.

**Fix**: new `ml7/src/batch_infer.rs` - `BatchedNet<B>`, a shared
micro-batching wrapper. Callers submit a single-decision request (via a
`forward_single`-compatible method) through a `Mutex<VecDeque>` + condvar
queue and block on a per-call oneshot channel; a dedicated background
thread drains the queue (waiting a small 300µs grace period to let
concurrently-running threads' requests land in the same batch, capped at
`max_batch=64`), runs ONE `net::forward` call over the batch, and routes
results back. `Drop` stops and joins the batcher thread cleanly, so
callers just do `BatchedNet::spawn(...)` at the top of a generation call
and let it fall out of scope at the end.

**Threaded through** (all the found `net::forward_single`/`net::forward`
per-decision call sites, confirmed via a full grep sweep of `ml7/src/`):
- `mcts.rs`: `value_at_state`, `root_priors`, `leaf_value_choose_action`,
  `leaf_value_chain_response`, and the three public entry points
  (`search_choose_action`, `search_chain_response`,
  `successor_value_choose_action`) all take `&BatchedNet<B>` now instead
  of a `net: &Net<B>, device: &B::Device` pair.
- `az.rs`: `AzMctsPlayer`/`GreedyNetPlayer`/`CollisionScanNetPlayer` hold
  an `Arc<BatchedNet<B>>` field instead of separate net+device fields;
  `generate_az_games`, `eval_win_rate`, `scan_live_play_for_collisions`
  each spawn one `BatchedNet` internally and share it across their
  rayon-parallel game closures.
- `td_value.rs`: `TdSampler` and `generate_td_samples` - same pattern.
- `evals.rs`: `run_value_rank` - same pattern (it also parallelizes
  ~13-14 decision points at once, each scoring several candidate actions
  sequentially, same batch-of-1 problem at smaller scale).
- `net::forward_single` itself (the old free function) was deleted -
  fully superseded, would otherwise be dead code.

**Deliberately unchanged**: every public entry point kept its existing
`net: &Net<B>, device: &B::Device` signature - only the *internal*
implementation spawns a `BatchedNet` now. This meant **zero changes to
`main.rs`** (every CLI command call site - `cmd_td_value`, `cmd_az`,
`cmd_value_rank`, the `winrate`/`fidelity`/`token-scan` commands - was
unaffected) and zero changes to most existing tests; only the handful of
tests that called the internal MCTS search functions directly
(`mcts.rs`'s own test module) needed updating to spawn a `BatchedNet`
first.

**Found and fixed a real deadlock during local verification** (not
theoretical - reproduced reliably): running the full local test suite (5
BatchedNet-using tests concurrently) hung indefinitely - confirmed via
`/proc/<pid>/task/*/wchan` showing every one of ~29 threads parked on
`futex_do_wait` with zero CPU time advancing over a 15s+ sampling window.
An isolated single low-concurrency test did NOT hang (misleading at
first - cost real time chasing "is this just slow" before checking
harder). Root cause: `ndarray` (which `burn-ndarray` wraps) depends on
`rayon` and uses its **global** thread pool for its own internal tensor
ops. The self-play outer game loop was ALSO rayon (`.into_par_iter()`) -
so under enough concurrent games, rayon worker threads block on
`BatchedNet::forward_single`'s channel `recv()` while still holding their
pool slot, and the batcher thread's own `net::forward()` call then can't
get a free rayon worker for `ndarray`'s internal computation either.
Classic nested-thread-pool exhaustion deadlock, invisible at low
concurrency (not enough demand to saturate the pool) and guaranteed once
concurrency crosses some threshold - which the REAL pod self-play
generation (~13-14 concurrent games, or many more for AZ's MCTS) would
have hit hard.

**Fix**: added `batch_infer::parallel_for_each` - a small bounded pool of
plain `std::thread`s (via `std::thread::scope` + a shared atomic work
counter) pulling from the game index range, `DEFAULT_CONCURRENCY=32`
workers by default. Completely independent of rayon's global pool, so no
nesting hazard regardless of how saturated either pool gets. Replaced
`.into_par_iter()` with this at every site that also uses `BatchedNet`:
`td_value.rs::generate_td_samples`, `az.rs::generate_az_games`,
`az.rs::eval_win_rate`, `az.rs::scan_live_play_for_collisions`,
`evals.rs::run_value_rank`. Left rayon alone at sites that never touch
`BatchedNet` (`evals.rs::collect_decision_points`,
`az.rs::scan_training_data_for_collisions` - pure AdvCPU/ExpertCpu games,
no net, no nesting risk).

**Verification status**: compiles clean locally (`cargo check --tests`,
zero warnings). Full local test suite (all 24 tests, ndarray/CPU backend,
release build) passes: 21 in 117.25s + the 3 slower harness-bias controls
in 72.33s - the two tests that used to hang forever
(`self_play_produces_samples_with_valid_distributions`,
`real_games_produce_finite_td_targets`) now complete normally (a bit over
60s each, legitimate MCTS/full-game cost, not stuck). **Not yet
synced/rebuilt on the pod, not yet verified under the `tch`/CUDA backend,
not yet measured for actual wall-clock speedup on a real self-play run.**
Given the deadlock only manifested at higher concurrency, the pod run
(much higher real concurrency than these tests) is the real test of
whether `parallel_for_each` + `BatchedNet` holds up - don't skip that
verification step before trusting this for AZ.

## Budget

$25 USD loaded on the RunPod account. Revised estimate for this run on a
4090: ~$5-7 (down from ~$10.30 on the A4000). Opus 4.8 flagged that the
$15-25 total-campaign estimate (ASTRA7.md §10) inherited the same "compute
is cheap" premise the BC telemetry just broke - the AZ league stage in
particular (100-200 iterations × 150-200 games × ≥400 MCTS sims each) is
GPU-inference-bound self-play, not free, and is where AZ campaigns
typically spend most of their money. **Treat the $15-25 figure as
optimistic; budget for a top-up before starting the AZ league rather than
mid-run.**

## Operating notes (from the 2026-07-09 review, apply going forward)

- **Don't over-grind BC.** Its ceiling is roughly the teacher's (AdvCPU)
  level - the 55-65% target is reached in TD-value + AZ, not BC. Watch
  per-epoch val_acc; stop once it flattens or clears the §4.1 gate rather
  than forcing all 15 epochs. The old Atlas run plateaued by epoch 3-4
  even data-starved.
- **200 val-games is fine for the in-run token-accuracy metric** - it's
  not a §2 win-rate violation, that gate only applies to the §4.1
  milestone's 600/1000-game win-rate eval. Don't let a cheap in-run number
  substitute for that later.
- **Watch s/batch trend over the full run, not just a temperature
  snapshot.** 93°C at 8 minutes in isn't a throttle signal by itself, but
  thermal drift on a shared pod over many hours could quietly inflate both
  time and cost - the tell would show up as rising s/batch, not just a
  temperature number.
- **rsync-not-git is a single point of failure.** The repo's substantial
  local changes exist only on hermes's working tree + an ephemeral rented
  pod. Worth a throwaway WIP-branch commit as a safety net (not a "commit
  to main" §8.6-gated action) - ask before doing this, per this project's
  commit policy.
- **The heartbeat's cumulative epoch-average can look like a false
  plateau late in a large epoch - read epoch-end values or the windowed
  figure, not a late-epoch cumulative snapshot.** Found 2026-07-10: policy
  loss appeared frozen (~0.765, barely moving) ~4000+ batches into a
  ~20000-batch epoch. A Fable 5 consult caught it - at that point in the
  epoch, each new batch can only move a cumulative running average by
  ~1/n of its deviation, so it's *structurally* near-flat late in a large
  epoch regardless of what the model is actually doing batch-to-batch.
  The actual instantaneous behavior (checked via the epoch-end value one
  epoch later) was still improving. Fixed: heartbeat now also prints a
  windowed (last-`HEARTBEAT_EVERY`-batches) figure alongside the
  cumulative one - `policy=X (lastN=Y)` - and the dashboard charts the
  windowed figure by default. This fix is in the codebase and synced to
  the pod's source, but **not applied to the currently-running BC job**
  (purely an observability change, no reason to interrupt training for
  it) - will take effect on the next natural relaunch.
- **BC's teacher-vs-3xAdvCPU number is the load-bearing calibration
  reference, not the teacher's own self-play baseline - keep both handy,
  they answer different questions.** ExpertCpu vs 3 other ExpertCpu
  (self-play) measures seat-position fairness (~19.65% vs AdvCPU's
  ~20.0%, i.e. no skill-dependent effect). ExpertCpu vs 3x AdvCPU
  (**47.5%±0.9**, established earlier in the project) is what tells you
  whether the §4.1 gate (25-40%) is even reachable by a BC clone of this
  teacher - if the teacher itself can't clear it, no amount of BC epochs
  will. 47.5% clears the gate with real room, so BC has a legitimate
  shot - this was nearly re-litigated as an open question in a Fable 5
  consult simply because the number wasn't restated in that prompt, not
  because it was actually unknown. Worth keeping visibly documented
  (here) so it doesn't get "lost" again.

## Next steps

**RESOLVED (2026-07-10, overnight autonomous run): BC stopped at epoch 6,
TD-value launched.** The decision rule below (Fable 5 consult) played out
exactly as specified - win_rate never broke past 24% through epoch 6
(final trajectory: 14.20/21.70/19.20/23.10/23.00/22.90%, all n=1000), a
second Fable 5 consult confirmed the stop decisively given the complete
6-epoch picture, BC was stopped, the epoch-6 checkpoint was safety-copied
to `models/astra7_bc_epoch6_final.bin` before killing the process, and
TD-value was launched:

```
novacana-ml7 td-value --model models/astra7_bc_scaled_best.bin \
  --out models/astra7_td.bin --chunk-games 1500 --passes 3 \
  --temperature 0.8 --lr 0.001 --batch-size 64 --val-games 200
```

Log: `~/td_value.log` on the pod. **This is the first time this stage has
ever run against real data** - it was only unit/CPU-smoke-tested by the
agent that built it, never run for real. Early signs at launch were
healthy (self-play generation started correctly, no crash, GPU lightly
used as expected for single-sample forward passes during self-play), but
this is genuinely new territory and worth extra scrutiny on review - check
`~/td_value.log` in full, not just the tail, and sanity-check the first
pass's output before trusting it.

<details>
<summary>Original decision rule (for reference/audit trail - already executed above)</summary>

win_rate climbed 14.20%→21.70% (epochs 1-2), then read 19.20% at epoch 3
(n=1000). Confirmed NOT statistically significant (z≈1.4) - val_acc rose
monotonically, so not overfitting either. Fable 5's read: win_rate likely
pinned near the ~20% turn-order baseline while val_acc keeps climbing on
decisions that don't move win probability much - consistent with the
measured ~0.3-0.6 nat label-noise floor. Tripwire: >24% clearly at/before
epoch 6 → continue to 15 epochs; still 18-23% at epoch 6 → stop, move to
TD-value. The latter is what happened.
</details>

**Once TD-value concludes:**
1. Run `novacana-ml7 value-rank --model models/astra7_td_best.bin --games
   400 --points 1000 --players 4` - the §4.2 gate (≥65% required before
   trusting AZ's leaf evals). Re-run regardless of a stale earlier reading
   (the `hands_dealt` engine bug window may have contaminated it, and this
   is the first real post-fix measurement).
2. Also worth running `novacana-ml7 fidelity` on the TD-value checkpoint
   for a sanity check, and a fresh `token-scan` given the encoding is
   unchanged from BC but hasn't been re-verified against TD-value's
   self-play distribution specifically.
3. Move to the AZ league on the **same** rented instance - no hermes+Atlas
   split, no `server.rs`/`client.rs`. Code is DONE and tested (ASTRA7.md
   §11) - this is "run `novacana-ml7 az` against the value-rank-gated
   TD-value checkpoint," not "build the stage." **Budget a top-up before
   this stage, not during it** - the AZ league is likely to be the
   dominant remaining cost, and $15-25 was the original full-campaign
   estimate (now partly spent on BC - check current RunPod balance).
