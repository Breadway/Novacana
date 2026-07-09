# ASTRA-7 — Implementation Plan

Target: an entity-based transformer trained BC → TD-value → AlphaZero league, on a
hermes (CPU game gen) + Atlas (GTX 1050 Ti inference/training) split.
Goal: **55–65% win rate vs 3× AdvancedCpuPlayer** (4-player, parity = 25%).

This document is self-contained: a fresh agent should be able to implement v7
from it without the v6 session history. Read **§8 (hard constraints)** before
writing any code — every item there was paid for in wall-clock hours.

---

## 1. Where v6 ended (verified baselines)

All numbers 1000-game evals, 4-player, 1 ML seat vs 3 AdvCPU, seat 0, parity 25%.

| Model | Full-ML | Hybrid (AdvCPU aux) | File |
|---|---|---|---|
| v6 BC only (8K games) | 16.2% | 21.8% | `ml/models/astra6_v3_wide.bin` |
| v6 PPO best | 35.8% | 41.4% | `ml/models/astra6_v4b_rl_best.bin` |
| **v6 final (gentle AZ)** | **38.7%** | **42.0%** | `ml/models/astra6_v5b_az_best.bin` |
| v3 historical record | n/a | 43.8% (500 games) | not loadable (old format) |

v6 converged at ~38.7%: three follow-up configurations failed to beat it
(recipe repeat, value-leaf sims, AlphaZero exploration — see §8.4/§8.5).
**A pure heuristic bot, `ExpertCpuPlayer` (engine/src/cpu.rs), SUBSEQUENTLY
BEAT it at 47.5%** — see §1.1. The measured bottlenecks motivating v7's design:

1. **Fixed feature vectors** (162-dim handcrafted ×3 stacked) forced a 31-way
   abstract action space, which forced a bolt-on candidate head for concrete
   disambiguation, which caused: label-noise ceilings, aux-path training
   blindness, and a self-imitation bug. Every v6 pathology traces to this seam.
2. **Value head cannot rank nearby states** (measured: 1-step value lookahead
   scored 21.0% vs candidate head 23.2%). It was trained only on terminal
   outcomes. This capped MCTS quality: deeper/cheaper search made things worse.

### 1.1 Chain-response timing — a bigger bottleneck than anything in v6's own list

`ExpertCpuPlayer` is AdvCPU with two additions: 1-ply simulation search for
`choose_action` (a fitted-eval-scored lookahead, ~+4 pts) and 1-ply simulation
search for `respond_to_chain` (whether/how to react to an open Chain — Cosmic
Denial negation, Shockwave/Black Hole timing — scored by actually resolving
the chain and evaluating the result, **+9.6 pts, the single largest lever
measured anywhere in this project**, v3→v4 in the eval table above). Both use
the same trick: clone (state[, chain]), force the candidate response, let the
rest resolve via AdvCPU, score the successor.

That a static-heuristic bot with a cheap simulated-resolution fix on ONE
decision type can beat a trained transformer-free neural net by 8.8 points is
a strong signal that **chain-response quality — not policy sophistication —
was the largest single leak in v6's win rate.** Concretely: v6's BC recorder
and RL/AZ self-play all route chain responses through the same abstract-action
+ candidate-head pipeline as every other decision, with no decision-type-
specific treatment and no evidence anyone verified it handles high-stakes
negation timing correctly.

**Action items for v7, not optional:**
- In the BC data (§4.1), explicitly measure fidelity SEPARATELY for chain
  responses vs. choose_action — if it's much worse, the pointer-policy
  tokenization (§3.1) needs chain-specific action-token fields verified
  against real negation scenarios before trusting the rest of the pipeline.
- In the AZ league (§4.3), make sure MCTS actually re-searches at chain-
  response decision points with the same sim budget as choose_action — don't
  let an implementation shortcut skip search on "reactive" decisions because
  they seem lower-stakes. They are not; they're `resolve_chain`-adjacent and
  can decide the game outright.
- Consider adding `ExpertCpuPlayer` to the AZ opponent pool (§4.3) specifically
  BECAUSE its chain play is strong — training against a sharp reactive
  opponent should surface and punish v7's chain-response mistakes during
  self-play, not just at final eval time.
- If v7's BC+AZ pipeline, once complete, still loses to ExpertCpu, the
  fastest diagnostic is: run `ExpertCpuPlayer` vs `v7-full-ML` head to head
  and inspect chain-response decisions specifically before assuming the
  transformer architecture itself is the problem.

## 2. Eval protocol (non-negotiable)

- **Never trust a 300-game eval.** σ at 300 games ≈ 2.7% — it misled the v6
  campaign three separate times. In-run evals: **600 games minimum**. Any
  number you record, compare, or act on: **1000 games** on the checkpoint.
- Every training loop must **checkpoint best-eval weights to `*_best.bin`**
  separately from the rolling save. Peaks decline; the checkpoint is the
  deliverable.
- Intervention rule that worked: two consecutive out-of-band evals → stop,
  restart from `_best` with adjusted hyperparameters. One bad eval is noise.
- Report full-ML and hybrid configurations separately; compare only within a
  configuration.

## 3. Architecture: entity transformer with pointer policy

### 3.1 Tokens

Every token is `embed(type) + embed(fields...)` summed into `d_model`. One
shared vocabulary of learned embeddings; field embeddings below.

**State tokens** (set, no position encoding):
| Token | Count | Fields |
|---|---|---|
| `HandCard` | ≤ hand size (~7) | card_type (29), star_class (0–6) |
| `MySystem` | ≤ 9 | star_type (7), class (2–6), fill (bucketed 0–6), eclipse_prot, complete |
| `OppSystem` | ≤ 3 opp × 9 | as MySystem + opponent slot (1–3) |
| `Opponent` | 3 | slot, hand_size (bucketed), complete_count, approaching_win |
| `Global` | 1 | deck_size bucket, turn bucket, DMV class, active_player == me |

**History tokens** (sequence, learned positional encoding, most-recent-last,
truncate to last **64 events**):
| Field | Values |
|---|---|
| event type | play / chain-response / system-complete / destroy / discard-visible / omen-attach |
| card_type | 29 + unknown |
| actor | me / opp1 / opp2 / opp3 (relative seat — v6 bug: absolute seats are useless) |

Maintain the event log in the engine's existing `play_history` (extend from 20
→ 64 entries and add event type + relative-seat resolution at encode time; the
GameView already exposes `my_idx`).

**Action tokens** (one per legal concrete action — this replaces the abstract
action space AND the candidate head AND the aux-path encoders):
| Action | Fields |
|---|---|
| Pass / Protostar choice / etc. | action_kind (≈12 kinds) |
| PlayStar / PlayDiscovery / PlaySolarSystem | action_kind + card_type + target system's (star_type, class, fill) |
| PlayFlare / PlayFracture / PlayOmen | action_kind + card_type |
| chain response | action_kind + card_type |
| decide option | action_kind + prompt_type (12, reuse v6 keyword taxonomy) + option keyword flags |
| pick_target option | action_kind + target entity fields (reuse §3.1 system/opponent fields) |
| discard candidate | action_kind + card_type |

Total sequence: ~40 state + 64 history + ≤ ~40 action tokens ≈ **≤ 150 tokens**.

### 3.2 Network

- Encoder: **6 layers, d_model 256, 8 heads, FFN 1024, pre-norm**, ~6–8M params.
- **Policy**: pointer attention — a learned query over encoder outputs at the
  action-token positions → logits over legal actions. Masking is structural
  (only legal actions have tokens). Softmax over however many actions exist.
  ONE head covers choose_action, chain responses, decides, targets, discards —
  the decision *kind* is in the action tokens.
- **Value**: mean-pool state tokens → MLP(256→256→1), sigmoid. Trained on
  TD(λ) targets (§4.2), NOT terminal-only.
- **Auxiliary head** (cheap, recommended): predict opponent hand-size-weighted
  card-type distribution (29-way, KL loss, weight 0.1). Known to sharpen
  imperfect-information value estimates.
- Multi-pick decisions (pick_targets count>1, multi-discard): autoregressive —
  re-encode with chosen tokens marked, or greedy top-k by pointer logits for
  v7.0 (v6 used greedy top-k; acceptable).

### 3.3 Framework

**burn 0.21 with the `tch` (LibTorch/CUDA) backend on Atlas.** Do NOT use
burn-wgpu on the 1050 Ti (no benefit) and do NOT use burn-ndarray for
transformer training (too slow). Keep an ndarray CPU inference path ONLY if
the inference-server design (§5) proves inadequate — it shouldn't.

## 4. Training pipeline

### 4.1 Stage BC (gate: ≥ 97% concrete-action agreement on held-out)

- Data: AdvCPU self-play recorder emitting (token sequence, chosen action
  index) per decision — ALL five decision kinds through the one pointer head.
  Gen throughput on hermes ≈ 1,400 games/s → stream fresh chunks per epoch
  (`--chunk-games` pattern), never pregenerate (v6: 16K games ≈ 26GB RAM).
- 20K games/epoch equivalent, ~15 epochs, AdamW lr 3e-4, batch 512 sequences,
  label smoothing 0.05 over the legal set, val-checkpoint on held-out argmax
  accuracy (v6 pipeline pattern — port it).
- Expected: fidelity ceiling is ~99% now (no tie-break noise: the pointer
  target is AdvCPU's actual concrete choice).
- **Milestone eval**: full-ML 1000 games. v6 BC gave 16%; v7 BC should land
  30–40% because aux decisions are now first-class. If < 25%, stop and debug
  the tokenizer before any RL.

### 4.2 Stage TD-value (gate: value ranks 1-step successors correctly ≥ 65%)

- Generate self-play games with the BC policy (sampled, T≈0.8). For each
  state, target = TD(λ=0.9) blend of successor value estimates and final
  outcome. 2–3 passes, value+aux heads only (trunk frozen — see §8.2 for HOW
  to freeze in burn).
- New diagnostic to implement: `value-rank` — sample 1000 decision points,
  apply each legal action (engine `from_state`, the v6 `value_lookahead`
  machinery in `ml/src/mcts.rs`), check how often the value head ranks
  AdvCPU's choice in its top-2. v6's head was near chance; the ≥65% gate
  guards the entire MCTS stage.

### 4.3 Stage AZ league

- MCTS over **concrete actions** (the action-token set), PUCT c=1.5, root
  Dirichlet ε=0.25 with **α = 10/n_legal_actions**, visit-temperature sampling
  first 15 decisions. These exploration settings crashed v6 at 50–200 sims —
  they are only re-enabled here because sims ≥ 400 (batched GPU makes this
  affordable) wash the noise out of the visit targets. If eval crashes >5
  points in the first 5 iters with pl converging: the wash-out assumption is
  wrong at your sim count — halve ε and double sims before abandoning.
- Sims: leaf = engine playout of the current turn + value head (NOT 20-turn
  rollouts — with a TD-trained value head passing the §4.2 gate, value leaves
  are correct; v6's failure was the untrained head, not the technique).
- Per iter: 150–200 games × 4 seats, az-epochs 1 (v6: 4 epochs over the window
  oscillated), lr 5e-5 → 2e-5 cosine, window 300K samples, snapshot pool every
  20 iters, AdvCPU share 0.3. Add `ExpertCpuPlayer` (engine/src/cpu.rs,
  47.5%±0.9 vs 3 AdvCPU — AdvCPU + 1-ply full-round sim search + fitted eval + bounded follow-up search + chain-response search (this now BEATS Astra-6's 38.7%)) to the
  opponent mix for diversity.
- Continue while verified evals improve; expect 100–200 iters. Restart-from-
  best with fresh optimizer when two consecutive 600-game evals decline
  (the v6 "leg" pattern — it works).

## 5. Infrastructure: hermes + Atlas split

```
hermes (Zen5 16T)                    Atlas (i7-7700HQ 8T, GTX 1050 Ti 4GB, 16GB RAM)
┌──────────────────────┐   LAN      ┌─────────────────────────┐
│ game engines (rayon) │──batches──▶│ inference server (tch)  │
│ MCTS tree logic      │◀──logits───│  batch ≤512, fp32       │
│ sample collection    │──samples──▶│ trainer (same process)  │
│ eval harness         │◀──weights──│  swaps net atomically   │
└──────────────────────┘            └─────────────────────────┘
```

- Protocol: length-prefixed bincode over TCP (tokio or std). Request =
  token-id tensors for a batch of positions; response = per-action logits +
  value. Target: <5ms per batch-512 round trip on LAN.
- hermes-side client: a bounded queue; game threads block on their decision
  future. With 16 game threads and MCTS batching (collect all leaf evals of a
  PUCT batch into one request), utilization is fine.
- Weight sync: trainer publishes to the inference net every N gradient steps
  (same process, atomic swap). hermes never holds weights in v7.
- **Atlas bring-up: DONE (2026-07-08).** Ubuntu 26.04 LTS, NVIDIA driver 580
  (Pascal, GTX 1050 Ti, compute cap 6.1), CUDA toolkit 12.4, Rust stable
  (rustup), LibTorch **2.9.0+cu126** (exact version torch-sys 0.22.0 — the
  version burn-tch 0.21.0 actually pulls in — requires; check
  `torch-sys-X.Y.Z/build.rs`'s `TORCH_VERSION` const before downloading ANY
  other version, don't assume). `LIBTORCH`/`LD_LIBRARY_PATH`/
  `LIBTORCH_BYPASS_VERSION_CHECK=1` set in `~/.zshrc`. Reachable at
  `192.168.1.112` (LAN) and `100.119.124.28` (Tailscale, registered as
  `atlas-1` — a stale offline `atlas` node from the old decommissioned
  install squats the real name; delete it from the Tailscale admin console,
  then `tailscale set --hostname=atlas` to reclaim it, whenever convenient).
- **CRITICAL tch-rs/LibTorch linking gotcha — GPU silently downgrades to CPU
  with ZERO error if you miss this.** On Linux, `torch-sys`'s build.rs detects
  `libtorch_cuda.so` and emits `cargo:rustc-link-lib=torch_cuda`, but the
  default linker `--as-needed` behavior DROPS that link because nothing in a
  typical crate references a CUDA symbol directly at link time (the crate's
  own build.rs comments acknowledge this as a known unresolved issue). Result:
  the binary builds fine, runs fine, `Cuda::is_available()` just returns
  `false` — no error, no panic, indistinguishable from a real CPU-only
  environment unless you go looking. **Every ml7 crate that links tch/burn-tch
  needs its OWN `build.rs`** emitting, in this exact order:
  ```rust
  fn main() {
      let lib = format!("{}/lib", std::env::var("LIBTORCH").unwrap());
      println!("cargo:rustc-link-search=native={lib}");
      println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
      println!("cargo:rustc-link-arg=-ltorch_cuda");
      println!("cargo:rustc-link-arg=-Wl,--as-needed");
  }
  ```
  Verify with `ldd target/release/<bin> | grep torch` — must show
  `libtorch_cuda.so`, `libc10_cuda.so`, `libcudart*.so`, `libcudnn.so`. If
  they're missing, training will silently run on CPU (slow, no crash) —
  this is EXACTLY the trap that made the first ml7 BC-training dispatch
  (CPU-only, correctly so at the time — Atlas wasn't ready) look like
  "burn-ndarray is just slow"; don't let it happen again once Atlas is live.
- Verified working end-to-end: a `tch::Tensor` round-trip on
  `Device::Cuda(0)` returns correct values. GPU is confirmed real and usable.
- **Benchmark still owed** (not yet done): dummy transformer (6×256, batch
  512, seq 150) fwd+bwd throughput on Atlas via burn-tch, to validate the
  ~1.5–2 TFLOPS effective assumption below before committing to the full
  training timeline.

## 6. Repository plan

New crate `ml7/` alongside `ml/` (do NOT rewrite `ml/` in place — v6 must stay
runnable as the benchmark and fallback opponent). **Status 2026-07-08: the
first four files exist and are tested; see the note after the tree.**

```
ml7/src/
  tokens.rs        # DONE — GameView/Action/history → token ids (§3.1), 9 unit tests pass
  net.rs           # DONE — burn transformer, pointer head, value head (aux head deferred)
  server.rs        # NOT STARTED — Atlas inference+train server
  client.rs        # NOT STARTED — hermes-side batched inference client
  recorder.rs      # DONE — AdvCPU BC recorder, integration-tested against real games
  selfplay.rs      # NOT STARTED — PPO-style + AZ game gen against the client
  mcts.rs          # NOT STARTED — concrete-action PUCT (port structure from ml/src/mcts.rs)
  evals.rs         # NOT STARTED — eval harness + value-rank diagnostic
  main.rs          # PARTIAL — `bc` subcommand only; tdvalue/az/eval/value-rank/serve not built
```

**Build status**: `cargo build --release -p novacana-ml7` and
`cargo test -p novacana-ml7` both pass (11/11 tests). **Training status
(2026-07-09): REAL NUMBERS EXIST, run in progress.** `ml7` is now wired to
`burn-tch` on Atlas (LibTorch 2.9.0+cu126, GPU-linking verified via `ldd` —
see §5). A 7-epoch BC run (300 games/epoch, batch-size 64 — batch-512 OOMs
on the 4GB card, batch-64 holds a flat 3531MiB with no leak) is running
detached on Atlas, ~35min/epoch:

| Epoch | policy loss | val_acc | val_ce |
|---|---|---|---|
| 1 | 0.5206 | 0.7332 | 0.4862 |
| 2 | 0.4613 | 0.7445 | 0.4579 |
| 3 | 0.4379 | 0.7460 | 0.4491 |
| 4 | 0.4528 | 0.7477 | 0.4572 |

Reads as real learning (val_acc climbing from a 66% tiny-scale baseline) with
visibly diminishing returns already by epoch 3-4, and epoch 4 broke the clean
monotonic trend (policy loss and val_ce both ticked UP while val_acc barely
moved) — first non-monotonic epoch, not yet concerning on its own (single
epoch, small magnitude), worth watching over epochs 5-7. Total games seen so
far: ~1,200 — far short of v6's tens-of-thousands-of-games BC stage, so the
flattening more likely reflects data volume than an architecture ceiling,
but this is exactly the open question the next phase should resolve (see the
"chain-response fidelity" and "data-scaling curve" action items in §1.1 and
the recommendation this section should get updated with once a fresh
independent review of the whole plan — GPU choice, architecture, training
logistics — has been done; see below if that review has landed).

Judgment calls made building the above (revisit if they don't hold up):
multi-pick decisions use greedy top-1 (not autoregressive); the auxiliary
opponent-hand-prediction head (§3.2) was deferred, not built; `play_history`
was used as-is rather than extending `record_play`'s signature for richer
event-type tags (the event-type token field is reserved/always-zero,
extensible later without a schema migration); weight persistence uses burn's
native `Module::save_file`/`load_file` rather than v6's hand-rolled flat-
`Vec<f32>` round trip (there's no second non-burn net to interop with in v7,
so the manual approach's only justification doesn't apply).

Engine changes made so far:
- `play_history`: 20 → 64 entries, DONE (main session, not the ml7 agent).
  Chain-response plays are now ALSO recorded (previously only primary
  `choose_action` plays were) — real blind-spot fix given how much
  chain-response quality turned out to matter (§1.1). Event-type tagging
  NOT done — deferred by design, see judgment calls above.
- Everything in §8.1 (hands_dealt fix, card-conservation invariant,
  decision-count watchdog, `_with_state` trait methods) — already exists,
  do not remove or reinvent.

## 7. Milestones, gates, timeline

| # | Milestone | Gate | Status |
|---|---|---|---|
| 0 | Atlas bring-up + benchmark | ≥1 TFLOPS effective fwd+bwd | **Bring-up DONE, GPU verified working. Throughput benchmark still owed.** |
| 1 | tokens.rs + recorder + tests | round-trip unit tests green | **DONE** (11/11 tests pass) |
| 2 | net.rs + BC training loop | BC gate §4.1 (≥97% agree, ≥25% eval) | **Code done, NOT YET TRAINED** — was CPU-bound, blocked on Atlas; retry on burn-tch next |
| 3 | inference server + client | <5ms/batch-512; self-play ≥30 games/s PPO-style | not started |
| 4 | TD-value stage | value-rank ≥65% | not started |
| 5 | AZ league | beats 38.7% verified within 30 iters | not started |
| 6 | Wrap: verified table, dashboard, memory | — | not started |

Original estimate: **~1.5–2 weeks elapsed**, mostly training wall-clock after
day 5 — still holds, clock effectively starts now that Atlas is live.
The v6 dashboard (`ml/dashboard.py`) parses `Iter`/`Eval @`/epoch log lines —
keep the same log formats and it works for v7 unchanged.

## 8. Hard constraints — violations cost days (all discovered in v6)

**8.1 Engine/self-play**
- **`GameEngine::run()`'s initial-deal guard had TWO successive bugs, both
  fixed 2026-07-08.** First fix used `turn_number == 0` to detect "fresh
  game" — but `GameState::new` sets `turn_number: 1` (so "Turn 1" displays
  correctly), making that guard's condition permanently false: hands were
  NEVER dealt via this path for the entire window between the two fixes,
  including plain `GameEngine::new()` games (not just `from_state` sims).
  Final fix: an explicit `hands_dealt: bool` field on `GameState`, set once
  in `deal_initial_hands()`, checked in `run()`, correctly carried through
  `Clone`/`from_state`. If you're ever tempted to infer "is this a fresh
  game" from `turn_number`/hand-emptiness again — don't; that's exactly what
  broke twice. Consequences of the bug window: (a) the "value head can't rank
  successors" measurement (§1 item 2) may be contaminated — re-run the
  `value-rank` diagnostic on the current (fixed) engine before trusting it;
  (b) any v6/ExpertCpu numbers from that window need re-verification, though
  in practice ExpertCpu's post-fix bench (49.3–49.9%) reproduced its pre-fix
  number (47.5%) closely, suggesting the live interactive bench sessions
  mostly weren't inside the broken window. Regression tests now guard this
  (`engine/src/game.rs` `#[cfg(test)]`: `fresh_game_deals_initial_hands`,
  `from_state_does_not_redeal_hands`).
- **Card conservation is now a tested invariant** (`assert_card_conservation`
  in the same test module) — building it caught a real, independent bug:
  `StarChart::remove_discovery` silently dropped a Discovery's attached
  Planet of Life omen instead of returning it to the deck (fixed, both call
  sites — Athena, Wormhole — now recover the freed omen). Run this invariant
  after any new simulation entry point you add; it would have caught both
  of the above immediately instead of via score-progression archaeology.
- **`from_state(...).run()` is NOT deterministic**, even with identical
  players and identical starting state — unseeded `rand::thread_rng()` in
  `Deck::shuffle`/`shuffle_in` (several Fracture/Protostar effects) and the
  Wormhole blind-steal. Pre-existing, not introduced by any of today's work.
  Don't write tests or diagnostics that assume determinism from `from_state`
  replay; test AdvCPU's decision function purity given identical inputs
  instead (see `adv_cpu_decision_is_pure_given_identical_state`).
- ML players can loop forever inside one game turn (net-card-positive plays +
  deck reshuffle). The engine's 50-action/turn cap (game.rs) is the guard —
  keep it. Stochastic: ~1 in 100 games without it, hours of single-core MCTS each.
- A general `max_decisions` budget (not just per-turn/per-game caps) now
  exists on `GameEngine` (`set_max_decisions`, default 3000 in the CPU search
  engines) — covers `simulate_finish_chain`'s path too, which previously had
  NO cap at all since it never enters `run()`'s main turn loop.
- Cap AZ outer games at 400 turns (`run_one_az_game`), not 1000.
- `GameState::clone` dominates per-sim cost — batch/reuse where possible.
- `Player` trait now has `_with_state` variants for `choose_action`,
  `respond_to_chain` (with the real, non-approximated resume cursor — see
  `ChainResumeCursor` in chain.rs), `decide`, `pick_targets`, and
  `choose_discards` — all default-fallback (zero behavior change unless a
  player opts in). Search-capable players (v7's self-play, if it wants
  chain-response-quality search like ExpertCpu's) should use these rather
  than reinventing state-threading.

**8.2 burn 0.21**
- Autodiff graph memory is freed ONLY by `backward()`. Forward-only eval →
  run on `net.valid()` (inner backend). Freezing a trunk → compute it on the
  inner backend and re-import with `Tensor::from_inner` — **never
  `detach()`** (leaked 5GB/epoch). See `frozen_or_live_body` in
  `ml/src/net_burn.rs`.
- `Param` + `Tensor::from_data` silently sets `require_grad=false` — init via
  configs, replace data via `Param::map` (see `ml/src/net_burn.rs` load fns).

**8.3 System**
- NO OpenBLAS anywhere in ML crates (heap corruption under rayon — two crash
  modes documented in `ml/Cargo.toml` comment). ndarray without `blas`.
- hermes runs earlyoom (SIGTERM at <10% avail). Stream data; never
  pregenerate >8K games; monitor trainer RSS.
- `pkill -f <pattern>` matches your own shell — kill by PID or `pgrep -x`.
- gdb attach needs `sudo -n` (yama ptrace_scope=1).

**8.4 Training (from v6 measurements)**
- REINFORCE/policy-gradient on any head: choices must be SAMPLED during data
  collection (argmax + negative advantage = thrash), advantages mean-centered
  per batch, and the shared trunk frozen for auxiliary-head steps.
- AZ: 1 epoch per iteration over the window. 4 epochs oscillates.
- Value trained on terminal outcomes only cannot rank near states — that's
  what §4.2 exists to fix. Do not skip its gate to save a day.

**8.5 Exploration scale**
- Dirichlet ε=0.25/α=0.3 with ≤200 sims on ~5-action roots poisons visit
  targets (two crashes: 38.7 → ~28 in 5 iters, with pl converging — the
  policy *learns* the poison). The α = 10/n_legal scaling + ≥400 sims in §4.3
  is the intended fix; treat it as a hypothesis with the escape hatch given there.

**8.6 Process**
- Long runs: `systemd-inhibit --what=handle-lid-switch:sleep:idle` on hermes
  (BOS suspends on lid even on AC), release when done.
- Nothing gets committed without the user's say-so. No AI attribution ever.
- Eval discipline per §2 — it is the difference between a week of progress
  and a week of chasing noise.

## 9. Open questions (decide during implementation, don't block on them)

- Autoregressive multi-pick vs greedy top-k (§3.2) — start greedy.
- Whether the aux opponent-hand head earns its 0.1 loss weight — ablate once.
- KL-to-BC anchor during AZ — v6 never needed it after best-checkpointing;
  keep in reserve if the policy drifts destructively.
- If Atlas benchmarks catastrophically (<0.5 TFLOPS): fall back to d_model
  128 / 4 layers before falling back to hermes-CPU training.

## 10. Independent review verdict (2026-07-09) — supersedes parts of §5

A fresh-context Opus review (explicitly briefed to disagree with the plan
where warranted, not just confirm it) was run against real data: 4 completed
BC epochs on Atlas (policy loss 0.52→0.45, val_acc 0.733→0.748, val_ce
0.486→0.457, ~1,200 cumulative games) plus current GPU rental pricing.
Full verdict below; where this contradicts earlier sections, THIS is current.

**GPU — rent a single ≥12GB tensor-core card (RTX 4090 class), not A100/H100.**
The model is ~6-8M params; weights+grads+Adam state total ~120MB — trivial.
The 4GB VRAM ceiling on Atlas (batch-512 OOMs, batch-64 holds at 3.5GB) is
almost entirely O(seq²) attention activations retained for autodiff
(~0.5GB/layer × 6 layers ≈ 3GB at batch-512) — a **VRAM problem, not a
compute problem**. A100/H100 add nothing this model can use; explicitly
ruled out. Verified: Pascal (Atlas's GPU) runs FP16 at 1/64 the rate of
FP32 — mixed precision is actively harmful there, don't chase it. Estimated
realistic cost for the ENTIRE remaining campaign (scaled BC + TD-value + AZ
league): **~$20-50 total** — cheap because the model is tiny and the real
bottleneck is CPU game generation, not GPU compute. Keep using Atlas (free)
only to finish the in-flight smoke-test BC run and for future pipeline
smoke-tests before spending rental hours; do not run the AZ league on it
(MCTS inference-loop-bound, ~2 TFLOPS would stretch a 30h campaign to a week).

**Architecture — hold, do not change yet.** The tell: train policy loss
(0.4379 @ epoch 3) ≈ val_ce (0.4491), both still falling — this is the
signature of **underfitting from too little data** (~1,200 games vs v6's
tens-of-thousands), not a capacity ceiling (which would show train ≪ val
with val stalled). The epoch-4 non-monotonicity (§7 table) is most likely
fresh-data resampling noise, not a training pathology. Do not resize the
network off 4 data points. One flagged-but-gated concern: `tokens.rs`
hardcodes history event_type=0 always (§3.1's "event type" field is
reserved-only, per §6/§7's judgment-call log) — given chain-response
quality was this project's single largest lever (+9.6pts, §1.1), the model
currently cannot distinguish a past chain response from a primary play in
its own history. **Do not un-defer this speculatively** — gate it on the
per-decision-kind fidelity measurement below; only fix if that shows
chain-response kinds actually lagging.

**Training — data volume is THE lever, scale it first.** Go to something
like the originally-planned ~20K games/epoch on the rented card (cheap at
that VRAM headroom, batch 512+). Add LR warmup + cosine decay (cheap
insurance, current constant 3e-4 is fine to start from). Consider
down-weighting the value-loss term (currently 1:1 with policy in
`net.rs`, BC's value target is only a terminal-outcome warm start) to
~0.25-0.5 so it doesn't distract the policy gradient — minor, not urgent.
**Highest information-per-dollar single next step, do this BEFORE renting
anything**: extend `policy_eval` (net.rs) to bucket argmax accuracy by
`ActionKind` (tokens.rs field[0] — the recorder's tests already count these
buckets, plumbing is half-built) and run it on the finished Atlas BC
checkpoint. This directly answers the chain-response-fidelity question
above with real evidence instead of guessing, and it's free (runs on Atlas
or hermes CPU, no rental needed). **DONE 2026-07-09**: implemented as a
standalone `novacana-ml7 fidelity --model <path> --val-games <n>` subcommand
(`net::policy_eval_by_kind`, `tokens::action_kind_name`) — prints a per-
`ActionKind` accuracy/CE table plus a total row. Results, a real bug it
surfaced, and the fix that came out of it: see §10.1.

**Logistics — delete the hermes+Atlas LAN split (§5), never build
`server.rs`/`client.rs`.** Strongest disagreement with the existing plan.
§5's entire justification — hermes's strong CPU alongside Atlas's free GPU —
evaporates the moment you rent, since any rented GPU instance bundles its
own 16-32 vCPUs, easily enough for AZ's ~150-200 games/iter (game gen is
embarrassingly parallel via rayon; even well below hermes's 1,400 games/s
is plenty at that game count). Run BC, TD-value, AND the AZ league all
single-box, in-process, on one rented instance — exactly like the current
`bc` command already does on Atlas. This deletes the two largest, most
bug-prone unbuilt components (milestone 3) and removes WAN-latency/Tailscale
concerns entirely. Only reconsider the split if renting is refused outright.

**Concrete next steps, in order:**
1. ~~Let the in-flight Atlas 7-epoch run finish~~ **DONE 2026-07-09.**
   Completed cleanly: epoch 7 was the best epoch (val_acc 0.7546, val_ce
   0.4389, up from epoch 1's 0.7332) — a real, if noisy, learning curve.
2. ~~Build and run the per-`ActionKind` fidelity diagnostic~~ **DONE
   2026-07-09**, with two real findings — see below. Resolved the chain-
   response question (§1.1's original concern): **NOT the bottleneck, no
   fix needed.**
3. **Rent a single 4090-class instance and re-run BC properly** — this is
   the next real action, and the only one left that needs the user (money).
   ~20K games, batch 512+, warmup+cosine LR, per-kind fidelity reported
   each eval (the tooling now exists — run `fidelity` on the resulting
   checkpoint before trusting it), existing best-val-acc checkpointing
   (now correctly separate from the final save — see the bug below). Hit
   the real §4.1 gate (≥25% floor, 30-40% target, full-ML 1000-game eval).
4. Continue TD-value and AZ on the SAME rented instance, no split, no
   `server.rs`/`client.rs`. Before trusting TD-value's gate: re-run the
   value-rank diagnostic (§8.1's `hands_dealt` bug window may have
   contaminated any earlier reading of it — check on the current fixed
   engine, don't assume).

### 10.1 Fidelity diagnostic results (2026-07-09) — two real findings

Ran `novacana-ml7 fidelity --model models/astra7_bc_run1.bin --val-games 200`
(124,201 held-out samples) against the completed 7-epoch checkpoint. Full
per-kind table is reproducible via the same command; headline findings:

**Finding 1 — chain responses are FINE, resolving §1.1's original concern.**
`ChainResponsePass` 100.0% (n=4834), `ChainResponseCard` 95.4% (n=26165) —
among the best-performing kinds, nowhere near lagging. **Per this section's
own gating rule, do NOT un-defer history token event-type tagging** — there
was never a signal problem here to fix.

**Finding 2 (new, not originally flagged) — `DecideOption` was the real
bottleneck: 60.0% accuracy on 52,893 samples, 43% of ALL decisions in the
game**, dragging the whole-model average down to 77.2%. Root cause (by code
inspection, confirmed directionally by a quick fresh checkpoint): the
decide-option token encoder (`tokens.rs::encode_action_tokens_decide`)
carried no positional field — two options in the same decision that don't
trip any of `option_flags_bucket`'s 6 keyword checks produced BYTE-IDENTICAL
tokens, an information bottleneck no amount of data could fix. This is the
exact failure class v6 already learned to avoid (every one of its candidate
encoders in `ml/src/candidate.rs` always carried a slot/total pair via
`set_pos` for this reason) — it just hadn't made it into this one v7
encoder yet. **Fixed**: added slot-index and option-count fields (reusing
two already-existing-but-previously-always-zero embedding slots — no
architecture/shape change, fully backward-compatible with existing
checkpoints as a warm-start). Quick validation: a fresh 1-epoch/20-game
checkpoint with the fix hit 72.0% DecideOption accuracy vs. the old
encoder's 60.0% (full 300-game/7-epoch run) and 45.7% (equivalent tiny
scale) — not a rigorous controlled A/B, but a strong directional signal.
**`PickTargetOption` (19.7%, n=2090) is flagged but NOT fixed** — its token
encoding already carries real content fields (unlike decide's), all 7
field slots are in active use across its 5 `Target` variants, and a
positional-field fix there would need more careful field-budget analysis
to avoid clobbering existing signal for some variant. Left for the next
person to diagnose properly rather than risk a rushed edit — re-run
`fidelity` after the real rented-GPU BC run to see if it's still lagging
at scale before spending time on it.

**Bug found while building this diagnostic — best-checkpoint saves were
silently discarded, always overwritten by the final save regardless of
which epoch was actually best.** `run_bc`'s best-checkpoint path was
`format!("{out_path}.best")`; burn-core's `FileRecorder::save` calls
`PathBuf::set_extension("bin")`, which REPLACES an existing extension
rather than appending — `"{out_path}.best"` already has `.best` as its
apparent extension, so the write collapsed to `"{out_path}.bin"`, identical
to the plain final-save path. Every "best" save was silently overwriting
(or being overwritten by) the same file the final save also targets — so
whatever survived on disk after any `ml7 bc` run was always the LAST
epoch's weights, never necessarily the best one. Exactly the class of bug
the whole mechanism exists to prevent (v6's cited "epoch 8→30 regression").
**Fixed**: `best_path = format!("{out_path}_best")` (underscore, not a
dot — `set_extension` only touches text after the last dot, so this can't
collide). Verified: a rebuilt run now produces two genuinely separate,
independently-persisting files. The completed `bc_run1` was launched with
the pre-fix binary, but got lucky — epoch 7 happened to be the true best
epoch, so no data was actually lost this time. Any run launched after this
fix should show two distinct `_best.bin`/`.bin` files; if you only see one,
something has regressed.
