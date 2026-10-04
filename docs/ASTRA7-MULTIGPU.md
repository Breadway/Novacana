# Two-T4 Data-Parallel BC for Astra-7

| | |
|---|---|
| Author | TBD |
| Date | 2026-10-05 |
| Status | Ready for implementation |
| Scope | `novacana-ml7` `bc` only, `--features tch`, fp32 |
| Crates read | burn 0.21.0, burn-tch 0.21.0, burn-optim 0.21.0, burn-nn 0.21.0, tch 0.22.0, local checkout under `~/.cargo/registry/src` |

## Overview

Astra-7 behavior cloning (`novacana-ml7 bc`) runs one Adam step on one device. On Hestia that device is the Arc A380 through `Wgpu<f32, i32>` at `--batch-size 64`. This design adds an opt-in path for one Kaggle notebook with two NVIDIA T4s (16 GB each, PCIe, no NVLink) on the existing LibTorch CUDA build, `Autodiff<LibTorch<f32>>`.

The Kaggle quota charges one GPU-hour for a T4x2 session while both cards are available. The path is only worth shipping if both cards run a forward and a backward for every Adam step. The two `bc_backward_layout` calls run on two scoped threads and are joined before any host read, `tensor.to`, or Adam step. A sequential sync-after-each-device loop is incorrect, and so is one host thread that finishes device 0's tape before it enqueues device 1. Device 0 owns the only Adam, the checkpoints, and resume. Device 1 is a weight replica. Gradients are averaged on device 0 by a plain tensor copy of the fp32 grad (4,987,394 parameters, 19,949,576 bytes), then clipped with the existing optimizer, then stepped once, then broadcast back. There is no NCCL dependency and no `DistributedBackend`. `burn-tch` 0.21 does not implement that trait.

`--batch-size` stays the per-device micro-batch. It is not reinterpreted as a global batch. The new flag is `--devices N`, default 1. Absent, or set to 1, every current command keeps one device and the current batch default, including the Vulkan trainer. On that one-device path the heartbeat scalars are still read with `download_f32` before `backward`.

## Background and Motivation

### Current trainer

`ml7/src/main.rs` selects the backend at compile time. Enabling both `tch` and `vulkan` is a `compile_error` (lines 29-30). The Vulkan alias is `Wgpu<f32, i32>` (line 36). The comment above it records why: fp16 forced Adam's epsilon to `1e-2` and stalled the policy, and batch 64 is the largest fp32 batch that fits in 6 GB. Batch 128 OOMs the Arc. This spec does not change that binary, that dtype, or that default.

`cmd_bc` (`ml7/src/main.rs` around lines 910-933) defaults are: `--chunk-games 2000`, `--epochs 15`, `--batch-size 64`, `--lr 0.0003`, `--label-smoothing 0.05`, `--val-games 200`, `--winrate-every 1`, `--winrate-games 150`, `--checkpoint-every 300`. The teacher is `ExpertCpuPlayer` (`ml7/src/recorder.rs`). `--model` and `--resume` are mutually exclusive (lines 941-946). SIGTERM sets a flag, the loop finishes the current step, then writes the bin plus `astra7_bc.resume.txt`. `ResumeMeta` and `save_resume_meta` are lines 141-163. `install_shutdown_flag` is lines 237-247, not inside 123-163. The checkpoint poll that honors the flag is lines 550-577.

The tch arm of `cmd_bc` hardcodes `LibTorchDevice::Cuda(0)` (line 956). The same `Cuda(0)` literal is the CUDA entry point for `bench` (line 275), `fidelity` (733), `token-scan` (777), `behavior` (840), `winrate` (881), `td-value` (1259), `value-rank` (1328), and `az` (1578). Those commands stay on `Cuda(0)`.

`run_bc` (lines 403-656) generates games on the CPU, length-groups sequences (`length_grouped`, lines 65-70), and calls `net::bc_train_step` once per chunk. `global_step` increments once per call. The LR schedule is 5% warmup (`total_steps_est / 20`, floor 50) then cosine, indexed by `global_step`. `steps_per_epoch_est` divides by `batch_size` (lines 466-468). Heartbeat every 50 steps prints the last sampled policy and value and the mean of those heartbeat samples since epoch start. `checkpoint_every` counts those same steps. `HEARTBEAT_EVERY` is 50.

### What one step does today

`bc_train_step` / `bc_train_step_layout` (`ml7/src/net.rs` lines 296-365) run the compact forward, masked policy cross-entropy with label smoothing 0.05, and value MSE. `LOGIT_MASK` is `-1e4` (line 53). The value term is weighted by `VALUE_LOSS_WEIGHT = 0.3` only inside the scalar that is backward'd (lines 358-364). The heartbeat scalars are the unweighted hard CE mean and the unweighted value MSE, and only when `report_metrics` is set. The comment at lines 346-347 is the sync rule this design extends: reading a scalar back from CUDA synchronizes the device, so training does not do it on every step. Today's order on that function is `download_f32`, then `total.backward()`, then `optim.step`.

The Arc trainer is not an eager backend. `ml7/Cargo.toml` enables `burn-wgpu` features `fusion` and `vulkan`. Under `feature = "fusion"`, `Wgpu` is `Fusion<CubeBackend<WgpuRuntime, ...>>` (`burn-wgpu-0.21.0/src/lib.rs` lines 63-64). `download_f32` is `into_data` (`ml7/src/net.rs` lines 55-57), and that read is what drains the fusion queue. `Tensor::inner` (`burn-tensor-0.21.0/src/tensor/api/autodiff.rs` lines 57-61) only unwraps the primitive. It does not materialize. Holding an inner handle across `backward` and `optim.step` on a `report_metrics` step leaves an extra live consumer in the fusion queue. That can change which ops fuse on heartbeat steps (every 50th batch and the last batch of the epoch). The one-device path therefore keeps today's order. The two-device path is LibTorch, which is eager, and is the only path that defers the read.

`GradientsParams::from_grads(total.backward(), &net)` then `optim.step`. The optimizer is `make_optimizer` (lines 276-284): `AdamConfig::new()` with `GradientClippingConfig::Norm(1.0)` and the default epsilon. Burn 0.21's `AdamConfig` default epsilon is `1e-5` (`burn-optim-0.21.0/src/optim/adam.rs`). No weight decay. `docs/ASTRA7-NET.md` calls this "global-norm gradient clipping at 1.0". The implementation applies `Norm` inside `SimpleOptimizerMapper::map_float` once per parameter tensor (`burn-optim-0.21.0/src/optim/simple/adaptor.rs` lines 199-214, `grad_clipping/base.rs` `clip_by_norm`). It is not one norm of the flattened 4,987,394-vector. This change preserves that behavior. It does not add a second clip and it does not clip per device before the average.

### Model and checkpoint

`docs/ASTRA7-NET.md`: bidirectional encoder, sequence length 180, `d_model` 256, 8 heads, 6 layers, FFN 1024, 4,987,394 parameters. Constants match `ml7/src/net.rs` lines 64-69 and `ml7/src/tokens.rs` (`SEQ_LEN` 180, `NUM_FIELDS` 7). Dropout is 0.1, set in `Net::fresh` via `TransformerEncoderConfig::with_dropout(0.1)` (net.rs lines 88-91). There is no BatchNorm. A full-precision copy of the weights is 19.0 MiB (`4,987,394 * 4 = 19,949,576` bytes). Checkpoints are `BinFileRecorder<FullPrecisionSettings>` through `Module::save_file` / `load_file`. `STATE.md` measures the file at 19,952,515 bytes. Adam moments are not in the file. Resume rebuilds Adam and restores `epoch`, `global_step`, and `best_acc` from the sidecar.

Activations dominate VRAM, not the weights. Measured numbers, not new ones:

- Atlas GTX 1050 Ti (4 GB): fp32 batch 64 holds a flat 3531 MiB (`ASTRA7.md`, the bring-up note near the 7-epoch run).
- RTX A4000 16 GB: AZ batch 256 peaked at 13.5 GB. The comment on `cmd_az` (`ml7/src/main.rs` lines 1557-1561) says that peak is roughly linear in batch and is the same trunk and heads as `bc_train_step`. 256 sequences on one 16 GB card is near the top. 128 is the comfortable per-T4 default. 256 per T4 is allowed and may OOM.

### Why the manual step

`DistributedBackend` exists in burn-backend 0.21 (`burn-backend-0.21.0/src/backend/distributed/ops.rs`). Local impls are:

- `Autodiff<B, C>` only when `B: DistributedBackend` (`burn-autodiff-0.21.0/src/ops/distributed.rs`)
- `Fusion<B>` only when the inner backend implements it (`burn-fusion-0.21.0/src/ops/distributed.rs`)
- `CubeBackend` (`burn-cubecl-0.21.0/src/ops/distributed.rs`)
- `Dispatch` (`burn-dispatch-0.21.0/src/ops/distributed.rs`)

There is no `impl DistributedBackend for LibTorch`. `burn-tch-0.21.0` has no distributed module. A search of that crate for `DistributedBackend` returns nothing. The non-distributed `AutodiffTensor::backward` (`burn-autodiff-0.21.0/src/tensor.rs` lines 185-190, the `not(feature = "distributed")` method) walks the tape and does not call `sync_collective`. That is the backward this binary gets, because `LibTorch` is not a `DistributedBackend` and `ml7` does not enable Burn's `distributed` feature.

`LibTorch::device_count` is hardcoded to 1 (`burn-tch-0.21.0/src/backend.rs` lines 174-177: "tch only supports one device for each backend"). Visible GPU count must come from `tch::Cuda::device_count()` (`tch-0.22.0/src/wrappers/device.rs` lines 20-23), which returns `i64`, not `usize`. Call sites cast with `usize::try_from(tch::Cuda::device_count()).unwrap_or(0)`. Zero, including a negative `i64` that failed the cast, fails `--devices 2` closed. It is not a one-GPU success. `tch::Cuda` in that version exposes `device_count`, `is_available`, `synchronize`, cuDNN toggles, and seeds. It does not expose a device-name query. Startup prints `LibTorchDevice` indices (`Cuda(0)`, `Cuda(1)`), not a marketing name.

`Backend::sync` for CUDA is `tch::Cuda::synchronize(index)` (`burn-tch-0.21.0/src/backend.rs` lines 143-147). Host readback is `float_into_data`, which does `Vec::<f32>::try_from(&tensor)` (`burn-tch-0.21.0/src/ops/tensor.rs` lines 69-80), which calls `Tensor::f_copy_data` into a host buffer (`tch-0.22.0/src/tensor/convert.rs` lines 7-20 and `src/wrappers/tensor.rs` lines 396-417). The Rust wrapper does not itself call `Cuda::synchronize`. `download_f32` in `net.rs` is that path. `TchOps::to_device` is `tensor.tensor.to(device)` and returns the new tensor without a host read (`burn-tch-0.21.0/src/ops/base.rs` lines 12-22). That `to` is the allreduce, and it runs only after both backwards have been joined. `Backend::seed` is `fn seed(_device: &Device, seed: u64)` (`burn-tch` `backend.rs` lines 125-126). There is no one-argument `B::seed`.

## Goals and Non-Goals

### Goals

- Opt-in `--devices 2` on `bc` only, tch fp32, so each T4 runs a forward and a backward for every Adam step, and those two launches overlap on two scoped threads.
- One Adam on device 0, existing hyperparameters, no LR retune.
- Averaged gradient matches one device at batch `2B` when dropout is off, halves are equal, and the layout is shared. Tolerance is stated below.
- Same checkpoint format, same resume sidecar, same single-device commands. "Behavior-identical" on the one-device path means weights, `global_step`, checkpoint bytes, and heartbeat counters. The new `bc devices=1 ...` startup line is expected. Stdout is not byte-identical. The one-device heartbeat still calls `download_f32` before `backward`.
- A CPU unit test for the weighted average, the parameter pairing (including equal `ParamId`s in two containers), and the one-device versus two-device cutter. Hermes `cargo test -p novacana-ml7` without `--features tch` stays green. No CI test requires a CUDA device.

### Non-Goals

- fp16, bf16, or any change to the Vulkan fp32 Adam epsilon history. That failure is context only.
- Changing the Arc trainer: `Wgpu<f32, i32>`, one `DiscreteGpu(0)` (unless `NOVACANA_WGPU_DEVICE` is set, which already exists), default `--batch-size 64`. Moving the one-device heartbeat read to after `step` is one of those changes and is out of scope.
- Reinterpreting `--batch-size` as a global batch. Default stays 64.
- Scaling LR with global batch. `3e-4` was used at global batch 64 on the Arc. A 2-GPU run at per-device 128 is global batch 256 and is a different noise scale. That is a known difference, not a second feature.
- NCCL, `burn-collective`, or implementing `DistributedBackend` for `LibTorch`.
- A second Adam, per-device clip-then-average, or a one-device remainder step on the two-device path.
- Data-parallel `td-value`, `az`, `bench`, `winrate`, `behavior`, `fidelity`, `value-rank`, or `token-scan`.
- Moving validation, the 150-game win-rate eval, or the spectator game off device 0. They stay single-device.
- Dashboard changes. Heartbeat line shape stays, including the elapsed suffix.
- Moving `ExpertCpuPlayer` generation off the CPU. On Hestia, 200 games produced 124,730 validation samples in 7.3 s (`STATE.md`). That is a small fraction of BC wall time. Generation stays in `recorder::generate_samples` / `generate_samples_abortable`. Kaggle's CPU is about 4 cores and about 29 GB RAM. The runbook may pass `--gen-concurrency 4`. `batch_infer::default_concurrency` stays `(2 * cores).max(8)` and is not retuned here.
- Copying a Hestia Vulkan binary onto the notebook. glibc and the backend differ. `ml7/build.rs` requires `LIBTORCH` and forces `-ltorch_cuda` with `--no-as-needed` only for the `tch` feature. Build on the notebook with `--features tch`.

## Key Decisions

1. **Flag `--devices`, default 1.** `1` is today's one-device BC, including when two CUDA devices are visible. `2` is the only other legal value. Anything else exits before training. The flag is rejected on non-tch builds unless the value is 1. Rationale: multi-GPU must be opt-in so a visible second GPU cannot silently double the global batch, and the Arc binary must not grow a second device path.

2. **`--batch-size` is the per-device micro-batch.** Global batch is `devices * batch_size` and is printed, not parsed. Kaggle comfortable command uses `--batch-size 128` (global 256). Optional `--batch-size 256` is global 512 and may OOM. Rationale: the flag already means the slice passed to `bc_train_step`. Reinterpreting it would change every existing script.

3. **One Adam on device 0, then broadcast.** No second optimizer and no per-device clip. Average by sequence count, then one `optim.step`, which applies the existing per-tensor `Norm(1.0)` and Adam. Rationale: clip-then-average is a different function, and a second Adam would make device 0's checkpoint not the training state.

4. **Pair parameters by module path, not by `ParamId` and not by `HashMap` iteration order.** Both sides must have the same paths, ranks, and dims, 112 float tensors, and 4,987,394 elements. A `None` grad is a hard error named by path. Invariant: `clone().fork` keeps ids. `Param::map` keeps `id` (`burn-core-0.21.0/src/module/param/base.rs` lines 294-306). `Param::fork` is that map (`param/tensor.rs` lines 306-316). `burn-optim`'s `test_convert_grads` asserts `param_ids_1 == param_ids_2` after `clone()` then `fork`. `Net::fresh` mints new ids. After `net.clone().fork(&devices[1])`, `id0 == id1` on all 112 slots is the expected result, not a duplicate to skip. Equal ids are still two grad tensors in two `GradientsParams` containers. `remove` both by that shared id, average, `register` only into device 0's container, drop device 1's container, and never insert both sides into one map. `GradientsParams` is a `HashMap` keyed by `ParamId`, and `register` replaces an existing id (`burn-optim-0.21.0/src/optim/grads.rs`). A single-map merge keeps `len() == 112` while discarding one side. `SimpleOptimizerMapper::map_float` (`adaptor.rs` lines 178-234) does `grads.remove(id)` and, on `None`, leaves that weight unchanged with no error. `GradientsParamsConverter` also skips `None` (`visitor.rs` lines 38-40). Length equality is not a check. Paths are the key. Re-fork each step is unnecessary, not id-invalidating: fork does not mint or drop ids, and the alias with the ids Adam stores already exists from the initial fork. Do not rebuild the replica each step.

5. **Overlap is two scoped threads and no host sync until both have joined.** The forbidden calls, until both `bc_backward_layout` threads return, are `download_f32`, `Tensor::into_data`, `Vec<T>::try_from` on a `tch::Tensor`, `f_copy_data`, `B::sync`, and `tch::Cuda::synchronize`. `tensor.to(device)` is allowed only after the join, and it is the allreduce. One host thread that runs device 0's forward and its entire backward before any kernel is queued on device 1 leaves device 1 idle for that whole launch. Overlap would then be only whatever is still running on device 0. Rationale: that launch, and a sync between the two calls, is slower than global batch 256 on one T4 and leaves the second card idle inside a quota hour that already paid for both. The replica must be `clone().fork`, not a bare clone. `GraphMutexClient` stores graphs that share nodes under one mutex, "limiting parallelisation" (`burn-autodiff-0.21.0/src/runtime/graph.rs` lines 32-35). `backward` holds that graph's `state` mutex for the tape walk (lines 114-123). `graph()` locks the process-global `STATE` mutex only for `GraphLocator::select` (lines 89-100), then `register` locks the per-graph mutex (lines 105-110). The short global lock is not a reason to skip the second thread. Fork-disjoint graphs get separate per-graph mutexes. `TchTensor` is `Send + Sync` (`burn-tch-0.21.0/src/tensor.rs` lines 215-216). Raw `tch::Tensor` is `Send` only (`tch-0.22.0/src/wrappers/tensor.rs` line 22). Threads use the Burn module, not a split raw `tch::Tensor`.

6. **`global_step` counts Adam updates.** Heartbeat 50 and `--checkpoint-every 300` stay 300 Adam steps, not 300 micro-batches and not 600. The cosine denominator uses `batch_size * devices.len()` so a fresh 2-GPU run still spans the configured epochs. For one device the divisor is still `batch_size`. LR itself is not scaled. Rationale: the schedule index is already `global_step`. Leaving the denominator at the per-device batch would make a 2-GPU run finish the epoch list around the middle of the cosine.

7. **Remainder policy differs by device count, on purpose.** One device keeps `length_grouped`'s current rule (keep a trailing chunk of length >= 2, drop length 1). Two devices run a step only when both cards have a full `--batch-size` slice, and drop the tail, including a tail that is exactly one full micro-batch. `adam_steps` is `floor(n / global_batch)`. `dropped_tail` is the sequence count `n % global_batch`, not a step count. Any nonzero remainder removes one Adam step and `dropped_tail` sequences. It does not remove `dropped_tail` steps. Two devices also stride each window (even indices to device 0, odd indices to device 1) so both cards see the same span mix. A contiguous short/long split loads the cards unequally because attention is quadratic in span. Rationale: a one-device remainder step on the two-device path would change the global batch and would idle a card, and the locked rule forbids it. Applying the stricter rule to one device would change Hestia's update count.

8. **Parity is a separate `bc-parity` command, dropout 0, shared dense layout, equal halves, three fork-disjoint graphs.** Training dropout stays 0.1 and is independent per replica. The reference is `clone().fork`, not a bare clone. Rationale: Bernoulli masks (`Dropout::forward` uses `random_like(Bernoulli)`, `burn-nn-0.21.0/src/modules/dropout.rs`) make a naive bitwise compare fail, and per-micro-batch compact layouts are not the comparison the correctness rule is making. A shared `NodeRef` aliases the autodiff graph the tolerance subtracts against.

9. **Manual two-device step instead of NCCL or one fat single-GPU batch.** One T4 at batch 256 is the same global batch as 128+128 and needs no new code. It is the right fallback if overlap cannot be shown. It does not satisfy the quota reason for this work, which is that both T4s do a forward and a backward on every Adam step. See Alternatives.

## Proposed Design

### Command lines

Unchanged Hestia command (`STATE.md`), Vulkan binary, no `--devices`:

```
cd ~/Projects/novacana && nice -n 5 script -a -q -f -c \
  './target/release/novacana-ml7 bc --resume models/astra7_bc.bin \
   --out models/astra7_bc.bin --batch-size 64 --chunk-games 2000 \
   --epochs 15 --gen-concurrency 8' logs/astra7-bc.log
```

Kaggle, build on the notebook:

```
cargo build --release -p novacana-ml7 --features tch
```

Comfortable, 128 sequences on each T4, global batch 256:

```
./target/release/novacana-ml7 bc --devices 2 --batch-size 128 \
  --out models/astra7_bc.bin --chunk-games 2000 --epochs 15 \
  --gen-concurrency 4
```

Optional, 256 sequences on each T4, global batch 512. Allowed. May OOM:

```
./target/release/novacana-ml7 bc --devices 2 --batch-size 256 \
  --out models/astra7_bc.bin --chunk-games 2000 --epochs 15 \
  --gen-concurrency 4
```

`--gen-concurrency 4` is a runbook choice for a 4-core notebook. It is not a new default. Resume and warm start use the existing `--resume` / `--model` flags and the same `--devices 2`.

### Flag and startup

```rust
fn resolve_bc_devices(requested: usize, visible_cuda: Option<usize>) -> Result<usize, String>
```

- `requested == 1`: return `Ok(1)`. Ignore `visible_cuda`. Do not call `tch::Cuda::device_count` on this path, so a one-device tch run keeps today's failure mode if CUDA is missing.
- `requested == 2`: require `visible_cuda == Some(n)` with `n >= 2`. Otherwise return an error whose text contains `found {n}` and `Refusing to train on one GPU`. Exit 1 before `Net::fresh` and before game generation.
- Any other `requested`: error, `only 1 or 2`, exit 1.
- Non-tch build and `requested != 1`: error, `requires the tch feature`, exit 1. Vulkan and ndarray never construct a second device.

The tch call site, only when `requested == 2`, does `let visible = usize::try_from(tch::Cuda::device_count()).unwrap_or(0);` and passes `Some(visible)`. `device_count` returns `i64`. Do not pass that `i64` where a `usize` is required, and do not call `LibTorch::device_count`. `requested == 1` does not call `device_count` at all.

`cmd_bc` parses `--devices` with default 1 in every build. The tch arm, only when the resolved count is 2, sets devices to `LibTorchDevice::Cuda(0)` and `LibTorchDevice::Cuda(1)`. Count 1 keeps the existing `Cuda(0)` local.

Every `bc` run prints one line before generation:

```
bc devices=1 per_device_batch=64 global_batch=64 device=Cuda(0)
```

```
bc devices=2 per_device_batch=128 global_batch=256 devices=Cuda(0),Cuda(1)
bc lr=0.000300 not scaled (3e-4 was tuned at global batch 64 on the Arc; this run's global batch is 256)
```

Vulkan prints `device=DiscreteGpu(0)` (or whatever `vulkan_device()` returned) and `devices=1`. The line is how an operator sees that a two-GPU box did not silently double the batch. It is new stdout. It is not a claim that the rest of the log is byte-identical.

### Where the loop changes

`run_bc` gains `devices: &[B::Device]` with `devices.len()` already resolved to 1 or 2. Callers that are not the 2-GPU tch arm pass a one-element slice. The epoch shell stays one function: generation, validation, win-rate, spectator, checkpoint, SIGTERM. Do not fork a second copy of `run_bc`.

- Data generation, val set, and `ExpertCpu` stay on the CPU, as now.
- `let mut net = initial.unwrap_or_else(|| Net::fresh(&devices[0]));`
- `let mut opt = net::make_optimizer::<B>();` once. This is the only Adam.
- If `devices.len() == 2`: call `net::fork_bc_replica(&net, &devices[1])` once, before the first step, including after `--resume` / `--model` load. It collects the primary slots first, materializing every lazy weight on device 0, then calls `clone().fork` and pairs the slots. Burn 0.21 clones an uninitialized parameter into an independent lazy initializer, so cloning before this materialization draws different weights even though the IDs match. Load happens once, on device 0, in `cmd_bc`, exactly as today. A bare `clone()` is not the replica.
- Validation, `eval_win_rate`, and `spectator::record_spectator_game` keep using `net.valid()` and `devices[0]`. The second card is idle during those. That idle is accepted. Do not parallelize them in this change.
- Checkpoints call `net.clone().save_file` (device 0 only). The sidecar stays `epoch`, `global_step`, `best_acc`. Adam state is still not saved.

`steps_per_epoch_est` today divides by `batch_size` (main.rs lines 466-468). Change the divisor to `batch_size * devices.len()`. For one device the number is identical. Warmup is still `(total_steps_est / 20).max(50)`, and `global_step` from the sidecar is still the index. Document the resume footgun in the runbook and in `bc` help: a Hestia `global_step` (built at batch 64, one device) resumed with `--devices 2` and a larger per-device batch sits further along a shorter cosine, because `total_steps_est` is recomputed from the new global batch. That is not a new schedule mode. A clean Kaggle run starts without `--resume`, or uses `--model` if the intent is a warm start that resets Adam, `best_acc`, and the schedule (existing `--model` semantics).

### Micro-batches

One device: delegate to today's `length_grouped` and the `chunk.len() < 2` skip. A trailing partial chunk of length >= 2 still trains. A chunk of length 1 is still dropped. Do not route one device through the two-device window cutter.

Two devices: sort by `net::sequence_span` the same way, then cut exact windows of `2 * batch_size`. Drop the tail (`n % (2 * batch_size)` sequences), including a tail whose length is exactly `batch_size` (one full micro-batch and nothing for the other card). Shuffle the list of windows. Do not shuffle sequences inside a window. Then stride each window: even indices to device 0, odd indices to device 1. Both slices have length `batch_size` because the window length is even. Both see the same length mix. Do not give device 0 the first contiguous half and device 1 the second. That first half is the shorter half, and attention is quadratic in span. Do not shuffle after the stride. Reject `--devices 2` with `--batch-size < 2` at startup (the one-device path already refuses to train a chunk shorter than 2).

Once per epoch, after the windows are built:

```
bc epoch 1: sequences={n} adam_steps={windows} dropped_tail={n % global_batch} global_batch={global}
```

`adam_steps` is `floor(n / global_batch)`, the window count. `dropped_tail` is a sequence count, `n % global_batch`. They are not the same unit. Any nonzero remainder removes one Adam step and `dropped_tail` sequences. It does not remove `dropped_tail` Adam steps. A 2-GPU epoch has fewer updates than `ceil(n / batch_size)`. When the remainder is nonzero it also has one fewer update than `ceil(n / global_batch)`.

Per window, each replica builds its own `BatchLayout::for_batch` on its own slice, same as today's compact forward. Layouts may differ by masked padding rounded up to 8 tokens. That is the production behavior. It is not the parity setup.

The cutter is a pure function the CPU test calls. Device count 1 returns the `length_grouped` chunks of length >= 2. Device count 2 returns the strided windows, `adam_steps`, `dropped_tail`, and the cosine divisor `batch_size * devices`.

### Split of `bc_train_step_layout`

`bc_train_step` and `bc_train_step_layout` stay the one-device entry points used by `run_bc` when `devices.len() == 1`. The two-device path does not call them. `benchmark` keeps calling the one-device wrapper (`net.rs` call sites around lines 950 and 973). `bench` stays on `Cuda(0)`.

Shared forward, in `ml7/src/net.rs`. No `backward`, no `download_f32`, no `optim.step`:

```rust
struct BcForward<B: AutodiffBackend> {
    total: Tensor<B, 1>,
    /// Still on the autodiff backend. None unless report_metrics.
    /// `policy_ce` is hard CE mean (not the smoothed training loss).
    /// `value_mse` is unweighted value MSE (the 0.3 factor is only in `total`).
    policy_ce: Option<Tensor<B, 1>>,
    value_mse: Option<Tensor<B, 1>>,
    sequences: usize,
}

fn bc_forward_layout<B: AutodiffBackend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    label_smoothing: f32,
    report_metrics: bool,
    layout: BatchLayout,
    device: &B::Device,
) -> BcForward<B>;
```

Loss formula is unchanged: masked logits plus `LOGIT_MASK`, label smoothing 0.05, value MSE, `total = policy_loss + value_loss * 0.3`. When `report_metrics` is false, both `Option`s are `None` and no extra mean node is kept. That is every non-heartbeat step. `Tensor::mean()` returns `Tensor<B, 1>`.

The one-device wrapper, used only when `devices.len() == 1`, keeps today's order. `download_f32` consumes the metric tensors before `backward`. Do not also keep an `.inner()` handle on this path:

```rust
let fwd = bc_forward_layout(..., report_metrics, ...);
let pl = fwd.policy_ce.map(|t| download_f32(t).first().copied().unwrap_or(f32::NAN));
let vl = fwd.value_mse.map(|t| download_f32(t).first().copied().unwrap_or(f32::NAN));
let grads = GradientsParams::from_grads(fwd.total.backward(), &net);
let net = bc_apply_step(net, optim, lr, grads);
(net, pl, vl)
```

On Fusion, that read is what drains the queue, matching `bc_train_step_layout` lines 346-364. Moving it to after `step` can change which ops fuse on heartbeat steps, so the weights written by the in-progress Hestia run would not be behavior-identical. Hermes `cargo test` uses `NdArray` and cannot see that. PR 2 must not be what the Hestia `script` command restarts onto unless a Vulkan heartbeat step has been shown to match this wrapper. Behavior-identical means weights, `global_step`, checkpoint bytes, and heartbeat counters.

Two-device backward. This is what each scoped thread calls. It does not download:

```rust
pub struct BcBackward<B: AutodiffBackend> {
    pub grads: GradientsParams,
    /// Inner-backend scalars. None unless report_metrics.
    pub policy_ce: Option<Tensor<B::InnerBackend, 1>>,
    pub value_mse: Option<Tensor<B::InnerBackend, 1>>,
    pub sequences: usize,
}

fn bc_backward_layout<B: AutodiffBackend>(
    net: &Net<B>,
    seqs: &[TokenSeq],
    targets: &[u16],
    outcomes: &[f32],
    label_smoothing: f32,
    report_metrics: bool,
    layout: BatchLayout,
    device: &B::Device,
) -> BcBackward<B> {
    let fwd = bc_forward_layout(...);
    let policy_ce = fwd.policy_ce.map(|t| t.inner());
    let value_mse = fwd.value_mse.map(|t| t.inner());
    let grads = GradientsParams::from_grads(fwd.total.backward(), net);
    BcBackward { grads, policy_ce, value_mse, sequences: fwd.sequences }
}
```

`.inner()` is not a host read. Do not call `download_f32`, `into_data`, `detach`, `B::sync`, or `Cuda::synchronize` in this function. `detach()` leaks graph nodes (`td_train_step` comment, net.rs lines 373-376, about 5 GB/epoch in v6). The host downloads the inner scalars only after both threads have joined.

```rust
fn bc_apply_step<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
    net: Net<B>,
    optim: &mut O,
    lr: f64,
    grads: GradientsParams,
) -> Net<B>;
```

`bc_apply_step` is `optim.step(lr, net, grads)` and nothing else. Clip happens inside that call, on whatever tensors are in `grads`, on the device those tensors live on. The caller must pass already-averaged grads on device 0. The public one-device return type stays `(Net<B>, Option<f32>, Option<f32>)`.

### One Adam step on two devices

```mermaid
sequenceDiagram
    participant Host
    participant T0 as thread cuda0
    participant T1 as thread cuda1
    participant C0 as cuda0
    participant C1 as cuda1

    Host->>T0: scope spawn bc_backward_layout(net, even slice)
    Host->>T1: scope spawn bc_backward_layout(replica, odd slice)
    Note over T0,T1: join before any download, into_data, f_copy_data, sync, or tensor.to
    T0->>C0: forward then backward, eager ATen, no host read
    T1->>C1: forward then backward, eager ATen, no host read
    T0-->>Host: BcBackward joined
    T1-->>Host: BcBackward joined
    opt Host->>Host: heartbeat only, download both inner loss scalars
    Host->>C0: replica grads via tensor.to(cuda0), weighted mean
    Host->>C0: bc_apply_step, Norm clip per tensor, one Adam
    Host->>C1: broadcast device-0 weights, detach, require_grad, keep replica ParamIds
    Host->>Host: global_step += 1
```

Host order inside the window. Each device has its own phase atomic. Thread 0 writes only the cuda:0 slot. Thread 1 writes only the cuda:1 slot. A third host slot is written only by the host thread, and only after both joins. Do not store both device phases into one atomic: both stores happen at entry, before either backward runs, and last writer wins, so a panic on device 0 prints `cuda:1 forward-backward` whenever thread 1 stored later. `Cell` is not `Sync`. The hook prints all three slots. libtorch often reports a CUDA error on the next synchronizing call, so the host slot can name `grad-average` rather than the kernel that faulted. See Failure behavior.

1. `std::thread::scope`. Thread 0 stores `forward-backward` into the cuda:0 slot, then calls `bc_backward_layout` on `&net`, `devices[0]`, and the even-index slice. Thread 1 stores `forward-backward` into the cuda:1 slot, then calls `bc_backward_layout` on `&replica`, `devices[1]`, and the odd-index slice. On a normal return, that thread stores `returned` into its own slot and does not touch the other. Each thread uses only its own module and its own batch. Do not share a raw `tch::Tensor` across threads. Do not `clone()` the module inside the thread: that would share `NodeRef`s with the parent graph.
2. Join both. The overlap window ends at the join. No `download_f32`, `into_data`, `f_copy_data`, `B::sync`, `Cuda::synchronize`, `tensor.to`, or `bc_apply_step` runs before both joins return. No host read of either loss or either grad between the two spawns. `AutodiffTensor::backward` on this backend only walks the tape. It does not call `LibTorch::sync`.
3. The two graphs are fork-disjoint because the replica was built with `clone().fork` before the loop. A bare clone shares `NodeRef` (`burn-autodiff-0.21.0/src/tensor.rs` lines 13-17) and `GraphLocator` merges those nodes under one mutex, so the two `backward` calls serialize on `graph.state`. The global `STATE` lock is a short critical section per op (`graph()` lines 89-100). It is not a reason to go back to one thread.
4. burn-tch does not use libtorch autograd. Each thread issues eager ATen ops on its own device, including in-place Bernoulli (`f_bernoulli_float_`) on that thread's activation tensor. Dropout stays 0.1 in training and is independent per replica. The threads do not take a device-wide host lock and do not call `Cuda::synchronize`. If a libtorch build still serializes every CUDA launch on one host mutex, the runbook overlap check is what catches it. The fallback is one T4 at `--batch-size 256`, not a second Adam.
5. If this Adam step is a heartbeat (`(nb + 1) % 50 == 0` or last window of the epoch): the host stores `grad-average` into the host slot, then `download_f32` each inner scalar and combine. This syncs. That is correct, because both launches are already queued and both threads have joined. Doing it before the join is the bug. The one-device path does not use this deferred read. It already downloaded before its own `backward`. The device slots are already `returned` by this point. They are not cleared to hide which device ran.
6. Still in host slot `grad-average`: average grads onto device 0 with `tensor.to(device0)`. Do not `into_data` the grad and upload it again. A missing grad aborts here, before `bc_apply_step`.
7. The host stores `adam` into the host slot. `net = bc_apply_step(net, &mut opt, cur_lr, averaged)`.
8. The host stores `broadcast` into the host slot. Broadcast `net` onto `replica` (below). The next window's replica forward is enqueued after this broadcast so device 1's default stream orders the copy before the next forward. No extra `Cuda::synchronize` between windows.
9. `nb += 1` and `global_step += 1` once. Checkpoint when `checkpoint_every > 0 && nb % checkpoint_every == 0`. 300 stays 300 Adam steps. SIGTERM is polled after the broadcast, not between the two backwards and not while the scope is still running. Finish the in-flight Adam step, save device 0, write the sidecar, exit.

`cur_lr` is the existing warmup-then-cosine expression, using `global_step` from before the increment, same as the current loop.

### Weighted gradient

Losses are means. For sequence counts `n0` and `n1` (both equal to `--batch-size` in this loop):

```
g = (n0 * g0 + n1 * g1) / (n0 + n1)
```

That is the gradient of the concatenated-batch mean. Equal counts reduce to a plain mean of the two grad tensors. The training loop asserts `n0 == n1` and `n0 == batch_size`. The helper still takes counts so the CPU test can lock the unequal formula, and so a later caller cannot "simplify" it into a sum.

```rust
pub fn weighted_average<B: Backend, const D: usize>(
    grads_on_same_device: &[Tensor<B, D>],
    sequence_counts: &[usize],
) -> Tensor<B, D>;
```

Preconditions: same device, same shape, equal slice lengths, every count > 0. Zero total count is an error, not a NaN from division. Implementation is `sum_i grad_i * (count_i as f32) / total`, on device, no host read.

`None` is not a zero. `GradientsParams::remove` returning `None` aborts the step before `bc_apply_step`, with the module path in the error. Do not register a zero, and do not skip the slot. A skipped slot would leave that parameter's Adam moment untouched while the other replica's copy of the weight gets broadcast, or worse would apply only one side. Unused embedding rows are zeros inside a present grad tensor, not `None`. `None` means the parameter was not in the tape.

The average loop does not iterate `container.ids()`. `GradientsParams::remove::<B, D>` ends in `TensorContainer::remove`, which downcasts to a rank-erased `TensorPrimitive<B>` (`burn-backend-0.21.0/src/tensor/container.rs` lines 79-87). `None` means the id was absent. A wrong `D` is a later shape panic, or a silent mis-read, not the missing-grad error. Rank comes from the slot. For each `PairedSlot` in visit order:

1. `D` is `slot.rank`, which is 1 or 2. `remove::<B, D>` from device 0's container with `slot.id0`, and from device 1's container with `slot.id1`.
2. Require both `Some`. If either is `None`, abort before `bc_apply_step` and name `slot.path`. Do not treat `None` as zero.
3. `to_device(cuda:0)` on the device-1 grad. The device-0 grad is already there.
4. `weighted_average` of those two tensors with counts `n0, n1`.
5. `register` the result only under device 0's id. After the loop, drop device 1's container. Never insert both grads into one `GradientsParams`.

Because `clone().fork` keeps ids, `id0 == id1` on every slot. `remove` still finds each grad, because they live in two maps. A CPU test locks that: two rank-matched grads with the same `ParamId` in two containers produce `(n0 * g0 + n1 * g1) / (n0 + n1)`, and a single-map `register` of the second id replaces the first so that map's length stays 1.

After a successful average, every pair has been removed from both containers and registered only under device 0's `ParamId`. Never pass device 1's grads to `optim.step`. The adaptor moves the parameter to the grad's device when they differ (`adaptor.rs` lines 188-192). A device-1 grad would move the only Adam onto `cuda:1`.

### Parameter pairing

Burn 0.21's `#[derive(Module)]` visit (`burn-derive-0.21.0/src/module/codegen_struct.rs`) calls `enter_module(field_name, "Struct:TypeName")`, then `Module::visit`, then `exit_module`. `Vec`'s visit uses the index string and container type `"Vec"` (`burn-core-0.21.0/src/module/param/primitive.rs` lines 106-125). `Option` visits the inner value and does not push `"Some"`. `Param::visit` calls `visit_float` and does not call `visit_float_with_path`. The path is the `enter_module` stack, which is how `burn-store`'s `Collector` builds names (`burn-store-0.21.0/src/collector.rs`). Visit and map both walk parameter fields in source order. They do not walk a `HashMap`.

Record a slot only in `visit_float`. Dropout, `norm_first`, and other non-param fields push no slot. This net's float tensors are rank 2 (embedding weight, linear weight) or rank 1 (linear bias, layer-norm gamma and beta). `LinearConfig` defaults `bias: true`. `LayerNormConfig` defaults `bias: true`. Expected tree:

| Piece | Float tensors |
|---|---|
| `type_embed.weight` plus 7 `field_embed.{i}.weight` | 8 |
| Each encoder layer: 4 attention linears and 2 feed-forward linears, weight and bias, plus `norm_1` and `norm_2` gamma and beta | 16 |
| Six layers | 96 |
| `pointer_hidden`, `pointer_out`, `value_hidden`, `value_out`, weight and bias | 8 |
| Total | 112 |

Element sum is 4,987,394 (`docs/ASTRA7-NET.md`). Example paths: `field_embed.0.weight`, `encoder.layers.0.mha.query.weight`, `encoder.layers.0.mha.query.bias`, `encoder.layers.0.norm_1.gamma`, `encoder.layers.0.pwff.linear_inner.bias`, `pointer_out.bias`, `value_out.weight`.

```rust
pub struct ParamSlot {
    pub path: String,
    pub id: ParamId,
    pub rank: u8,
    pub numel: usize,
    pub dims: Vec<usize>,
}

pub struct PairedSlot {
    pub path: String,
    pub id0: ParamId,
    pub id1: ParamId,
    pub rank: u8,
    pub dims: Vec<usize>,
}

pub fn float_param_slots<B: Backend>(net: &Net<B>) -> Vec<ParamSlot>;
pub fn pair_slots(a: &[ParamSlot], b: &[ParamSlot]) -> Result<Vec<PairedSlot>, PairError>;
```

`pair_slots` fails closed on: different lengths, path mismatch at an index, duplicate path, rank or dim mismatch, rank other than 1 or 2, element sum other than 4,987,394, slot count other than 112. Zip by visit order and require path equality. Do not sort paths before the equality check. A sort would hide a swapped visit. Do not pair by iterating a `HashMap` of ids. Do not skip a pair because `id0 == id1`.

Build the `Vec<PairedSlot>` once, after the initial fork, and reuse it for every step. After the first broadcast, and again on debug builds every step, re-read ids and assert they still match the table. Do not compare tensor bytes there (that syncs).

Initial replica: `fork_bc_replica(&net, &devices[1])` materializes all primary weights via `float_param_slots` before `net.clone().fork(&devices[1])`, then pairs the initialized slots. This is tensor initialization and metadata inspection, with no host read of weight data. `fork` is the right constructor because it copies weights and gives the replica its own autodiff graph (`Module::fork` docs, `burn-core-0.21.0/src/module/base.rs` lines 97-103). `float_detach` builds a new `AutodiffTensor` (`burn-autodiff-0.21.0/src/ops/tensor.rs` lines 1859-1871). Do not `Net::fresh` the replica. That would draw a second initialization. Do not use a bare `clone()` as the replica.

Broadcast after every Adam step, including the step that is about to checkpoint or shut down:

- Walk device 0's live weights with a `ModuleMapper` that pushes the same `enter_module` stack the visitor used: field name, struct type, `Vec` index strings, no `"Some"`. `map_float` does not receive a path unless the mapper pushes that stack itself.
- Zip that order against `PairedSlot`. On a path mismatch, `eprintln` the two paths and `process::exit(1)`. Do not iterate `container.ids()`.
- For each paired path, `to_device(devices[1]).detach()` and `require_grad()` if the source required grad. Same sequence as `Param::fork` (`burn-core-0.21.0/src/module/param/tensor.rs` lines 306-316).
- `Param::from_mapped_value` keeps the replica's `ParamId`.
- Do not replace the replica with `net.clone().fork(...)` each step. Re-fork does not mint ids and does not drop the replica's ids. It is unnecessary. The id alias with device 0 already exists from the initial fork. Rebuilding would also be extra copies. The pairing table stays valid because the paths, and the ids captured after the initial fork, do not change.

Int and bool parameters: this net has none. If `visit_int` or `visit_bool` fires, `pair_slots` errors. Do not ignore them.

### Heartbeat

`HEARTBEAT_EVERY` stays 50 Adam steps. The printed line is the current format string from `run_bc` (`ml7/src/main.rs` lines 544-547). Paste it. Do not replace the tail with `...`:

```
    heartbeat: batch {}/{}  sampled_policy={:.4} (mean={:.4})  sampled_value={:.4} (mean={:.4})  lr={:.2e}  {:.2}s elapsed  ({:.3}s/batch)
```

`tools/dashboard_server.py` lines 358-364 require `lr=...`, then `([\d.]+)s elapsed`, then `\(([\d.]+)s/batch\)`. The elapsed fields are part of the shape. `nb` and the denominator (`total_batches` today, the window count on two devices) count Adam steps, not micro-batches.

Only the two sampled numbers change on the two-device path. `sampled_policy` and `sampled_value` (`last_pl`, `last_vl`) become the sequence-weighted means of the two per-device means:

```
(n0 * s0 + n1 * s1) / (n0 + n1)
```

With equal `n` that is the arithmetic mean of the two scalars, which equals the mean over the global batch because each scalar is already a mean. The cumulative `mean=` is still the mean of heartbeat samples since epoch start (`tpl / metric_batches`), not a mean over every step. `lr`, the elapsed seconds, and the seconds-per-batch suffix stay. Do not print per-device losses on this line. Do not change the dashboard parser.

On one device the two sampled numbers are still the single device's scalars, read before `backward`. The epoch-end Iter line keeps using those same accumulators for its sampled policy and sampled value fields. It is not the heartbeat line, and it does not replace the elapsed suffix on the heartbeat.

### Correctness check (`bc-parity`)

Add subcommand `bc-parity` in PR 3, so the first binary that accepts `--devices 2` also contains the check. It is not a training run and it does not write a checkpoint. The compare function lives in `ml7/src/net.rs` next to `BatchLayout::dense`. `BatchLayout` and `dense()` are private. The `bc-parity` arm in `main.rs` calls that function. It does not name `BatchLayout`.

Conditions, all required:

- Dropout 0. Add `Net::fresh_with_dropout(device, dropout)`. `Net::fresh` calls it with `0.1`. Parity calls it with `0.0`. No CLI flag to turn training dropout off.
- One shared dense layout (`BatchLayout::dense()`) for both halves and for the reference forward. Compact per-slice layouts are out of this comparison.
- Equal halves. Micro-batch `B = 2` is enough. Global reference batch is `4`. Use `recorder::benchmark_samples` or the same deterministic CPU fixture the compact tests use, duplicated out to 4 sequences if needed.
- Seed with `B::seed(device, seed)` before `fresh_with_dropout`. `Backend::seed` is `seed(&Device, u64)`. There is no one-argument seed.
- Same initial weights, three graphs. Build `net` on `Cuda(0)`. The replica and reference both use `fork_bc_replica`, on `Cuda(1)` and `Cuda(0)` respectively, so the primary weights are initialized once before either `clone().fork`. Neither is a bare clone. `AutodiffTensor` is `#[derive(Clone)]` and the clone copies `NodeRef` (`burn-autodiff-0.21.0/src/tensor.rs` lines 13-17). `Param::clone` copies the cached tensor only after initialization (`param/base.rs`). Before initialization it copies the lazy initializer into an independent state and preserves the ID, so matching IDs alone do not prove matching weights. A shared `NodeRef` aliases the autodiff client with device 0. `Module::fork` detaches into a new graph. Shared weight storage on the same device is acceptable for a grad-only compare: `TchOps::to_device` returns the same storage when the device matches. A shared `NodeRef` is not.
- No `optim.step` on any of the three modules. Compare grads only.
- Reference: one `bc_backward_layout` on all `2B` sequences, on the forked reference module.
- Two-device side: the same overlapped scoped-thread pair used in training, then `weighted_average` with counts `B, B`. No host read before both threads join. The reference backward and the pair run on three graphs.

For every paired tensor, let `d = g_avg - g_ref` on device 0. Fail if

- `max_abs(d) > 1e-4`, or
- `l2(d) / max(l2(g_ref), 1e-8) > 1e-4`.

Print the worst path, both norms, and the max abs. Exit 1 on failure. Exit 0 on success. Do not loosen `1e-4` to paper over a shared-`NodeRef` reference.

Device count. `tch::Cuda::device_count()` returns `i64`. Cast with `usize::try_from` only when a `usize` is required. Compare the `i64` directly for the skip:

- tch build and `tch::Cuda::device_count() >= 2`: run the check.
- tch build and fewer than 2 devices, including a negative count: print `bc-parity: skipped, visible CUDA devices = {n}` (the `i64`) and exit 0. A skip is not a pass. Exit 0 plus the word `skipped` means do not train. The runbook and `bc-parity` help say that.
- Non-tch build: print `bc-parity requires --features tch` to stderr and exit 2. Do not link this arm to `burn-tch`.

`#[cfg(all(test, feature = "tch"))]` test calls the same function and returns immediately when `tch::Cuda::device_count() < 2`. Without the `tch` feature the test is not compiled. Hermes does not pass `--features tch`.

### CPU tests (no CUDA)

In `ml7/src/net.rs` tests, backend `Autodiff<NdArray>` / `NdArray`, device `NdArrayDevice::Cpu`:

- `weighted_average` on two rank-1 tensors: equal weights match the arithmetic mean; weights 1 and 3 match `(1*a + 3*b) / 4`; length mismatch and a zero count panic or return the error type, they do not produce a silent short tensor.
- `float_param_slots` on `Net::fresh`: 112 slots, element sum 4,987,394, paths unique, ranks only 1 or 2.
- Two `Net::fresh` calls: `pair_slots` succeeds, paths match, `ParamId`s differ.
- `clone().fork` onto the same CPU device: `pair_slots` succeeds and `ParamId`s match. This locks the burn 0.21 fact. The average must not skip the equal-id case.
- Two rank-matched grads with the same `ParamId`, stored in two `GradientsParams`, still produce `(n0 * g0 + n1 * g1) / (n0 + n1)` through the remove-both, average, register-only-device-0 helper. A single-map `register` of the second id replaces the first, and that map's length stays 1. This is the bug a merge-then-`len() == 112` check would miss.
- A unit test on hand-built `ParamSlot` lists: swapped paths, a dropped tail slot, and a dim mismatch all return `PairError`.
- Fork a freshly created BC model through the production constructor before any other parameter access and require identical starting weights in the primary, replica, and reference.
- Run the same dense-layout, dropout-free `bc_parity` calculation on CPU with two fork-disjoint micro-batches and the unchanged `1e-4` absolute and relative L2 thresholds.

`resolve_bc_devices` tests live next to that function (a `#[cfg(test)]` module on the binary is fine): requested 1 with visible `Some(2)` returns 1; requested 2 with visible `Some(1)` errors and the message contains `Refusing`; requested 2 with visible `Some(2)` returns 2; requested 3 errors; requested 2 with `None` errors.

Cutter tests, pure CPU, no tensors, next to the window builder. They are part of PR 3:

- Device count 1 delegates to today's `length_grouped` result. With `batch_size = 4`, a trailing chunk of length 3 is kept and a chunk of length 1 is dropped. The cosine divisor is `batch_size * devices`, which is `batch_size`. `global_step` still increments once per kept chunk. The test does not need to run Adam to lock the chunk list and the divisor.
- Device count 2 yields `n / (2 * batch_size)` windows and `dropped_tail = n % (2 * batch_size)`. A tail of exactly `batch_size` is dropped rather than run. A sorted window is strided: even indices to device 0, odd indices to device 1, so the shorter spans are not all on one card. The cosine divisor is `batch_size * 2`.

### Failure behavior

OOM and device loss. 256-per-card is allowed to fail. Do not catch the failure and retry at 128, and do not fall back to one GPU. libtorch surfaces CUDA errors as a Rust panic through tch, often on the next synchronizing call (`f_copy_data` or `Cuda::synchronize`), not on the launch that faulted. A 256-per-card OOM inside device 1's backward can surface at the heartbeat `download_f32` or the grad `tensor.to`, after the host slot has already moved to `grad-average` and both device slots say `returned`. That names the synchronizing call, not the faulting kernel. It is not a precise fault address.

Install a panic hook only for the `--devices 2` run. Three atomics, not one:

- cuda:0 slot, written only by thread 0: `idle`, `forward-backward`, `returned`.
- cuda:1 slot, written only by thread 1: the same three values.
- host slot, written only by the host after both joins: `idle` during the overlap, then `grad-average`, `adam`, `broadcast`.

The hook prints both device slots and the host slot:

```
bc: fatal during two-device step, cuda:0 {phase0}, cuda:1 {phase1}, host {host_phase}. No fallback to one GPU and no batch shrink. 256-per-card may OOM a 16 GB T4; rerun with --batch-size 128.
```

`std::thread::scope` joins the sibling before the scope's panic reaches the host, so the hook can run twice: once on the panicking worker, at the fault, and again if that panic unwinds on the host. Both prints show both device slots. A panic on device 0 cannot be relabeled as device 1 because thread 1 never writes the cuda:0 slot. The second print may show the sibling as `returned` if it finished. That is still both devices.

Then the previous hook runs. Do not `catch_unwind` and continue: the CUDA context is not usable, and a rebuilt replica would not match Adam's moments. If libtorch aborts the process instead of panicking, the last device-0 checkpoint (at most 300 Adam steps old) is the recovery point. That is the same bound as today. Checkpoints never read the replica. Do not add a watchdog.

A missing grad, a pairing failure, or a broadcast that cannot find a path is a normal `eprintln` plus `process::exit(1)` before `bc_apply_step`, so device 0 is not half-updated. The previous checkpoint stays valid.

Replica desync. The broadcast is in the same function as the step, not a best-effort branch. The next window does not start if broadcast returned an error. On resume, the replica is forked from the loaded device-0 weights before step 1, so a killed job cannot come back with a stale device 1.

Single-GPU fallback. The only success path for `--devices 2` is `resolve_bc_devices` returning 2. There is no `unwrap_or(1)`.

## API and Interface Changes

CLI, `bc` only:

| Flag | Default | Meaning |
|---|---|---|
| `--devices` | 1 | `1` one device, today's behavior, including heartbeat `download_f32` before `backward`. `2` two CUDA devices, tch only. |
| `--batch-size` | 64 | Unchanged. Per-device micro-batch, not global. |

New subcommand `bc-parity`, described above, landed in the same PR as `--devices 2`. Help text for `bc` and for `bc-parity` documents the flag and the check. The help text must include the 128 and 256 commands, the unscaled-lr note, the resume cosine footgun, and the sentence that exit 0 plus `skipped` means do not train. No other subcommand's help or defaults change. Do not add a new markdown file for those commands.

`run_bc`'s device argument becomes a slice. One-device callers, including Vulkan and ndarray, pass `&[device]`.

`Net::fresh_with_dropout` is crate-visible. Training still calls `fresh`, which is dropout 0.1.

No change to `make_optimizer`, `VALUE_LOSS_WEIGHT`, `LOGIT_MASK`, or the public one-device return type of `bc_train_step`: `(Net<B>, Option<f32>, Option<f32>)`.

## Data Model Changes

No schema change. `BinFileRecorder<FullPrecisionSettings>` stays. File size stays about 19,952,515 bytes. The sidecar stays three lines: `epoch`, `global_step`, `best_acc`. `global_step` is Adam updates under either device count. Adam moments stay out of the file. A resume still rebuilds Adam on device 0 and copies weights to device 1 before the first step.

Migration: none. Old checkpoints load on the new binary. A one-device resume of a two-device checkpoint is the same weight file; only the noise scale of the run that produced it differed.

## Alternatives Considered

### Burn `DistributedBackend` / NCCL DDP

The trait is the library's real multi-device API, including a backward path that allreduces grads (`AutodiffTensor::backward` under `feature = "distributed"` calls `sync_collective`). It is the right long-term tool on a backend that implements it. `burn-tch` 0.21 does not. Implementing it, and pulling NCCL into the Kaggle image, is a larger change than a 20 MB copy. T4x2 on this notebook has no NVLink, so NCCL would still cross PCIe. Rejected for this spec. Not rejected forever.

### One T4 at batch 256, no second card

Same global batch as 128+128. No new code. The A4000 measurement (13.5 GB at AZ batch 256, same trunk and heads) says 256 on one 16 GB T4 is the tight but plausible single-card point, and 128 is comfortable. This uses one card. The T4x2 quota hour still makes both cards available, and the idle card is the thing this spec exists to avoid. If a Kaggle run shows the two micro-steps do not overlap, stop using `--devices 2` and run one card at `--batch-size 256` instead. That fallback needs none of these patches. Be explicit with operators: the two-card path is justified only by both cards working inside the same quota hour, and only while the micro-steps overlap.

### One host thread, sequential launch

Rejected as the implementation of overlap. One thread can enqueue device 1 only after it has finished walking device 0's tape. Device 1 is idle for that whole launch. A contiguous short/long split makes the imbalance worse. Strict alternation on `nvidia-smi` can then happen with no forbidden `into_data` at all, so the runbook cannot treat alternation as proof of a host read. Two scoped threads in one process, one Adam, joined before any read, are the design. They still do not overlap if the two modules share an autodiff graph, which is why the replica is `fork` and not `clone`.

### Two processes

Two notebooks, or two processes pinned to `cuda:0` and `cuda:1`, would each run the current binary. They would also keep two Adams, two checkpoints, and two different weight trajectories. That is not data parallel. It burns the quota on two models. Rejected. A second host thread is not this alternative.

### Gradient accumulation on one GPU

Two sequential micros of 128 on one card, average, one Adam step: same noise scale as 128+128, no second card, and the micros cannot overlap. Useful only as a way to get a large global batch when a single forward OOMs. Out of scope. It is not a substitute for `--devices 2`.

### fp16 on the T4s

Locked out. The Vulkan fp16 epsilon failure is why the live trainer is fp32. This spec does not reopen dtype.

## Security and Privacy Considerations

This is a local training loop, not a service. No new socket, no new credential, no new remote write. Checkpoints remain local weight files plus the three-line sidecar. They do not gain Adam moments, game logs, or the other replica's tensors.

The threat that matters here is operational, not disclosure:

- Silent use of a second GPU changes the global batch and the number of Adam steps. Mitigation: default `--devices 1`, startup line, hard exit when 2 was requested and fewer than 2 CUDA devices exist.
- A panic hook that continues after OOM could write a checkpoint that does not match the replica. Mitigation: the hook prints both device slots and the host slot, then aborts. No `catch_unwind`. Saves happen only after a finished Adam step and only from device 0.
- `bc-parity` and `--devices` must not grow a path that shells out to `nvidia-smi` with unsanitized arguments. Device identity is `LibTorchDevice` debug output and `tch::Cuda::device_count()` only.

Kaggle notebook data stays in the notebook. This design adds no upload step.

## Observability

No new metrics system and no dashboard change.

- Startup line: device count, indices, per-device batch, global batch. Two-device runs also print the unscaled-lr note.
- Once per epoch: sequence count, `adam_steps` (Adam steps, floor division), `dropped_tail` (sequences, not steps).
- Heartbeat: the exact `main.rs` lines 544-547 format, every 50 Adam steps, including `{:.2}s elapsed` and `({:.3}s/batch)`. `batch i/n` is Adam steps. Only the two sampled numbers become the weighted means on two devices.
- Fatal line: `cuda:0 {phase0}, cuda:1 {phase1}, host {host_phase}`, plus the no-fallback sentence. One slot per device, so the other thread cannot overwrite the name. The host slot can still be the next synchronizing call rather than the faulting kernel.
- `bc-parity`: either per-tensor success, the worst mismatch, or `bc-parity: skipped, visible CUDA devices = N`.

Overlap is not a CI metric. The runbook tells the operator to watch both T4s during a step (utilization on both cards between heartbeats). A step where only one card moves, or where the cards strictly alternate for whole micro-steps, means either a host sync or a host-bound launch. It is not, by itself, proof that `into_data` ran. Sequential launch from one thread can alternate with no forbidden read, because device 1 is idle until the host enters the second call. The design forbids that launch shape: two scoped threads, joined before any read. If only one card moves, or the cards alternate for whole micro-steps, stop the run. One T4 at `--batch-size 256` is the fallback if overlap cannot be shown. That is the same global batch, no new code, and it idles a card the quota already paid for.

## Rollout Plan

The flag is the feature flag. Default 1 is the rollback: rebuild or just omit `--devices` and the process is today's one-device BC, including the pre-backward heartbeat read. Vulkan rollback is "do not pass `--devices 2`", which already errors, and do not change the Hestia command. Do not point the Hestia `script` command at the PR 2 binary unless a Vulkan heartbeat step has been shown to match the old wrapper.

Staged merge is the PR plan below. PR 2 rebases onto PR 1. They both edit `ml7/src/net.rs` and are merged in order, not as parallel merges. PRs 1-3 merge and test on Hermes with no NVIDIA GPU. PR 3 is the first binary that can start `--devices 2`, and it includes `bc-parity`. PR 4 is the Kaggle execution of that binary: `bc-parity` must pass, then the 128-per-card command. Do not start a 15-epoch job until parity has passed on that notebook and a short run shows both cards busy. 256-per-card is a separate manual attempt after 128 is stable, not the first run. Exit 0 plus `skipped` means do not train.

Resume during rollout uses the existing sidecar. Switching device count on resume does not migrate Adam moments (there are none) and does shift the cosine horizon, as specified above. Prefer a fresh Kaggle run over resuming a Hestia `global_step` into `--devices 2` unless that horizon shift is accepted.

Rollback of a bad two-device checkpoint is the same as any bad BC checkpoint: start again from an earlier bin, or from `Net::fresh`. There is no second file to delete.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Silent single-GPU fallback after `--devices 2`, or silent dual-GPU when the flag is absent | High | `resolve_bc_devices` tested on CPU. Default 1 does not query or use `Cuda(1)`. Startup line prints count and global batch. Exit 1 before generation if 2 was asked and fewer than 2 CUDA devices exist. `device_count` is the `i64` from `tch::Cuda`, cast with `usize::try_from`. |
| Host sync, or a single host thread, so device 1 starts only after device 0's launch has finished. That is slower than one T4 at global batch 256 and wastes the quota. A contiguous short/long split loads the cards unequally even when both do run. | High | Two `std::thread::scope` threads on fork-disjoint graphs. Join before `download_f32` / `into_data` / `f_copy_data` / `sync` / `tensor.to` / `bc_apply_step`. Stride each window, even indices to device 0 and odd to device 1. Runbook: alternation means a host sync or a host-bound launch, not proof of `into_data`. Fallback is one T4 at batch 256. |
| Parameter order mismatch, equal `ParamId`s skipped, or both grads inserted into one map, so Adam updates nothing or only one side while `len() == 112` | High | Path zip plus shape plus 112 slots plus 4,987,394 elements. `remove` both containers by the shared id, `register` only device 0. `None` aborts with the path before `step`. CPU test covers the same-id two-container average and the single-map replace. Ids are not the key, and equal ids are not a skip. |
| Dropout makes a naive parity test fail, and someone "fixes" it by weakening `1e-4` or disabling dropout in training. A bare-clone reference aliases device 0's `NodeRef` and fails for the same bad reason. | Medium | Parity constructs `fresh_with_dropout(..., 0.0)` and `clone().fork` for the reference and the replica, three graphs. Training `fresh` stays 0.1. Compare function lives in `net.rs` next to `BatchLayout::dense`. Tolerance stays both `1e-4` checks. |
| Arc default batch or Vulkan behavior changes because the two-device loop shares `run_bc`, or because the one-device heartbeat read moves across `backward` | High | `devices.len() == 1` calls the one-device wrapper: `download_f32`, then `backward`, then `step`. It does not use the deferred `.inner()` path. The cutter CPU test locks `length_grouped` for device count 1, including a trailing chunk of length 3 at batch 4 and a dropped chunk of length 1, and locks the cosine divisor at `batch_size`. Vulkan never forks. Default `--batch-size` stays 64. PR 2 must not be the Hestia restart unless a Vulkan heartbeat step matches this order. `NdArray` tests cannot see a Fusion queue change. |
| LR looks "wrong" because global batch grew from 64 to 256 or 512 and the noise scale changed | Medium | Do not scale LR. Startup prints the note when `devices == 2`. `3e-4` at global batch 64 on the Arc is the historical point, not the 2-GPU point. Help text repeats the note. |
| Remainder logic changes updates per epoch, so cosine and checkpoint cadence surprise the operator, or `dropped_tail` is implemented as a step delta | Medium | One device keeps partial chunks of length >= 2. Two devices drop the tail and log `adam_steps` as `floor(n / global_batch)` and `dropped_tail` as a sequence count. A nonzero remainder removes one Adam step and that many sequences. `checkpoint_every` stays 300 Adam steps. Cosine denominator uses `batch_size * devices`. The cutter test locks both policies. |
| Resume of a large one-GPU `global_step` onto `--devices 2` jumps the cosine | Medium | Runbook and help text: prefer a fresh Kaggle start or `--model`. Not a new schedule. Single-device estimate is unchanged. |
| Replica desync if broadcast is skipped, or if the replica is rebuilt every step instead of mapped | High | Broadcast is part of the step. The mapper pushes the same `enter_module` stack and exits 1 on a path mismatch. Re-fork each step is unnecessary, not id-invalidating, and is not the broadcast. Checkpoints read device 0 only. Resume forks once before step 1. |
| 256-per-card OOM hangs, silently retries, or the panic line is read as the faulting kernel. One shared phase atomic also names whichever thread stored last. | Medium | Allowed to fail. The hook prints both device slots and the host slot, then aborts. No `catch_unwind`, no batch shrink, no one-GPU fallback. A fault inside backward often surfaces on the next sync: both device slots say `returned` and the host slot says `grad-average`. That is the synchronizing call, not the kernel. First Kaggle run is 128. Recovery is the last device-0 checkpoint, at most 300 Adam steps old. |
| Heartbeat `.inner()` or an extra `mean()` pins the autodiff graph and OOMs only on heartbeat steps | Medium | Non-heartbeat steps pass `report_metrics = false` and keep no scalar node. The two-device heartbeat uses `.inner()`, not `detach()`, and downloads only after the join. The one-device heartbeat downloads and drops the metric tensors before `backward`. If a heartbeat step OOMs at 256 while a normal step fits, that is still a fatal, not a retry. |
| `LibTorch::device_count` (always 1) used by mistake, so `--devices 2` always errors, or the `i64` is truncated | Medium | Call `tch::Cuda::device_count` only, and cast with `usize::try_from`. Comment cites `burn-tch` lines 174-177 and `device.rs` lines 20-23. The CPU test injects the visible count; it does not call the backend method. |
| Cross-device `to()` on the legacy default stream joins the two cards | Low for overlap, if it happens after both launches | The join is the average, which is after both backwards are queued and both threads have joined. Do not move the copy into the first backward. 19.0 MiB on PCIe is milliseconds next to a transformer step. Do not add NCCL to "fix" it. |
| Scoped threads serialize anyway because the two modules share a graph mutex, or because raw `tch::Tensor` was sent across threads | High | Replica and parity reference are `clone().fork`. Threads hold `&Net` / `&replica` of `TchTensor`, which is `Send + Sync`. Do not split a raw `tch::Tensor` (`Send` only). The runbook overlap check is the backstop if libtorch still host-locks every launch. |

## Open Questions

Locked, and not open: LR scaling, fp16 or bf16, Vulkan behavior, Arc batch size, whether to use both GPUs, NCCL, a second Adam, per-device clip, remainder on one device, dashboard work, and moving the one-device heartbeat read to after `backward`.

One API detail could not be verified from Rust sources, and the design assumes a conservative reading of it.

`tch-0.22.0`'s `Tensor::f_copy_data` (`src/wrappers/tensor.rs` lines 396-417 and `src/tensor/convert.rs` lines 7-20) writes into a host slice via `at_copy_data`. The Rust wrapper does not call `tch::Cuda::synchronize` itself. The sync is inside libtorch, and this checkout does not contain that C++ body. The trainer already treats scalar readback as a device sync (`ml7/src/net.rs` lines 346-347). Assumption used everywhere above: `download_f32`, `Tensor::into_data`, `Vec<T>::try_from` on a CUDA tensor, and `f_copy_data` are host syncs and are forbidden until both `bc_backward_layout` threads have joined. A host-bound launch that is not one of those calls is also forbidden before the join: one thread must not finish device 0 before device 1 is started. `TchOps::to_device` (`tensor.to(device)`) is not a host read and is the allowed copy, only after the join. If an implementer finds that `tensor.to(other_cuda)` itself blocks the host until both devices are idle, that still does not break the design as long as the `to` stays after both backwards are queued. It must not be inserted between them. Alternation on `nvidia-smi` does not prove which of these happened. No other unverified choice blocks implementation.

## References

- `ml7/src/main.rs`: feature `compile_error`, `Gpu` alias, `cmd_bc`, `run_bc`, `length_grouped` (lines 65-70), `steps_per_epoch_est` (lines 466-468), heartbeat format string (lines 544-547), checkpoint poll (lines 550-577), `ResumeMeta` / `save_resume_meta` (lines 141-163), `install_shutdown_flag` (lines 237-247), `LibTorchDevice::Cuda(0)` entry points, AZ 13.5 GB comment (lines 1557-1561).
- `ml7/src/net.rs`: `bc_train_step`, `bc_train_step_layout` (download before backward, lines 346-364), `download_f32` (lines 55-57), `make_optimizer`, `LOGIT_MASK`, `Net::fresh` dropout 0.1, scalar-readback comment, `td_train_step` detach leak comment (lines 373-376). `BatchLayout::dense` is private; the parity compare sits next to it.
- `ml7/src/tokens.rs`: `SEQ_LEN` 180, `NUM_FIELDS` 7.
- `ml7/src/recorder.rs`: `ExpertCpuPlayer` teacher. `ml7/src/batch_infer.rs`: `default_concurrency` `(2 * cores).max(8)`.
- `ml7/Cargo.toml`: `tch` and `vulkan` features, `fusion` on `burn-wgpu`, burn 0.21. `ml7/build.rs`: `LIBTORCH` and `-ltorch_cuda`.
- `tools/dashboard_server.py` lines 358-364: heartbeat regex requires the elapsed suffix.
- `docs/ASTRA7-NET.md`: 4,987,394 parameters, 19.0 MiB, dropout 0.1, Adam epsilon `1e-5`, clip described as global-norm 1.0.
- `ASTRA7.md`: Atlas GTX 1050 Ti, fp32 batch 64 at 3531 MiB.
- `STATE.md`: checkpoint 19,952,515 bytes, Hestia command, 200 games in 7.3 s, resume sidecar, Arc batch 64.
- burn 0.21.0 local sources: `burn-tch` `Backend::sync`, `device_count` hardcoded to 1, `Backend::seed(&Device, u64)` lines 125-126, `TchTensor` `Send + Sync` lines 215-216, `TchOps::to_device`, `float_into_data`; `burn-optim` `GradientsParams` (`register` replaces), `AdamConfig`, `GradientClipping::clip_by_norm`, `SimpleOptimizerMapper` lines 178-234, `GradientsParamsConverter` lines 38-40; `burn-derive` `gen_visit` / `gen_map`; `burn-core` `Module::fork` lines 97-103, `Param::map` lines 294-306, `Param::fork` lines 306-316, `Vec` index push lines 106-125; `burn-nn` `Dropout`; `burn-backend` `DistributedBackend` and `TensorContainer::remove` lines 79-87; `burn-wgpu` fusion `Wgpu` alias lines 63-64; `burn-autodiff` `AutodiffTensor` clone lines 13-17, non-distributed `backward` lines 185-190, `GraphMutexClient` `graph.rs` lines 32-35 and 89-127, `float_detach` lines 1859-1871, `Tensor::inner` in `burn-tensor` autodiff.rs lines 57-61.
- tch 0.22.0: `Cuda::device_count() -> i64` lines 20-23, `Cuda::synchronize`, `Tensor` is `Send` only (wrappers/tensor.rs line 22), `Tensor::f_copy_data` lines 396-417.

## PR Plan

Four PRs, merged in order. PR 1 and PR 2 both edit `ml7/src/net.rs`. Develop them separately if useful, then rebase PR 2 onto PR 1 and merge in that order. Do not merge them as two parallel edits. PRs 1-3 are reviewable on Hermes with no NVIDIA GPU. PR 4 is the only one that needs a real two-GPU run. 256-per-card is a later manual attempt, not part of merging PR 4.

### PR 1: CPU-tested gradient average and parameter pairing

- Title: `ml7: pair BC parameters by module path and average grads on CPU`
- Files: `ml7/src/net.rs` (slots, `pair_slots`, `weighted_average`, the two-container remove/register helper, tests). No change to `run_bc`, `bc_train_step`, or CLI.
- Dependencies: none.
- Description: Add the path collector, the fail-closed pairing (112 tensors, 4,987,394 elements, rank and dim checks), and `weighted_average`. Equal `ParamId`s are two grads in two containers: `remove` both, average, `register` only one container. Tests use `NdArray` only: equal and unequal weights, swapped paths, dropped slot, dim mismatch, two `fresh` nets with different ids, `clone().fork` id stability, the same-`ParamId` two-container average, and a single-map `register` that replaces the first id. `cargo test -p novacana-ml7` on Hermes without `tch` passes. Training behavior is unchanged.

### PR 2: Split backward from the Adam step, keep the one-device read order

- Title: `ml7: split bc_train_step into forward, backward, and apply`
- Files: `ml7/src/net.rs` (`BcForward`, `BcBackward`, `bc_forward_layout`, `bc_backward_layout`, `bc_apply_step`, `Net::fresh_with_dropout`). `bc_train_step` / `bc_train_step_layout` become the one-device wrapper. `benchmark` stays on that wrapper.
- Dependencies: rebase onto PR 1. Merged after PR 1, not in parallel with it. PR 3 needs both.
- Description: `bc_forward_layout` builds the loss and the optional metric tensors and does not download or step. The one-device wrapper calls `download_f32` on those metric tensors, then `backward`, then `bc_apply_step`. That is today's order. It does not move the heartbeat read to after `step`. `bc_backward_layout` is the two-device helper: `.inner()` the metric tensors, then `backward`, no `download_f32`. Nothing calls it until PR 3. `bc_apply_step` is `optim.step` only, so the per-tensor `Norm(1.0)` stays inside the single step. The public return type stays `(Net, Option<f32>, Option<f32>)`. `fresh` still uses dropout 0.1. Behavior-identical for this PR means weights, `global_step`, checkpoint bytes, and heartbeat counters. The startup line is not in this PR. This PR must not be what the Hestia `script` command restarts onto unless a Vulkan heartbeat step has been shown to match the old wrapper. Hermes `NdArray` tests cannot see a Fusion queue change. Existing compact gradient tests pass.

### PR 3: `--devices`, the overlapped step, the cutter, and `bc-parity`

- Title: `ml7: opt-in two-device BC step`
- Files: `ml7/src/main.rs` (`--devices`, `resolve_bc_devices`, `i64` cast of `tch::Cuda::device_count`, `run_bc` device slice, strided windows, startup line, epoch `dropped_tail`, panic hook with one phase atomic per device plus a host slot, `bc-parity` arm), `ml7/src/net.rs` (average-by-pairs with rank-selected `remove`, broadcast mapper that pushes `enter_module` and exits 1 on a path mismatch, `bc-parity` compare next to `BatchLayout::dense`, cfg-gated test). Help text can land here in stub form; PR 4 is what makes the runbook sentences mandatory.
- Dependencies: PR 1 and PR 2, already merged in that order.
- Description: Default `--devices 1` keeps `length_grouped`, `Cuda(0)` or the Vulkan device, batch default 64, the one-device wrapper (download, then backward, then step), and the existing heartbeat format. `--devices 2` on a tch build requires two CUDA devices, forks one replica, strides each window, runs both `bc_backward_layout` calls on `std::thread::scope` threads, joins before any host read or `tensor.to` or `bc_apply_step`, averages with sequence weights, steps once, broadcasts, and counts Adam steps for `global_step`, heartbeat, and `checkpoint_every`. `bc-parity` runs the dropout-0, dense-layout, equal-half grad compare on three forked graphs at tolerance `1e-4` absolute and `1e-4` relative L2, seeded with `B::seed(device, seed)`. Fewer than 2 CUDA devices skips with exit 0 and the `skipped` line. Non-tch exits 2. The `#[cfg(all(test, feature = "tch"))]` test no-ops below 2 devices. CPU tests cover `resolve_bc_devices` and the cutter: device count 1 matches `length_grouped` (length 3 kept at batch 4, length 1 dropped) and cosine divisor `batch_size`; device count 2 yields `n / (2 * batch_size)` windows, `dropped_tail = n % (2 * batch_size)`, drops a tail of exactly `batch_size`, and strides even/odd. Non-tch `--devices 2` errors. No change to `td-value`, `az`, `bench`, `winrate`, `behavior`, `fidelity`, `value-rank`, or `token-scan`. Hermes still builds and tests without `tch`.

### PR 4: Kaggle runbook in help text, and the notebook run

- Title: `ml7: T4x2 runbook in bc help`
- Files: `bc` and `bc-parity` help text, or the module comment next to those arms. Do not add a new markdown file. Do not put the only copy of the commands in this design doc. `STATE.md` only if the implementer is already editing it for the command lines.
- Dependencies: PR 3. The parity subcommand is already in that binary.
- Description: This PR does not invent a second copy of the training loop. It is the notebook run plus the help text. The help text must include the comfortable command (`--devices 2 --batch-size 128`), the optional command (`--batch-size 256`, may OOM, not the first run), the unscaled-lr note (`3e-4` was tuned at global batch 64 on the Arc; this run's global batch is 256 or 512), the resume cosine footgun (a large one-GPU `global_step` resumed onto `--devices 2` sits further along a shorter cosine; prefer a fresh start or `--model`), and the sentence that exit 0 plus `skipped` means do not train. Also state that cards which strictly alternate for whole micro-steps are not this design: that is a host sync or a host-bound launch, and the fallback is one T4 at `--batch-size 256`. Build on the notebook with `--features tch`. Do not copy a Hestia Vulkan binary. `--gen-concurrency 4` is runbook only. Do not start 15 epochs until `bc-parity` passes on that notebook and a short run shows both cards busy. This is the PR reviewed against a real two-GPU run.
