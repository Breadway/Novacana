# GPU Rental Requirements — Astra-7 Training

Shopping list for renting a cloud GPU to replace/supplement Atlas (GTX 1050
Ti, 4GB VRAM — too small for the batch sizes this model wants). Background
and full reasoning: `ASTRA7.md` §10.

## GPU spec

| Requirement | Why |
|---|---|
| **≥12GB VRAM** | Hard minimum. The model itself is tiny (~6-8M params, ~120MB), but attention activations at batch-512 need ~3GB — batch-512 OOMs Atlas's 4GB card outright. 12GB+ removes the batch-size ceiling entirely. |
| **Tensor cores (Turing/Ampere/Ada or newer)** | Atlas's Pascal card has none — FP16 runs at 1/64 the FP32 rate there, mixed precision is actively harmful. Any modern consumer/datacenter card from ~2018+ has them. |
| **Do NOT get A100/H100** | Overkill and 3-5x the price for zero benefit — this model can't use that much compute or memory. Explicitly not worth it. |

**Target: RTX 4090 (24GB)** — the sweet spot for price/availability/headroom.
A 4070/L4/A10-class card (12-16GB) would also fully saturate on compute for
a model this size if a 4090 isn't available or costs more; pick by price and
VRAM, not by chasing the biggest number.

## CPU / RAM

Game generation (Rust, rayon-parallel, embarrassingly parallel) runs
co-located on the same box — no need for a separate CPU machine. Whatever
vCPU count the GPU instance bundles (typically 8-32 on these platforms) is
plenty; this isn't the bottleneck. 16GB+ system RAM is comfortable, not
tight.

## OS / software

- **Ubuntu 22.04 or 24.04 LTS** (or whatever the provider's default
  CUDA-preinstalled template uses — see below).
- **Prefer a template with NVIDIA driver + CUDA toolkit already installed.**
  Most GPU rental platforms offer these; picking one skips the driver
  install step that took real effort to get right on Atlas.
- Everything else (Rust toolchain, LibTorch, the actual `ml7` build) I'll
  set up myself once I have access — no need to pre-install anything beyond
  the driver/CUDA template.

## Providers (pricing as of 2026-07, re-check — moves weekly)

| Provider | RTX 4090 | Notes |
|---|---|---|
| [RunPod](https://www.runpod.io/pricing) | ~$0.34-0.69/hr | Simpler, more predictable — Community Cloud cheaper, Secure Cloud more reliable |
| [Vast.ai](https://vast.ai/pricing) | ~$0.09-0.59/hr (spot) | Cheapest, but spot instances can be reclaimed with ~15s notice — fine for BC/TD-value, riskier for a long unattended AZ run |
| Lambda Labs | — | Didn't quote 4090 directly; better for A100/H100 tier, not needed here |

**Recommendation**: RunPod on-demand for simplicity, or Vast.ai if you want
to shop for the cheapest instantaneous rate and don't mind occasional
interruption (checkpointing already exists — best-checkpoint saves survive
a restart).

## Budget expectation

**~$20-50 total for the entire remaining campaign** (scaled BC run →
TD-value stage → AZ league), not per-stage. This is cheap specifically
because the model is small — the real cost driver is wall-clock hours, not
GPU tier. Renting continuously for a multi-day stretch is fine and probably
simpler than spinning the instance up/down repeatedly.

## What I need from you once it's up

1. **SSH access** — IP/hostname, port, and either a key file or root
   password (whatever the provider gives you). Most platforms show this
   directly on the instance's dashboard.
2. That's it — I'll handle the rest (Rust, LibTorch version-matching, repo
   sync, build, training launch, monitoring).

## Optional: skip the shopping if you want

If picking a provider/instance feels like a rabbit hole, RunPod's
on-demand RTX 4090 at their published rate is a safe, simple default — spin
one up, hand me the SSH details, done. The cost difference vs. hunting for
the cheapest Vast.ai spot instance is a few dollars over the whole project,
not worth much optimization time.
