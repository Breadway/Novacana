#!/usr/bin/env bash
# Astra-7 BC training, tuned for the brother's PC (RTX 3080 10GB, 16GB
# system RAM, i7-10700K). Meant to be run inside tmux (start-training.bat
# does this for you) so it survives the Ubuntu window being closed.
#
# Auto-resumes: if models/astra7_bc.bin + its .resume.txt sidecar already
# exist (e.g. from a previous run that was paused via stop-training.bat),
# this picks up where it left off instead of starting over.
#
# --- Tuning rationale (2026-07-15) ---
#
# --batch-size 128: the A4000 (16GB) measured a stable 13.5GB peak VRAM at
# batch-size 256 (STATE.md run-history row 3); batch-512 OOM'd even on a
# 24GB 4090. Memory scales roughly linearly with batch size for this
# architecture (cross-checked against an old 4GB-card data point at
# batch-64 -> 3.5GB - same ~1:1 ratio). Halving to 128 should land around
# ~6.75GB, leaving comfortable headroom under the 3080's 10GB, including
# some slack for WDDM/Windows-side VRAM reservation that a headless Linux
# pod doesn't have to account for. If `nvidia-smi` (run it in a second
# Ubuntu window while training) shows several GB of consistent headroom,
# it's safe to try bumping this to 160-192; drop to 96 immediately if you
# see a CUDA OOM.
#
# --chunk-games 1500: this is the RAM-sizing knob, not a VRAM one - it
# controls how many fresh self-play games get generated and held in
# memory before each epoch's training pass (ASTRA7.md §8.3's "never
# pregenerate more" guard, originally sized for hermes/pod-scale RAM).
# With only 16GB total system RAM shared with Windows itself, this is
# sized conservatively; the .wslconfig caps WSL2 at 12GB. If you watch
# Task Manager during a run and see plenty of headroom, this can go up -
# but every doubling roughly doubles the pregenerated buffer's RAM use, so
# increase gradually (e.g. try 2500 next, not 8000).
#
# --checkpoint-every 200: on a shared, sometimes-paused machine, this
# bounds how much training you can lose to an ungraceful kill (e.g. a
# Windows update forcing a reboot) to ~200 batches - a few minutes at
# most. The *graceful* path (stop-training.bat) saves immediately
# regardless of this number.
#
# Everything else (lr, label-smoothing, players, winrate-every/games)
# matches the values already validated in real runs - see STATE.md's
# "Run history" table.

set -euo pipefail

cd "$HOME/novacana-engine"
export LIBTORCH="$HOME/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

OUT=models/astra7_bc.bin
RESUME_FLAG=()
if [ -f "$OUT" ] && [ -f "${OUT%.bin}.resume.txt" ]; then
    echo "Found existing checkpoint + resume metadata - continuing that run."
    RESUME_FLAG=(--resume "$OUT")
fi

exec ./target/release/novacana-ml7 bc \
    "${RESUME_FLAG[@]}" \
    --out "$OUT" \
    --chunk-games 1500 \
    --epochs 20 \
    --batch-size 128 \
    --val-games 200 \
    --players 4 \
    --lr 0.0003 \
    --label-smoothing 0.05 \
    --winrate-every 1 \
    --winrate-games 150 \
    --checkpoint-every 200
