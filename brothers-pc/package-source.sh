#!/usr/bin/env bash
# Run this ON HERMES (not on the brother's PC) whenever the ml7 code
# changes and needs to reach the brother's PC again. Produces
# novacana-source.tar.xz in this directory - small (~1MB, source only) and
# self-contained enough to satisfy the Cargo workspace (engine/client/
# server/ml/ml7 must all exist for `members = [...]` in the root
# Cargo.toml to resolve, even though only ml7 actually gets built there).
#
# Deliberately excluded: ml/data (1.2GB legacy v5/v6 pretraining corpus -
# astra7/ml7 generates its own self-play data, no static corpus
# dependency), ml/models + top-level models/ (old, architecture-
# incompatible checkpoints - astra7 always starts from Net::fresh() or its
# own checkpoints), target/ dirs (build artifacts, rebuilt on the other
# end), games/ + logs/ (autosave/log noise), src.tar.xz/src.zip (stale
# pre-ml7 snapshots), the rest of tools/ (only dashboard_server.py is
# needed there, to run in ASTRA7_LOCAL=1 mode - see README step 6).
#
# Transfer this file to the brother's PC via USB/network share for the
# first sync (Tailscale isn't joined there yet), or use the rsync command
# in the README for subsequent syncs once it is.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="brothers-pc/novacana-source.tar.xz"

tar -cJf "$OUT" \
    --exclude='target' \
    --exclude='__pycache__' \
    --exclude='*.bin' \
    --exclude='*.bin.bak' \
    Cargo.toml Cargo.lock .gitignore ASTRA7.md STATE.md \
    engine client server \
    ml/Cargo.toml ml/build.rs ml/src \
    ml7 \
    tools/dashboard_server.py \
    brothers-pc/windows brothers-pc/linux

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
