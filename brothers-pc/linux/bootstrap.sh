#!/usr/bin/env bash
# Astra-7 brother's-PC bootstrap, step 2 of 2 (WSL2/Ubuntu side).
#
# Run this INSIDE Ubuntu-on-WSL2 (open "Ubuntu" from the Start menu), after
# setup-wsl2.ps1 + the .wslconfig have been done on the Windows side.
#
# What it does:
#   1. installs build deps + tmux (training runs inside tmux so it survives
#      you closing the Ubuntu window)
#   2. installs Rust via rustup
#   3. downloads + unpacks LibTorch 2.9.0+cu126 to ~/libtorch (matched to
#      this repo's pinned torch-sys 0.22.0, see ml7/Cargo.toml)
#   4. wires LIBTORCH/LD_LIBRARY_PATH into ~/.bashrc
#   5. extracts the source tarball (expects novacana-source.tar.xz to sit
#      next to this script - see the README for how to get it here)
#   6. runs the plain (non-GPU) test suite as a cheap sanity check
#   7. builds the real release binary with --features tch (CUDA)
#   8. runs nvidia-smi + a 1-line CUDA visibility check so you know before
#      training starts whether the GPU is actually reachable
#
# Safe to re-run - every step checks whether its work is already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HOME/novacana-engine"
LIBTORCH_DIR="$HOME/libtorch"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip"

section() { echo; echo "=== $1 ==="; }

section "System packages"
if ! dpkg -s build-essential >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential pkg-config libssl-dev curl git unzip tmux htop
else
    echo "already installed, skipping"
fi

section "Rust"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup default stable
echo "$(rustc --version)"

section "LibTorch 2.9.0+cu126"
if [ ! -d "$LIBTORCH_DIR/lib" ]; then
    echo "Downloading LibTorch (~3.2GB, this is the slow step)..."
    curl -L -o /tmp/libtorch.zip "$LIBTORCH_URL"
    unzip -q /tmp/libtorch.zip -d "$HOME"
    rm /tmp/libtorch.zip
else
    echo "already present at $LIBTORCH_DIR, skipping"
fi

section "Environment (LIBTORCH / LD_LIBRARY_PATH)"
MARKER="# astra7-libtorch"
if ! grep -q "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo "$MARKER"
        echo "export LIBTORCH=\"$LIBTORCH_DIR\""
        echo "export LD_LIBRARY_PATH=\"\$LIBTORCH/lib:\${LD_LIBRARY_PATH:-}\""
    } >> "$HOME/.bashrc"
    echo "added to ~/.bashrc"
else
    echo "already present in ~/.bashrc, skipping"
fi
export LIBTORCH="$LIBTORCH_DIR"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

section "Source tarball"
if [ ! -d "$REPO_DIR" ]; then
    # novacana-source.tar.xz lives at brothers-pc/novacana-source.tar.xz -
    # one level up from this script (brothers-pc/linux/bootstrap.sh) -
    # check there first, then a couple of other plausible drop locations.
    TARBALL="$SCRIPT_DIR/../novacana-source.tar.xz"
    if [ ! -f "$TARBALL" ]; then
        TARBALL="$HOME/novacana-source.tar.xz"
    fi
    if [ ! -f "$TARBALL" ]; then
        TARBALL="$HOME/brothers-pc/novacana-source.tar.xz"
    fi
    if [ ! -f "$TARBALL" ]; then
        echo "ERROR: novacana-source.tar.xz not found next to this script or in \$HOME." >&2
        echo "See the README for how to get it onto this machine (USB copy, or" >&2
        echo "the rsync-pull command once Tailscale is joined)." >&2
        exit 1
    fi
    mkdir -p "$REPO_DIR"
    tar -xJf "$TARBALL" -C "$REPO_DIR"
    echo "extracted $TARBALL -> $REPO_DIR"
else
    echo "$REPO_DIR already exists, skipping extraction (re-sync with the rsync"
    echo "command in the README if you want to pull in newer code)"
fi
cd "$REPO_DIR"

section "Sanity check: plain test suite (CPU, no GPU needed)"
cargo test -p novacana-ml7 --release 2>&1 | tail -30

section "Building release binary with CUDA (--features tch)"
cargo build --release -p novacana-ml7 --features tch

section "GPU visibility check"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader || {
    echo "nvidia-smi not found inside WSL2 - this usually means the Windows-side"
    echo "NVIDIA driver needs updating (see setup-wsl2.ps1's driver check), or"
    echo "the WSL2 GPU passthrough kernel component isn't active yet (try a"
    echo "'wsl --shutdown' from Windows PowerShell, then reopen Ubuntu)."
}

section "Done"
echo "Binary: $REPO_DIR/target/release/novacana-ml7"
echo "Launch training with: bash $SCRIPT_DIR/launch-bc.sh"
echo "(or let the Windows start-training.bat shortcut do it for you)"
