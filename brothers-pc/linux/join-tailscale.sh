#!/usr/bin/env bash
# One-shot Tailscale join for the brother's PC (run inside WSL2/Ubuntu).
# Replaces the interactive browser-login flow in README Step 5 with a
# single command using a short-lived auth key.
#
# The key is never stored here. Export TAILSCALE_AUTH_KEY, or pass it as the
# first argument. Generate a short-lived, single-use key at
# https://login.tailscale.com/admin/settings/keys and revoke it once the
# brother's PC has joined.

set -euo pipefail

AUTH_KEY="${TAILSCALE_AUTH_KEY:-${1:-}}"
if [ -z "${AUTH_KEY}" ]; then
    echo "Set TAILSCALE_AUTH_KEY or pass the auth key as the first argument." >&2
    exit 1
fi
HOSTNAME="brotherspc"

if ! command -v tailscale >/dev/null 2>&1; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi

sudo tailscale up --authkey="$AUTH_KEY" --hostname="$HOSTNAME"

echo
echo "Joined as '$HOSTNAME'. Verify from hermes/hestia with: tailscale status"
echo "Its tailnet IP (needed for ASTRA7_SSH_HOST / rsync in later steps):"
tailscale ip -4
