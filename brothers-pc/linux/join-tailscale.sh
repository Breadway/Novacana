#!/usr/bin/env bash
# One-shot Tailscale join for the brother's PC (run inside WSL2/Ubuntu).
# Replaces the interactive browser-login flow in README Step 5 with a
# single command using a pre-generated auth key.
#
# NOT committed to git (this whole brothers-pc/ folder is untracked, keep
# it that way while this file has a live key in it). The key below is a
# REUSABLE key with a ~7-day expiry (set when generated 2026-07-16) - if
# setup gets redone after it expires, generate a fresh one at
# https://login.tailscale.com/admin/settings/keys and swap it in below.
# Revoke it from that same page once the brother's PC has successfully
# joined and you don't expect to need it again.

set -euo pipefail

AUTH_KEY="tskey-auth-kwwqeQjPJ521CNTRL-fE6DWF1xDBWSAHJqCKCmAWh51JQM8LQFc"
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
