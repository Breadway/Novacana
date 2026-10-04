# Astra-7 on the brother's PC - install guide

Target machine: Windows, RTX 3080 (10GB VRAM), i7-10700K (8C/16T), 16GB RAM.

This turns that PC into a second GPU training node for Astra-7 (alongside
Atlas, the existing GTX 1050 Ti box) - self-play generation on CPU, BC
training on GPU via WSL2 + CUDA passthrough, pausable/resumable so it
doesn't have to monopolize a machine that isn't yours.

Everything referenced below lives in this `brothers-pc/` folder.

## What you need before starting

- Physical/remote access to the PC with an admin account
- ~10GB free disk (LibTorch alone is ~7GB unpacked; WSL2/Ubuntu itself is
  a few GB)
- `novacana-source.tar.xz` (already built, sitting in this folder) copied
  onto that machine somehow - USB stick is simplest for the first sync
  since Tailscale isn't joined there yet. Anything after that can go over
  Tailscale (see step 5).

## Step 1 - Windows: WSL2 + Ubuntu

1. Copy this whole `brothers-pc/` folder onto the PC (USB stick, or a
   Windows file share, whatever's easiest).
2. Right-click PowerShell → **Run as administrator**, `cd` into the copied
   `brothers-pc\windows` folder, then:
   ```
   powershell -ExecutionPolicy Bypass -File setup-wsl2.ps1
   ```
3. This checks the NVIDIA driver, then installs WSL2 + Ubuntu 22.04. If
   WSL wasn't already present on the machine, **it'll ask for a reboot** -
   do that, then Ubuntu finishes its own first-run setup (asks you to pick
   a Linux username/password - anything is fine, it's local to WSL2).
4. Copy `windows\dot-wslconfig` to `C:\Users\<their-username>\.wslconfig`
   (yes, the leading dot - easiest from a terminal:
   `copy dot-wslconfig %USERPROFILE%\.wslconfig`). This caps WSL2 at 12GB
   RAM / 6 cores so Windows itself doesn't get starved - see the comments
   in that file for why those particular numbers.
5. From any PowerShell: `wsl --shutdown`, then reopen **Ubuntu** from the
   Start menu to apply it.

## Step 2 - WSL2/Ubuntu: build everything

Inside the Ubuntu window:

1. Get `novacana-source.tar.xz` into the Ubuntu filesystem. From Ubuntu,
   Windows drives are at `/mnt/c/...`, so if the USB stick or folder is at
   e.g. `D:\brothers-pc`, that's `/mnt/d/brothers-pc` in Ubuntu - copy the
   whole `brothers-pc` folder into your Linux home directory first (WSL2
   performance on the Windows-drive mount is much worse than the Linux
   filesystem, and the build step especially will be painfully slow
   otherwise):
   ```
   cp -r /mnt/d/brothers-pc ~/
   cd ~/brothers-pc/linux
   ```
2. Run the bootstrap:
   ```
   bash bootstrap.sh
   ```
   This installs build tools + Rust, downloads LibTorch 2.9.0+cu126
   (~3.2GB, the slow step - matches the version already proven working on
   Atlas), extracts the source tarball to `~/novacana-engine`, runs the
   plain test suite as a sanity check, then builds the real
   `--features tch` (CUDA) release binary. Safe to re-run if anything
   fails partway - every step skips work that's already done.
3. Watch for the GPU visibility check at the end - it should print the
   RTX 3080's name and VRAM. If `nvidia-smi` isn't found inside WSL2, the
   Windows-side driver likely needs updating (back to step 1.3) or WSL2's
   GPU passthrough needs a kick (`wsl --shutdown` from Windows, reopen
   Ubuntu).

## Step 3 - start training

From Windows, double-click **`windows\start-training.bat`**. This:
- launches training inside a `tmux` session in WSL2 (survives closing the
  Ubuntu window - it only stops if you explicitly stop it or the PC
  reboots/sleeps)
- if run again while training is already going, attaches you to the live
  output instead of starting a second copy
- auto-resumes from the last checkpoint if one exists

To pause (e.g. your brother needs the PC back), double-click
**`windows\stop-training.bat`**. This sends a graceful stop signal - the
training process finishes its current batch, saves a checkpoint, and
exits (typically under 2 seconds, verified). Run `start-training.bat`
again whenever it's free and it picks up exactly where it left off.

**Do not** just close the Ubuntu window or shut down the PC as your normal
way of pausing - that skips the graceful-save and can lose up to a few
minutes of progress (still bounded, not catastrophic, but stop-training.bat
is strictly better and just as easy).

The exact launch command and why each number was chosen (batch size for
the 10GB card, chunk-games for the 16GB RAM budget, etc.) is documented in
`linux/launch-bc.sh`'s comments.

## Step 4 - getting checkpoints back

Nothing rsyncs automatically. Checkpoints land at
`~/novacana-engine/models/astra7_bc.bin` inside WSL2. Once Tailscale is
joined (step 5), pull them back to hermes with:
```
rsync -avz brothersPC:novacana-engine/models/ ~/Projects/novacana/models/
```
(adjust the Tailscale hostname/IP to whatever it actually registers as -
check `tailscale status` on either end). Do this after any meaningful
training milestone - **this is the lesson from the RunPod incident**: a
checkpoint that only exists on rented/borrowed compute is one bad day away
from gone. See STATE.md's "SESSION PAUSED 2026-07-10" section for exactly
what that cost.

## Step 5 - Tailscale (for remote monitoring + future syncs)

1. Inside Ubuntu (WSL2 shares the Windows network, but Tailscale needs to
   run somewhere with a stable presence - WSL2 itself is fine for this):
   ```
   bash ~/novacana-engine/brothers-pc/linux/join-tailscale.sh
   ```
   This installs Tailscale (if needed) and joins non-interactively using a
   pre-generated auth key - no browser login required. It registers as
   `brotherspc` on the tailnet (matches the dashboard's default
   `ASTRA7_SSH_HOST`) and prints its tailnet IP at the end.

   The auth key is reusable but expires ~2026-07-23 (7 days from
   generation) - if this script is run after that, generate a fresh key at
   https://login.tailscale.com/admin/settings/keys and swap it into
   `join-tailscale.sh`. Revoke the key from that same page once joined
   successfully and no further (re)joins are expected.

   (If `join-tailscale.sh` isn't present for some reason, fall back to
   manual: `curl -fsSL https://tailscale.com/install.sh | sh && sudo
   tailscale up` - this prints a login URL to open in a browser instead.)
2. Once joined, it'll show up in `tailscale status` on hermes/hestia too.
   From there:
   - `ssh` in directly to check on things without needing physical/RDP access
   - pull checkpoints with the rsync command in step 4
   - re-sync newer code any time with:
     ```
     rsync -avz --exclude target hermes:novacana-engine/ml7 ~/novacana-engine/ml7
     ```
     (run from inside WSL2, replace `hermes` with whatever hostname/IP
     `tailscale status` shows - hermes's tailnet IP is `100.75.233.34` as
     of this writing) then rebuild:
     `cd ~/novacana-engine && cargo build --release -p novacana-ml7 --features tch`

Note: WSL2's own Tailscale identity is separate from the Windows host's -
if Windows already has Tailscale installed for something else, that's a
different node than this one. Not a problem, just don't conflate the two
when looking at `tailscale status`.

## Step 6 - the dashboard (both of you can watch)

The dashboard (`tools/dashboard_server.py`) runs directly on this machine
now, in "local mode" - it reads the training log/GPU/checkpoint files
straight off disk instead of over SSH, and serves the same page to
whoever hits it:

1. Double-click **`windows\start-dashboard.bat`** - runs it in its own
   tmux session (`astra7-dash`, independent of the training session, so
   it can run whether or not training is currently active).
2. **Your brother** watches locally: `http://localhost:8420` in a browser
   on his machine (WSL2 forwards this to Windows automatically).
3. **You** watch remotely, once Step 5's Tailscale join is done: find this
   machine's tailnet IP (`tailscale ip -4` inside Ubuntu, or `tailscale
   status` from hermes/hestia - it registers as `brotherspc`), then open
   `http://<that-ip>:8420` from anywhere on the tailnet. No SSH tunnel,
   no port forwarding on his router - Tailscale runs inside this same
   WSL2 instance, so its interface can reach the dashboard directly.
4. Stop it any time with **`windows\stop-dashboard.bat`** - it has no
   state to save (read-only), safe to stop/start independent of training.

If the tailnet URL doesn't load on the first try (WSL2 + Tailscale
networking can be finicky depending on WSL version), fall back to an SSH
tunnel from hermes instead: `ssh -L 8420:localhost:8420 brotherspc`, then
open `http://localhost:8420` locally on your end.

## Files in this folder

| File | Runs where | Purpose |
|---|---|---|
| `windows/setup-wsl2.ps1` | Windows, elevated, once | Installs WSL2 + Ubuntu, checks NVIDIA driver |
| `windows/dot-wslconfig` | copied to `%USERPROFILE%\.wslconfig` | Caps WSL2 RAM/CPU so Windows isn't starved |
| `windows/start-training.bat` | Windows, double-click | Start/resume/attach to training |
| `windows/stop-training.bat` | Windows, double-click | Graceful pause (saves a checkpoint first) |
| `windows/start-dashboard.bat` | Windows, double-click | Start/attach the local-mode dashboard (Step 6) |
| `windows/stop-dashboard.bat` | Windows, double-click | Stop the dashboard (stateless, safe any time) |
| `linux/bootstrap.sh` | WSL2/Ubuntu, once | Rust, LibTorch, build, sanity test |
| `linux/join-tailscale.sh` | WSL2/Ubuntu, once | Non-interactive Tailscale join (has a time-limited auth key baked in - see Step 5, don't commit this file to git) |
| `linux/launch-bc.sh` | WSL2/Ubuntu, via start-training.bat | The actual tuned training command |
| `package-source.sh` | hermes, whenever code changes | Rebuilds `novacana-source.tar.xz` |
| `novacana-source.tar.xz` | - | The source snapshot to transfer over |

## If something breaks

- **"LIBTORCH env var must be set"** during build → `source ~/.bashrc` (or
  open a new Ubuntu window) so bootstrap.sh's env-var export takes effect.
- **CUDA OOM** during training → edit `linux/launch-bc.sh`, drop
  `--batch-size` from 128 to 96, re-run.
- **Machine feels sluggish / Windows unusable while training runs** → the
  `.wslconfig` caps are probably too generous for how the PC actually gets
  used - lower `memory=`/`processors=` in `%USERPROFILE%\.wslconfig`, then
  `wsl --shutdown` and restart training.
- **`nvidia-smi` not found inside WSL2** → Windows-side NVIDIA driver
  needs updating (do NOT install a Linux NVIDIA driver inside WSL2 itself
  - that actively breaks the passthrough, it should only ever come from
  Windows).
- Anything training-logic-related (not setup-related) → check
  `~/novacana-engine/training.log` (written by start-training.bat) or the
  `ASTRA7.md` / `STATE.md` docs synced alongside the source.
