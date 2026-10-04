#!/usr/bin/env python3
"""Live training-progress "mission control" dashboard for astra7 training runs.
A background thread polls the training node over SSH on a fixed cadence and
caches the result; the browser gets updates pushed to it over Server-Sent
Events the instant that cache refreshes - no client-side polling delay (see
"real-time" note below). stdlib only.

Repointing at a different node (2026-07-15 - moved off the RunPod pod, onto
the brother's PC): set ASTRA7_SSH_HOST / ASTRA7_LOG_PATH, or just edit the
defaults below. No RunPod-specific cost tracking anymore - this is borrowed,
not rented, compute - the dashboard shows session duration instead.

Local mode (2026-07-16): set ASTRA7_LOCAL=1 to run this directly on the
training box itself (e.g. inside WSL2 on the brother's PC) instead of over
SSH from a separate machine - probe commands run via a local shell instead
of `ssh`, and the HTTP server binds 0.0.0.0 instead of 127.0.0.1 so it's
reachable both from the Windows host (WSL2's default localhost forwarding)
and from anyone else on the tailnet at this machine's Tailscale IP. Lets
the brother watch on his own screen (http://localhost:PORT from Windows)
at the same time you watch remotely (http://<its-tailscale-ip>:PORT) -
both hit the same local server, no double-SSH-hop needed. Override the
bind address directly with ASTRA7_DASHBOARD_HOST if the default per-mode
choice isn't what you want.

Real-time polling (2026-07-15): previously the *browser* polled /api/data
every 6s. Two changes: (1) one SSH round-trip per cycle instead of five
(log/gpu/ps/spectator-game/card-spectrum used to be five separate `ssh`
subprocess invocations - now one script, delimited output, parsed
client^H^H^H server-side) plus SSH multiplexing (ControlMaster) so that one
round-trip reuses a warm connection instead of paying a fresh TCP+SSH
handshake every time; (2) a background thread does that fetch on a tight
POLL_INTERVAL_S cadence and the browser is pushed each result over SSE
(EventSource) the moment it lands, rather than fetching on its own 6s timer.
Net effect: the dashboard reflects the remote log within ~POLL_INTERVAL_S of
it changing, not up to 6s+round-trip-time later, and EventSource's built-in
auto-reconnect means a flaky Tailscale link recovers on its own.
"""
import http.server
import subprocess
import re
import json
import os
import datetime
import socketserver
import threading
import time
import random
import math

PORT = int(os.environ.get("ASTRA7_DASHBOARD_PORT", "8420"))
# SSH alias/hostname for the training node. Once Tailscale is joined on the
# brother's PC (brothers-pc/README.md step 5), add an SSH config Host entry
# for its WSL2 instance's Tailscale IP/hostname and point this at it - or
# just override with ASTRA7_SSH_HOST=whatever at launch, no edit needed.
SSH_HOST = os.environ.get("ASTRA7_SSH_HOST", "brotherspc")
# launch-bc.sh tees training output to ~/novacana-engine/training.log.
LOG_PATH = os.environ.get("ASTRA7_LOG_PATH", "~/novacana-engine/training.log")
# Sibling outputs write_spectator_outputs() (ml7/src/main.rs) produces
# alongside the checkpoint - see spectator.rs's module doc.
SPECTATOR_GAME_PATH = os.environ.get("ASTRA7_SPECTATOR_PATH", "~/novacana-engine/models/spectator_game.json")
CARD_SPECTRUM_PATH = os.environ.get("ASTRA7_SPECTRUM_PATH", "~/novacana-engine/models/card_spectrum.jsonl")
POLL_INTERVAL_S = float(os.environ.get("ASTRA7_POLL_INTERVAL_S", "2"))

LOCAL_MODE = os.environ.get("ASTRA7_LOCAL", "") == "1"
DASHBOARD_HOST = os.environ.get("ASTRA7_DASHBOARD_HOST", "0.0.0.0" if LOCAL_MODE else "127.0.0.1")

# Fake-data demo mode (2026-07-16): set ASTRA7_FAKE=1 to drive the whole page
# off a synthetic, self-advancing training run instead of SSHing anywhere -
# useful for previewing the UI (new panels, palette tweaks, etc.) with no
# training box available. Bypasses get_data()/fetch_remote() entirely.
FAKE_MODE = os.environ.get("ASTRA7_FAKE", "") == "1"

# SSH multiplexing: the first connection in a poll cycle opens the master;
# every subsequent one (this process's whole lifetime, up to 10 min idle)
# rides the same authenticated TCP connection - cuts per-poll SSH latency
# from a fresh handshake to essentially a local socket round-trip. Unused
# in LOCAL_MODE (no ssh subprocess at all).
_CONTROL_PATH = f"/tmp/astra7-dashboard-control-{os.getpid()}-%C"
SSH_OPTS = [
    "-o", "ConnectTimeout=5",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_CONTROL_PATH}",
    "-o", "ControlPersist=10m",
]

_SEP = "\x1e"  # ASCII record separator - won't collide with real log content


def run_probe(cmd: str) -> str:
    """Runs the combined probe script either over SSH (default) or via a
    local shell (LOCAL_MODE) - same command string either way, since both
    ssh and bash tilde-expand `~` and run in a real shell."""
    try:
        if LOCAL_MODE:
            r = subprocess.run(["bash", "-c", cmd],
                                capture_output=True, text=True, timeout=15)
        else:
            r = subprocess.run(["ssh", *SSH_OPTS, SSH_HOST, cmd],
                                capture_output=True, text=True, timeout=15)
        return r.stdout
    except Exception:
        return ""


def fetch_remote():
    """One SSH round-trip for everything the dashboard needs, instead of
    five - see module doc's "real-time polling" note."""
    script = (
        f'echo {_SEP}LOG{_SEP}; cat {LOG_PATH} 2>/dev/null; '
        f'echo {_SEP}GPU{_SEP}; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu '
        f'--format=csv,noheader,nounits 2>/dev/null; '
        f'echo {_SEP}PS{_SEP}; ps aux | grep novacana-ml7 | grep -v grep; '
        f'echo {_SEP}SPECTATOR{_SEP}; cat {SPECTATOR_GAME_PATH} 2>/dev/null; '
        f'echo {_SEP}SPECTRUM{_SEP}; cat {CARD_SPECTRUM_PATH} 2>/dev/null; '
        f'echo {_SEP}END{_SEP}'
    )
    out = run_probe(script)
    parts = out.split(f"{_SEP}LOG{_SEP}\n", 1)
    if len(parts) < 2:
        return "", "", "", "", ""
    rest = parts[1]
    log, rest = rest.split(f"{_SEP}GPU{_SEP}\n", 1) if f"{_SEP}GPU{_SEP}\n" in rest else (rest, "")
    gpu, rest = rest.split(f"{_SEP}PS{_SEP}\n", 1) if f"{_SEP}PS{_SEP}\n" in rest else (rest, "")
    ps, rest = rest.split(f"{_SEP}SPECTATOR{_SEP}\n", 1) if f"{_SEP}SPECTATOR{_SEP}\n" in rest else (rest, "")
    spectator, rest = rest.split(f"{_SEP}SPECTRUM{_SEP}\n", 1) if f"{_SEP}SPECTRUM{_SEP}\n" in rest else (rest, "")
    spectrum = rest.split(f"{_SEP}END{_SEP}", 1)[0] if f"{_SEP}END{_SEP}" in rest else rest
    return log, gpu, ps, spectator, spectrum


# Product GDDR6, in MiB. The A380's PCI BAR is 8 GiB, which is larger than
# the memory the card can actually use, so the BAR size is the wrong total.
_INTEL_GPU = {
    "0x56a5": ("Intel Arc A380", 6144),
}


def _parse_drm_bytes(text: str) -> float:
    parts = text.split()
    n = float(parts[0])
    if len(parts) == 1:
        return n
    scale = {"kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "kb": 1000, "mb": 10 ** 6, "gb": 10 ** 9}
    return n * scale.get(parts[1].lower(), 1)


def _intel_card():
    """(device_id, energy1_input path) for the first Intel DRM card, or (None, None)."""
    drm = "/sys/class/drm"
    try:
        cards = os.listdir(drm)
    except OSError:
        return None, None
    for card in cards:
        dev = os.path.join(drm, card, "device")
        try:
            vendor = open(os.path.join(dev, "vendor")).read().strip().lower()
            device = open(os.path.join(dev, "device")).read().strip().lower()
        except OSError:
            continue
        if vendor != "0x8086":
            continue
        energy = None
        try:
            for mon in os.listdir(os.path.join(dev, "hwmon")):
                path = os.path.join(dev, "hwmon", mon, "energy1_input")
                if os.path.isfile(path):
                    energy = path
                    break
        except OSError:
            pass
        return device, energy
    return None, None


def _drm_clients():
    """One (render_ns, resident_vram_bytes) per i915/xe client this user can see."""
    clients = {}
    try:
        pids = os.listdir("/proc")
    except OSError:
        return clients
    for pid in pids:
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if not target.startswith("/dev/dri/"):
                continue
            try:
                text = open(f"/proc/{pid}/fdinfo/{fd}").read()
            except OSError:
                continue
            driver = None
            cid = render = local = None
            for line in text.splitlines():
                if line.startswith("drm-driver:"):
                    driver = line.split(":", 1)[1].strip()
                elif line.startswith("drm-client-id:"):
                    cid = line.split(":", 1)[1].strip()
                elif line.startswith("drm-engine-render:"):
                    render = int(line.split()[1])
                elif line.startswith("drm-resident-local0:"):
                    local = _parse_drm_bytes(line.split(":", 1)[1].strip())
            if driver not in ("i915", "xe") or cid is None or render is None:
                continue
            prev = clients.get(cid)
            if prev is None or (local or 0) > prev[1]:
                clients[cid] = (render, local or 0.0)
    return clients


def sample_nvidia():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = r.stdout.strip()
    if r.returncode != 0 or not line:
        return None
    parts = [p.strip() for p in line.splitlines()[0].split(",")]
    if len(parts) != 5:
        return None
    try:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), None
    except ValueError:
        return None


# i915 adds engine time when a request retires, not while it runs. A sample
# shorter than one training batch often reads zero. The poller calls this
# about every 2s, and the delta is taken across those calls.
_intel_prev = None


def sample_intel_gpu():
    """Utilization from the render-engine counter, VRAM from fdinfo, power from
    the energy counter. intel_gpu_top aborts on this Arc, so it is not used.
    This card has no temperature sensor in hwmon; power fills that slot."""
    global _intel_prev
    device, energy_path = _intel_card()
    if device is None:
        return None
    name, total_mib = _INTEL_GPU.get(device, (f"Intel {device}", None))
    after = _drm_clients()
    try:
        energy = int(open(energy_path).read()) if energy_path else None
    except OSError:
        energy = None
    now = time.perf_counter()
    util = None
    power = None
    if _intel_prev is not None:
        t1, before, e1 = _intel_prev
        dt = max(now - t1, 1e-6)
        delta_ns = 0
        for cid, (ns, _local) in after.items():
            prev_ns = before.get(cid)
            if prev_ns is not None:
                delta_ns += max(0, ns - prev_ns)
        util = min(100.0, delta_ns / (dt * 1e9) * 100.0)
        if e1 is not None and energy is not None and energy >= e1:
            power = (energy - e1) / 1e6 / dt
    _intel_prev = (now, {cid: ns for cid, (ns, _local) in after.items()}, energy)
    mem = max((local for _ns, local in after.values()), default=0.0)
    mem_mib = mem / (1024 * 1024)
    if total_mib is None:
        total_mib = max(mem_mib, 1.0)
    return (
        name,
        None if util is None else round(util, 1),
        round(mem_mib, 1),
        float(total_mib),
        None,
        None if power is None else round(power, 1),
    )


def get_data():
    log, gpu_line, ps_line, spectator_raw, spectrum_raw = fetch_remote()
    gpu_line = gpu_line.strip()
    ps_line = ps_line.strip()
    is_running = bool(ps_line)

    gpu_name = None
    gpu_util = gpu_mem_used = gpu_mem_total = gpu_temp = gpu_power = None
    # Local mode is the Hestia host. nvidia-smi is empty there; the Arc
    # reports through fdinfo. SSH mode still reads the nvidia-smi line below.
    if LOCAL_MODE:
        sampled = sample_nvidia() or sample_intel_gpu()
        if sampled:
            gpu_name, gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp, gpu_power = sampled
    if gpu_name is None and gpu_line:
        parts = [p.strip() for p in gpu_line.split(",")]
        if len(parts) == 5:
            gpu_name = parts[0]
            gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = (float(x) for x in parts[1:])

    m = re.search(r"Chunked BC training: (\d+) games/epoch x (\d+) players", log)
    games_per_epoch = int(m.group(1)) if m else None
    players = int(m.group(2)) if m else None

    ep_re = re.compile(
        r"Iter (\d+)/(\d+): (?:sampled_)?policy=([\d.]+) (?:sampled_)?value=([\d.]+)\s+"
        r"Eval @ val_acc=([\d.]+) val_ce=([\d.]+)\s+"
        r"\(([\d.]+)s\)(\s+\*best)?"
    )
    epochs = [{
        "epoch": int(mm.group(1)), "total_epochs": int(mm.group(2)),
        "policy": float(mm.group(3)), "value": float(mm.group(4)),
        "val_acc": float(mm.group(5)), "val_ce": float(mm.group(6)),
        "seconds": float(mm.group(7)), "best": bool(mm.group(8)),
    } for mm in ep_re.finditer(log)]

    # winrate@epochN: 0.1267  (n=150, 38.4s)  [gate: ...]
    wr_re = re.compile(r"winrate@epoch(\d+): ([\d.]+)\s+\(n=(\d+), ([\d.]+)s\)")
    winrates = {int(mm.group(1)): float(mm.group(2)) for mm in wr_re.finditer(log)}
    for e in epochs:
        e["win_rate"] = winrates.get(e["epoch"])

    diag_re = re.compile(r"diag@epoch(\d+): (\{.*\})")
    diags = {}
    for mm in diag_re.finditer(log):
        try:
            diags[int(mm.group(1))] = json.loads(mm.group(2))
        except json.JSONDecodeError:
            continue
    for e in epochs:
        if e["epoch"] in diags:
            e["diag"] = diags[e["epoch"]]
    latest_diag = diags.get(epochs[-1]["epoch"]) if epochs else None

    # Scope heartbeats to the current (in-progress) epoch only - the panel
    # is titled "current epoch", so slice the log to after the last epoch
    # summary line rather than showing every epoch's batches concatenated.
    last_ep_match = None
    for mm in ep_re.finditer(log):
        last_ep_match = mm
    current_epoch_log = log[last_ep_match.end():] if last_ep_match else log

    # 2026-07-10: heartbeat format gained a windowed (last-N-batch) policy/
    # value figure alongside the cumulative epoch-so-far one, since the
    # cumulative-only figure is structurally guaranteed to look flat late
    # in a large epoch and read as a false plateau (Fable 5 consult).
    # Falls back to the old cumulative-only format for log lines written
    # before this change (windowed group becomes None -> use cumulative).
    hb_re = re.compile(
        r"heartbeat: batch (\d+)/(\d+)\s+"
        r"(?:sampled_)?policy=([\d.]+)(?: \((?:last\d+|mean)=([\d.]+)\))?\s+"
        r"(?:sampled_)?value=([\d.]+)(?: \((?:last\d+|mean)=([\d.]+)\))?\s+"
        r"lr=([\d.eE+-]+)\s+"
        r"([\d.]+)s elapsed\s+\(([\d.]+)s/batch\)"
    )
    heartbeats = [{
        "batch": int(mm.group(1)), "total_batches": int(mm.group(2)),
        "policy": float(mm.group(3)),
        "policy_recent": float(mm.group(4)) if mm.group(4) else float(mm.group(3)),
        "value": float(mm.group(5)),
        "value_recent": float(mm.group(6)) if mm.group(6) else float(mm.group(5)),
        "lr": float(mm.group(7)), "elapsed": float(mm.group(8)),
        "s_per_batch": float(mm.group(9)),
    } for mm in hb_re.finditer(current_epoch_log)]

    last_panic_idx = log.rfind("panicked")
    last_heartbeat_idx = log.rfind("heartbeat:")
    crashed = last_panic_idx != -1 and last_panic_idx > last_heartbeat_idx and not is_running
    panic_excerpt = None
    if crashed:
        tail = log[last_panic_idx:]
        panic_excerpt = tail.split("\n")[0][:300]

    latest_hb = heartbeats[-1] if heartbeats else None
    total_epochs = epochs[0]["total_epochs"] if epochs else (15 if heartbeats else None)
    current_epoch_num = len(epochs) + (1 if (latest_hb and (not epochs or True)) else 0)
    if total_epochs and current_epoch_num > total_epochs:
        current_epoch_num = total_epochs

    est_seconds_per_epoch = None
    if latest_hb and latest_hb["s_per_batch"] > 0:
        est_seconds_per_epoch = latest_hb["s_per_batch"] * latest_hb["total_batches"]
    elif epochs:
        est_seconds_per_epoch = sum(e["seconds"] for e in epochs) / len(epochs)

    elapsed_total_s = sum(e["seconds"] for e in epochs)
    if latest_hb and len(epochs) < (total_epochs or 999):
        elapsed_total_s += latest_hb["elapsed"]

    eta_s = None
    if est_seconds_per_epoch and total_epochs:
        done_fraction_current = 0.0
        if latest_hb and current_epoch_num == len(epochs) + 1:
            done_fraction_current = latest_hb["batch"] / max(latest_hb["total_batches"], 1)
        remaining = (total_epochs - len(epochs) - done_fraction_current) * est_seconds_per_epoch
        eta_s = max(remaining, 0)

    status = "crashed" if crashed else ("running" if is_running else ("idle" if not heartbeats else "stopped"))

    tail_lines = [l for l in log.strip().split("\n") if l.strip()][-14:]

    # spectator.rs's SpectatorGame, written fresh each winrate-eval cycle -
    # None until the first eval completes.
    spectator_game = None
    if spectator_raw.strip():
        try:
            spectator_game = json.loads(spectator_raw)
        except json.JSONDecodeError:
            pass  # mid-write on the remote end; next poll picks up the completed file

    # card_spectrum.jsonl: one {"epoch": N, "spectrum": [u8;29]} line per
    # eval - aggregate the 29 per-card-type slots into the 5 CardKind
    # categories (Star/Discovery/Flare/Omen/Fracture) for the chart; the
    # slot ranges are CardKind::type_idx's own layout (engine/src/card.rs).
    card_spectrum_log = []
    if spectrum_raw.strip():
        for line in spectrum_raw.strip().split("\n"):
            try:
                row = json.loads(line)
                s = row["spectrum"]
                card_spectrum_log.append({
                    "epoch": row["epoch"],
                    "star": sum(s[0:7]),
                    "discovery": sum(s[7:12]),
                    "flare": sum(s[12:20]),
                    "omen": sum(s[20:22]),
                    "fracture": sum(s[22:29]),
                })
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue  # one malformed line shouldn't sink the whole chart

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "panic_excerpt": panic_excerpt,
        "games_per_epoch": games_per_epoch,
        "players": players,
        "total_epochs": total_epochs,
        "current_epoch": current_epoch_num,
        "epochs": epochs,
        "heartbeats": heartbeats,
        "latest_heartbeat": latest_hb,
        "gpu_name": gpu_name,
        "gpu_util": gpu_util,
        "gpu_mem_used": gpu_mem_used,
        "gpu_mem_total": gpu_mem_total,
        "gpu_temp": gpu_temp,
        "gpu_power_w": gpu_power,
        "elapsed_total_s": elapsed_total_s,
        "eta_s": eta_s,
        "best_val_acc": max((e["val_acc"] for e in epochs), default=None),
        "tail_lines": tail_lines,
        "ssh_host": f"{SSH_HOST} (local)" if LOCAL_MODE else SSH_HOST,
        "spectator_game": spectator_game,
        "card_spectrum_log": card_spectrum_log,
        "diag": latest_diag,
    }


# ── Fake-data demo mode ──────────────────────────────────────────────────
#
# A self-advancing synthetic run: same shape as get_data()'s return value,
# so the front-end can't tell the difference. State lives in _fake and
# mutates a little on every call (from the poller, on POLL_INTERVAL_S) -
# batches climb, an epoch closes out every _FAKE_TOTAL_BATCHES batches, and
# the whole 15-epoch run loops back to epoch 1 once it finishes so a demo
# left open keeps looking alive indefinitely.
_fake_lock = threading.Lock()
_fake = {
    "epoch_num": 0,
    "batch": 0,
    "epochs": [],
    "heartbeats": [],
    "card_spectrum_log": [],
    "spectator_game": None,
}
_FAKE_TOTAL_EPOCHS = 15
_FAKE_TOTAL_BATCHES = 180
_FAKE_GAMES_PER_EPOCH = 6000
_FAKE_PLAYERS = 4
_FAKE_PLAYER_NAMES = ["Vesper", "Kessler", "Marrow", "Orin"]
_FAKE_STAR_NAMES = ["Protostar", "YellowDwarf", "RedGiant", "RedSupergiant", "WhiteDwarf", "RedDwarf", "BrownDwarf"]
_FAKE_ACTIONS = ["plays a Star card", "discovers a new world", "triggers a Flare event", "reads an Omen", "suffers a Fracture", "passes"]


def _fake_spectator_game(epoch):
    n_frames = random.randint(8, 14)
    frames = []
    for i in range(n_frames):
        active = i % _FAKE_PLAYERS
        star_charts = []
        for _p in range(_FAKE_PLAYERS):
            systems = []
            for _s in range(random.randint(0, 3)):
                cls = random.randint(1, 4)
                systems.append({
                    "star_name": random.choice(_FAKE_STAR_NAMES),
                    "class": cls,
                    "discoveries": [1] * random.randint(0, cls),
                })
            star_charts.append({"systems": systems})
        frames.append({
            "player_name": _FAKE_PLAYER_NAMES[active],
            "action": random.choice(_FAKE_ACTIONS),
            "board": {
                "star_charts": star_charts,
                "active_player": active,
                "hand_sizes": [random.randint(2, 7) for _ in range(_FAKE_PLAYERS)],
                "dark_matter_void": random.choice([None, None, None, random.randint(1, 4)]),
                "turn": i + 1,
            },
        })
    return {
        "players": _FAKE_PLAYER_NAMES,
        "frames": frames,
        "winner_idx": random.randrange(_FAKE_PLAYERS),
    }


def fake_get_data():
    with _fake_lock:
        _fake["batch"] += random.randint(4, 11)
        if _fake["batch"] >= _FAKE_TOTAL_BATCHES:
            _fake["batch"] = 0
            if _fake["epoch_num"] >= _FAKE_TOTAL_EPOCHS:
                # full demo run "finished" last cycle -- loop back to epoch 1
                _fake["epoch_num"] = 0
                _fake["epochs"] = []
                _fake["card_spectrum_log"] = []
            ep_num = _fake["epoch_num"] + 1
            frac = ep_num / _FAKE_TOTAL_EPOCHS
            policy = max(0.15, 1.9 * (1 - frac) ** 1.3 + random.uniform(-0.03, 0.03))
            value = max(0.05, 0.9 * (1 - frac) ** 1.5 + random.uniform(-0.02, 0.02))
            val_acc = min(0.97, 0.35 + 0.55 * frac + random.uniform(-0.01, 0.02))
            val_ce = max(0.3, 2.0 * (1 - frac) + random.uniform(-0.05, 0.05))
            best_so_far = max((e["val_acc"] for e in _fake["epochs"]), default=-1)
            _fake["epochs"].append({
                "epoch": ep_num, "total_epochs": _FAKE_TOTAL_EPOCHS,
                "policy": round(policy, 4), "value": round(value, 4),
                "val_acc": round(val_acc, 4), "val_ce": round(val_ce, 4),
                "seconds": round(random.uniform(140, 210), 1),
                "best": val_acc > best_so_far,
                "win_rate": round(min(0.62, 0.05 + 0.5 * frac + random.uniform(-0.03, 0.03)), 4),
            })
            _fake["epoch_num"] = ep_num

            spectrum = [0] * 29
            for _ in range(400):
                spectrum[random.randrange(29)] += 1
            _fake["card_spectrum_log"].append({
                "epoch": ep_num,
                "star": sum(spectrum[0:7]), "discovery": sum(spectrum[7:12]),
                "flare": sum(spectrum[12:20]), "omen": sum(spectrum[20:22]),
                "fracture": sum(spectrum[22:29]),
            })
            _fake["spectator_game"] = _fake_spectator_game(ep_num)
            _fake["heartbeats"] = []

        ep_num = _fake["epoch_num"]
        batch = _fake["batch"]
        frac_cur = (ep_num + batch / _FAKE_TOTAL_BATCHES) / _FAKE_TOTAL_EPOCHS
        policy_now = max(0.15, 1.9 * (1 - frac_cur) ** 1.3 + random.uniform(-0.05, 0.05))
        value_now = max(0.05, 0.9 * (1 - frac_cur) ** 1.5 + random.uniform(-0.03, 0.03))
        hb = {
            "batch": batch, "total_batches": _FAKE_TOTAL_BATCHES,
            "policy": round(policy_now, 4),
            "policy_recent": round(max(0.1, policy_now + random.uniform(-0.03, 0.03)), 4),
            "value": round(value_now, 4),
            "value_recent": round(max(0.03, value_now + random.uniform(-0.02, 0.02)), 4),
            "lr": 3e-4 * (0.94 ** ep_num),
            "elapsed": round(batch * 0.85, 1),
            "s_per_batch": round(random.uniform(0.7, 1.0), 2),
        }
        _fake["heartbeats"].append(hb)
        if len(_fake["heartbeats"]) > 200:
            _fake["heartbeats"] = _fake["heartbeats"][-200:]

        t = time.time()
        gpu_util = max(0, min(100, 78 + 12 * math.sin(t / 7) + random.uniform(-3, 3)))
        gpu_mem_used = 9800 + 400 * math.sin(t / 11)
        gpu_temp = 61 + 6 * math.sin(t / 13)

        elapsed_total_s = sum(e["seconds"] for e in _fake["epochs"]) + hb["elapsed"]
        current_epoch_num = min(ep_num + 1, _FAKE_TOTAL_EPOCHS)
        eta_s = max((_FAKE_TOTAL_EPOCHS - ep_num - batch / _FAKE_TOTAL_BATCHES) * 175, 0)

        tail_lines = [
            f"heartbeat: batch {hb['batch']}/{hb['total_batches']} policy={hb['policy']:.4f} "
            f"(last32={hb['policy_recent']:.4f}) value={hb['value']:.4f} (last32={hb['value_recent']:.4f}) "
            f"lr={hb['lr']:.2e} {hb['elapsed']:.1f}s elapsed ({hb['s_per_batch']:.2f}s/batch)",
            "[FAKE DATA] synthetic feed for UI preview -- no real training node connected",
        ]

        return {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "running",
            "panic_excerpt": None,
            "games_per_epoch": _FAKE_GAMES_PER_EPOCH,
            "players": _FAKE_PLAYERS,
            "total_epochs": _FAKE_TOTAL_EPOCHS,
            "current_epoch": current_epoch_num,
            "epochs": list(_fake["epochs"]),
            "heartbeats": list(_fake["heartbeats"]),
            "latest_heartbeat": hb,
            "gpu_name": "Fake RTX 4090 (demo)",
            "gpu_util": round(gpu_util, 1),
            "gpu_mem_used": round(gpu_mem_used, 1),
            "gpu_mem_total": 24576.0,
            "gpu_temp": round(gpu_temp, 1),
            "gpu_power_w": None,
            "elapsed_total_s": elapsed_total_s,
            "eta_s": eta_s,
            "best_val_acc": max((e["val_acc"] for e in _fake["epochs"]), default=None),
            "tail_lines": tail_lines,
            "ssh_host": "fake-data (demo mode)",
            "spectator_game": _fake["spectator_game"],
            "card_spectrum_log": list(_fake["card_spectrum_log"]),
            "diag": None,
        }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Astra-7 · Mission Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #05070D;
  --surface: #0E1420;
  --surface-2: #141C2C;
  --border: #1E2740;
  --border-bright: #2C3B5C;
  --text: #E8ECF4;
  --text-dim: #8B95AC;
  --text-faint: #4A5474;
  --accent: #5EEAD4;
  --accent-2: #A78BFA;
  --good: #34D399;
  --warn: #FBBF24;
  --crit: #F87171;
  --accent-glow: rgba(94, 234, 212, 0.35);
  --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-ui);
  -webkit-font-smoothing: antialiased;
}
#starfield { position: fixed; inset: 0; z-index: 0; display: block; }
body {
  position: relative;
  padding: 28px 20px 60px;
  max-width: 1180px;
  margin: 0 auto;
}
main { position: relative; z-index: 1; }
/* faint scanline texture over the whole console - mission-control CRT cue */
main::before {
  content: "";
  position: fixed; inset: 0; z-index: 2; pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px);
}
a { color: var(--accent); }

.top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 22px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border);
}
.eyebrow {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin: 0 0 6px;
}
h1 {
  margin: 0;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 0.01em;
  text-wrap: balance;
  text-shadow: 0 0 24px var(--accent-glow);
}
.subtitle { color: var(--text-dim); font-size: 13px; margin-top: 6px; font-family: var(--font-mono); }

.top-right { display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 16px 7px 12px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid var(--border-bright);
  background: var(--surface);
  white-space: nowrap;
}
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot.running { background: var(--good); box-shadow: 0 0 10px var(--good); animation: pulse 1.8s ease-in-out infinite; }
.dot.crashed { background: var(--crit); box-shadow: 0 0 10px var(--crit); animation: pulse 0.9s ease-in-out infinite; }
.dot.idle, .dot.stopped { background: var(--text-faint); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.met { text-align: right; }
.met .met-label { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.14em; color: var(--text-faint); text-transform: uppercase; }
.met .met-clock {
  font-family: var(--font-mono);
  font-size: 22px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.05em;
  color: var(--accent);
  text-shadow: 0 0 12px var(--accent-glow);
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
.stat {
  position: relative;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 14px 16px;
}
/* HUD corner brackets - reticle cue instead of a plain box */
.stat::before, .stat::after {
  content: "";
  position: absolute; width: 8px; height: 8px;
  border: 1px solid var(--border-bright);
}
.stat::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
.stat::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }
.stat .label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 6px;
}
.stat .value {
  font-family: var(--font-mono);
  font-size: 20px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}
.stat .value.accent { color: var(--accent); text-shadow: 0 0 10px var(--accent-glow); }
.stat .sub { font-size: 11px; color: var(--text-dim); margin-top: 3px; font-family: var(--font-mono); }

.panel {
  position: relative;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 18px 18px 14px;
  margin-bottom: 18px;
}
.panel h2 {
  margin: 0 0 12px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 10px;
}
.panel h2 .chan { color: var(--accent); }
.panel h2 .legend { font-size: 11px; font-weight: 500; text-transform: none; letter-spacing: 0; margin-left: auto; display: flex; gap: 14px; font-family: var(--font-mono); }
.legend-item { display: inline-flex; align-items: center; gap: 5px; color: var(--text-dim); }
.legend-swatch { width: 9px; height: 9px; border-radius: 2px; }

.telemetry-row { display: grid; grid-template-columns: 220px 1fr; gap: 18px; }
@media (max-width: 820px) { .telemetry-row { grid-template-columns: 1fr; } }
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 760px) { .chart-row { grid-template-columns: 1fr; } }

.gauge-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; padding: 6px 0; }
.gauge-pct { font-family: var(--font-mono); font-size: 13px; fill: var(--text); font-weight: 700; }
.gauge-sub { font-family: var(--font-mono); font-size: 8.5px; fill: var(--text-faint); letter-spacing: 0.08em; }
.vram-meter { width: 100%; }
.vram-track { width: 100%; height: 8px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }
.vram-fill { height: 100%; background: var(--accent-2); box-shadow: 0 0 8px rgba(167,139,250,0.5); transition: width 0.4s ease; }
.vram-label { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim); margin-top: 6px; }

svg.chart { width: 100%; height: 190px; display: block; overflow: visible; }
.grid-line { stroke: var(--border); stroke-width: 1; }
.axis-label { fill: var(--text-faint); font-family: var(--font-mono); font-size: 9.5px; }
.series-line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.endpoint-ring { r: 6.5; }
.endpoint { r: 4.5; }
.empty-note { color: var(--text-faint); font-size: 12.5px; font-style: italic; padding: 30px 0; text-align: center; font-family: var(--font-mono); }
.diag h3 {
  margin: 14px 0 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.diag h3:first-child { margin-top: 0; }
.diag .stats { margin-bottom: 4px; }
.diag-bins { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 8px; }
.diag-bins .label {
  font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-faint); width: 100%;
}
.diag-bin {
  font-family: var(--font-mono); font-size: 12px;
  border: 1px solid var(--border); background: var(--surface-2); padding: 6px 8px;
}
.diag-bin b { margin: 0 8px; color: var(--text); font-weight: 600; }
.diag-bin em { color: var(--text-faint); font-style: normal; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th {
  text-align: right;
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-faint);
  font-weight: 500;
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
}
thead th:first-child, tbody td:first-child { text-align: left; }
tbody td {
  text-align: right;
  padding: 7px 10px;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  border-bottom: 1px solid var(--border);
  color: var(--text-dim);
}
tbody tr:last-child td { border-bottom: none; }
tbody tr.best td { color: var(--text); }
.best-badge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 2px;
  background: rgba(94, 234, 212, 0.14);
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.table-wrap { overflow-x: auto; }

pre.log {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--text-dim);
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 260px;
  overflow-y: auto;
}
pre.log .crit-line { color: var(--crit); }
.cursor { display: inline-block; width: 7px; height: 13px; background: var(--accent); margin-left: 2px; vertical-align: -2px; animation: blink 1s step-end infinite; }
@keyframes blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }

.foot {
  margin-top: 26px;
  color: var(--text-faint);
  font-size: 11px;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  text-align: center;
}

/* ---- milestone banner: "new best" celebration ---- */
#milestone-banner {
  position: fixed;
  top: 18px;
  left: 50%;
  transform: translateX(-50%) translateY(-14px);
  z-index: 10;
  padding: 10px 22px;
  background: var(--surface-2);
  border: 1px solid var(--accent);
  box-shadow: 0 0 24px var(--accent-glow), 0 4px 18px rgba(0,0,0,0.5);
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.4s ease, transform 0.4s ease;
  white-space: nowrap;
}
#milestone-banner.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* ---- mission patches (win-rate badges) ---- */
.patches { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }
.patch {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 10px;
  border: 1px solid var(--border);
  background: var(--surface);
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-faint);
  transition: all 0.4s ease;
}
.patch .patch-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-faint); flex: none; }
.patch.lit { color: var(--accent); border-color: var(--border-bright); }
.patch.lit .patch-dot { background: var(--accent); box-shadow: 0 0 6px var(--accent-glow); }
.patch.lit.new { animation: patch-flash 1.6s ease-in-out 1; }
@keyframes patch-flash { 0%, 100% { background: var(--surface); } 30% { background: rgba(94,234,212,0.18); box-shadow: 0 0 16px var(--accent-glow); } }

/* ---- propulsion (thruster) panel ---- */
.thruster-wrap { display: flex; align-items: center; gap: 12px; }
.thruster-label { font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim); letter-spacing: 0.04em; white-space: nowrap; }

.chatter-line { color: var(--text-faint); font-style: italic; opacity: 0.75; }

/* ---- replay theater ---- */
.theater-caption {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--accent);
  min-height: 1.4em;
  margin-bottom: 12px;
  text-shadow: 0 0 8px var(--accent-glow);
}
.theater-board { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.theater-player {
  border: 1px solid var(--border);
  background: var(--surface-2);
  padding: 10px 12px;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.theater-player.active { border-color: var(--accent); box-shadow: 0 0 14px var(--accent-glow); }
.theater-player-name {
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.04em;
  color: var(--text-dim); margin-bottom: 8px; display: flex; justify-content: space-between;
}
.theater-void { color: var(--crit); font-size: 10px; }
.theater-system { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.theater-star { flex: none; border-radius: 50%; }
.theater-discoveries { display: flex; gap: 3px; flex-wrap: wrap; }
.theater-disc { width: 9px; height: 9px; border-radius: 50%; background: var(--good); box-shadow: 0 0 4px rgba(52,211,153,0.5); }
.theater-slot-empty { width: 9px; height: 9px; border-radius: 50%; border: 1px dashed var(--border-bright); }
.theater-frame-nav { text-align: right; font-family: var(--font-mono); font-size: 10px; color: var(--text-faint); margin-top: 10px; }

/* ---- card spectrum ---- */
.spectrum-legend { display: flex; gap: 14px; font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim); margin-bottom: 8px; flex-wrap: wrap; }
.spectrum-legend .legend-swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 4px; }
</style>
</head>
<body>
<canvas id="starfield"></canvas>
<div id="milestone-banner"></div>
<main>

<div class="top">
  <div>
    <p class="eyebrow">Novacana · Astra-7 · Node: <span id="node-name">-</span></p>
    <h1>Mission Control</h1>
    <p class="subtitle" id="subtitle">Establishing uplink&hellip;</p>
  </div>
  <div class="top-right">
    <div class="status-pill" id="status-pill"><span class="dot idle"></span><span id="status-text">connecting</span></div>
    <div class="met">
      <div class="met-label">Mission Elapsed Time</div>
      <div class="met-clock" id="met-clock">00:00:00</div>
    </div>
  </div>
</div>

<div class="stats" id="stats"></div>

<div class="panel diag" id="diag-panel">
  <h2><span class="chan">▲</span> Validation: measured on the held-out games</h2>
  <div id="diag"><p class="empty-note">Fills in at the end of the next epoch.</p></div>
</div>

<div class="patches" id="patches"></div>

<div class="panel">
  <div class="telemetry-row">
    <div>
      <h2><span class="chan">▲</span> GPU Telemetry</h2>
      <div class="gauge-wrap">
        <div id="gpu-gauge"></div>
        <div class="vram-meter">
          <div class="vram-track"><div class="vram-fill" id="vram-fill" style="width:0%"></div></div>
          <div class="vram-label"><span id="vram-text">- / -</span><span id="gpu-temp-text">-°C</span></div>
        </div>
        <div class="thruster-wrap">
          <canvas id="thruster" width="120" height="56"></canvas>
          <span class="thruster-label" id="thruster-label">BURN: STANDBY</span>
        </div>
      </div>
    </div>
    <div>
      <h2>
        <span class="chan">▲</span> Loss Uplink - recent-batch window
        <span class="legend">
          <span class="legend-item"><span class="legend-swatch" style="background:var(--accent)"></span>policy</span>
          <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-2)"></span>value</span>
        </span>
      </h2>
      <div id="loss-chart"></div>
    </div>
  </div>
</div>

<div class="chart-row">
  <div class="panel">
    <h2><span class="chan">▲</span> Nav - Validation Accuracy</h2>
    <div id="acc-chart"></div>
  </div>
  <div class="panel">
    <h2><span class="chan">▲</span> Comms - Win Rate vs AdvCPU</h2>
    <div id="winrate-chart"></div>
  </div>
</div>

<div class="panel">
  <h2>Flight Log - per epoch</h2>
  <div class="table-wrap">
    <table id="epoch-table"><thead><tr>
      <th>Epoch</th><th>Train policy</th><th>Train value</th><th>Val acc</th><th>Val CE</th><th>Win rate</th><th>Duration</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>

<div class="panel">
  <h2><span class="chan">▲</span> Replay Theater - latest spectator game</h2>
  <div class="theater-caption" id="theater-caption">Awaiting first spectator game&hellip;</div>
  <div class="theater-board" id="theater-board"></div>
  <div class="theater-frame-nav" id="theater-frame-nav"></div>
</div>

<div class="panel">
  <h2><span class="chan">▲</span> Stellar Spectrum - card kinds on board, per epoch</h2>
  <div class="spectrum-legend" id="spectrum-legend"></div>
  <div id="spectrum-chart"></div>
</div>

<div class="panel">
  <h2>Comms Log - raw uplink</h2>
  <pre class="log" id="log-tail">&hellip;</pre>
</div>

<p class="foot" id="foot">UPLINK /api/stream · LIVE PUSH · SSH_HOST <span id="foot-host">-</span></p>

</main>

<script>
// ---- starfield (ambient, cheap: ~160 stars, sine-wave twinkle) ----
// Also hosts the "epoch constellation" (§ below render()) and transient
// starburst particles from milestone celebrations - one canvas, one RAF
// loop, so the fun stuff doesn't cost extra layers/repaints.
const constellation = []; // { epoch, fx, fy, valAcc, winRate, best }
const bursts = []; // { x, y, t0, color }

function seededFrac(seed) {
  const a = Math.sin(seed * 12.9898) * 43758.5453;
  const b = Math.sin(seed * 78.233 + 1) * 43758.5453;
  return [a - Math.floor(a), b - Math.floor(b)];
}

function spawnBurst(x, y, color) {
  bursts.push({ x, y, t0: performance.now(), color });
}

(function () {
  const c = document.getElementById('starfield');
  const ctx = c.getContext('2d');
  let stars = [];
  function resize() {
    c.width = window.innerWidth; c.height = window.innerHeight;
    const n = Math.floor((c.width * c.height) / 9000);
    stars = Array.from({length: n}, () => ({
      x: Math.random() * c.width, y: Math.random() * c.height,
      r: Math.random() * 1.2 + 0.3,
      phase: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.0006 + 0.0003,
    }));
  }
  function frame(t) {
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#5EEAD4';
    for (const s of stars) {
      const tw = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * s.speed + s.phase));
      ctx.globalAlpha = tw * 0.7;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Epoch constellation: one permanent star per completed epoch,
    // brightness = val accuracy, hue = win rate, the best-ever epoch gets
    // a gold pulse. The sky fills in as the run progresses.
    for (const m of constellation) {
      const x = m.fx * c.width, y = m.fy * c.height;
      const acc = m.valAcc ?? 0.5;
      const wr = m.winRate ?? 0;
      const hue = 200 + Math.min(wr, 0.6) / 0.6 * 100; // cyan-blue -> gold as win rate climbs
      const baseR = 1.5 + acc * 3.5;
      const pulse = m.best ? 0.7 + 0.3 * Math.sin(t * 0.0025) : 1;
      const r = baseR * pulse;
      const light = m.best ? 72 : 55 + acc * 20;
      ctx.beginPath();
      ctx.fillStyle = `hsl(${hue}, 85%, ${light}%)`;
      ctx.globalAlpha = m.best ? 1 : 0.85;
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      if (m.best) {
        ctx.globalAlpha = 0.25 * pulse;
        ctx.beginPath();
        ctx.arc(x, y, r * 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // Transient starburst particles (milestone celebrations) - short-lived,
    // self-prune once their 900ms life is up.
    for (let i = bursts.length - 1; i >= 0; i--) {
      const b = bursts[i];
      const age = t - b.t0;
      if (age > 900) { bursts.splice(i, 1); continue; }
      const p = age / 900;
      const rays = 10;
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1.5 * (1 - p);
      for (let j = 0; j < rays; j++) {
        const ang = (j / rays) * Math.PI * 2;
        const len = 6 + p * 46;
        ctx.globalAlpha = (1 - p) * 0.9;
        ctx.beginPath();
        ctx.moveTo(b.x + Math.cos(ang) * 4, b.y + Math.sin(ang) * 4);
        ctx.lineTo(b.x + Math.cos(ang) * len, b.y + Math.sin(ang) * len);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    requestAnimationFrame(frame);
  }
  window.addEventListener('resize', resize);
  resize();
  requestAnimationFrame(frame);
})();

function updateConstellation(epochs) {
  const seen = new Set(constellation.map(m => m.epoch));
  for (const e of epochs) {
    if (seen.has(e.epoch)) continue;
    const [fx, fy] = seededFrac(e.epoch * 7.919);
    constellation.push({ epoch: e.epoch, fx, fy: fy * 0.85 + 0.05, valAcc: e.val_acc, winRate: e.win_rate, best: false });
  }
  // recompute which one is "best" (highest val acc) so the gold pulse
  // follows the actual best checkpoint as training progresses
  let bestEpoch = null, bestAcc = -1;
  for (const m of constellation) {
    m.valAcc = epochs.find(e => e.epoch === m.epoch)?.val_acc ?? m.valAcc;
    m.winRate = epochs.find(e => e.epoch === m.epoch)?.win_rate ?? m.winRate;
    if (m.valAcc > bestAcc) { bestAcc = m.valAcc; bestEpoch = m.epoch; }
  }
  for (const m of constellation) m.best = (m.epoch === bestEpoch);
}

const ACCENT = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
const ACCENT2 = getComputedStyle(document.documentElement).getPropertyValue('--accent-2').trim();
const GOOD = getComputedStyle(document.documentElement).getPropertyValue('--good').trim();
const WARN = getComputedStyle(document.documentElement).getPropertyValue('--warn').trim();
const CRIT = getComputedStyle(document.documentElement).getPropertyValue('--crit').trim();
const SURFACE = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim();

function fmt(x, d=4) { return (x === null || x === undefined) ? '-' : Number(x).toFixed(d); }
function fmtDur(s) {
  if (s === null || s === undefined) return '-';
  s = Math.round(s);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function fmtClock(s) {
  s = Math.max(0, Math.round(s || 0));
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  const pad = n => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(sec)}`;
}

function niceExtent(vals, padFrac=0.1) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * padFrac;
  return [lo - pad, hi + pad];
}

// Mark spec: 2px round-cap line, >=8px endpoint marker with a 2px
// surface-color ring so it stays legible where lines cross/overlap.
function lineChart(el, series, opts={}) {
  const w = 560, h = 190, padL = 38, padR = 10, padT = 10, padB = 22;
  if (!series.length || series.every(s => s.ys.length === 0)) {
    el.innerHTML = '<p class="empty-note">no telemetry yet&hellip;</p>';
    return;
  }
  const n = Math.max(...series.map(s => s.ys.length));
  if (n < 2) { el.innerHTML = '<p class="empty-note">acquiring signal&hellip;</p>'; return; }

  const flatY = series.flatMap(s => s.ys);
  const [yMin, yMax] = niceExtent(flatY);
  const xToPx = i => padL + (i / (n - 1)) * (w - padL - padR);
  const yToPx = v => padT + (1 - (v - yMin) / (yMax - yMin)) * (h - padT - padB);

  let svg = `<svg class="chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  for (let i = 0; i <= 3; i++) {
    const gy = padT + i * (h - padT - padB) / 3;
    const val = yMax - i * (yMax - yMin) / 3;
    svg += `<line class="grid-line" x1="${padL}" y1="${gy}" x2="${w-padR}" y2="${gy}" />`;
    svg += `<text class="axis-label" x="${padL - 6}" y="${gy + 3}" text-anchor="end">${val.toFixed(opts.yDec ?? 2)}</text>`;
  }
  series.forEach((s) => {
    const ys = s.ys;
    if (!ys.length) return;
    let d = '';
    ys.forEach((v, i) => { d += (i === 0 ? 'M' : 'L') + xToPx(i).toFixed(2) + ',' + yToPx(v).toFixed(2) + ' '; });
    svg += `<path class="series-line" d="${d}" stroke="${s.color}" />`;
    const lastI = ys.length - 1;
    const cx = xToPx(lastI).toFixed(2), cy = yToPx(ys[lastI]).toFixed(2);
    svg += `<circle class="endpoint-ring" cx="${cx}" cy="${cy}" fill="${SURFACE}" />`;
    svg += `<circle class="endpoint" cx="${cx}" cy="${cy}" fill="${s.color}" />`;
  });
  svg += `</svg>`;
  el.innerHTML = svg;
}

// Radial gauge for GPU utilization - arc fill colored by severity
// (nominal/elevated/critical), unfilled track a dim step of the surface.
function radialGauge(el, pct, opts={}) {
  const size = 148, stroke = 12, r = (size - stroke) / 2, cx = size/2, cy = size/2;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, pct ?? 0));
  const offset = circumference * (1 - clamped / 100);
  // Inverted vs. a typical meter: for GPU *utilization*, high is nominal
  // and low is the problem (this project's own batched-inference fix, see
  // ASTRA7.md/STATE.md, was built around exactly a "only ~30% GPU util"
  // bottleneck - self-play calls serializing one-at-a-time through the
  // CUDA driver instead of batching). A generic high=bad meter would color
  // healthy 90%+ training amber, which is backwards.
  const color = pct === null || pct === undefined ? getComputedStyle(document.documentElement).getPropertyValue('--text-faint').trim()
    : clamped < 20 ? CRIT : clamped < 40 ? WARN : ACCENT;
  el.innerHTML = `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${SURFACE}" stroke-width="${stroke}" opacity="0.6"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}"
        stroke-linecap="round" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"
        transform="rotate(-90 ${cx} ${cy})" style="filter: drop-shadow(0 0 6px ${color}); transition: stroke-dashoffset 0.5s ease;"/>
      <text x="${cx}" y="${cy - 2}" text-anchor="middle" class="gauge-pct">${pct === null || pct === undefined ? '-' : Math.round(pct) + '%'}</text>
      <text x="${cx}" y="${cy + 14}" text-anchor="middle" class="gauge-sub">GPU UTIL</text>
    </svg>`;
}

// ---- replay theater: animates spectator.rs's move-by-move game trace ----
const STAR_COLORS = {
  Protostar: '#5A6B9E', YellowDwarf: '#F5D76E', RedGiant: '#E8724C',
  RedSupergiant: '#D6432B', WhiteDwarf: '#EAF2FF', RedDwarf: '#C0483A', BrownDwarf: '#8B6248',
};
const STAR_RADIUS = { Protostar: 5, YellowDwarf: 8, RedGiant: 10, RedSupergiant: 11, WhiteDwarf: 6, RedDwarf: 6, BrownDwarf: 6 };

let theaterGameKey = null, theaterFrameIdx = 0, theaterTimer = null;

function renderTheaterFrame(game, idx) {
  const frame = game.frames[idx];
  if (!frame) return;
  document.getElementById('theater-caption').textContent = `${frame.player_name} ${frame.action}.`;
  const board = frame.board;
  document.getElementById('theater-board').innerHTML = game.players.map((name, pi) => {
    const chart = board.star_charts[pi];
    const isActive = pi === board.active_player;
    const systemsHtml = (chart && chart.systems.length) ? chart.systems.map(sys => {
      const color = STAR_COLORS[sys.star_name] || '#8B95AC';
      const r = STAR_RADIUS[sys.star_name] || 7;
      const filled = sys.discoveries.length;
      const slots = Array.from({length: sys.class}, (_, i) =>
        i < filled ? '<span class="theater-disc"></span>' : '<span class="theater-slot-empty"></span>'
      ).join('');
      return `<div class="theater-system">
        <svg width="${r*2+4}" height="${r*2+4}"><circle class="theater-star" cx="${r+2}" cy="${r+2}" r="${r}" fill="${color}" style="filter:drop-shadow(0 0 4px ${color})"/></svg>
        <div class="theater-discoveries">${slots}</div>
      </div>`;
    }).join('') : '<p class="empty-note" style="padding:8px 0">no systems yet</p>';
    const voidTag = (board.dark_matter_void !== null && board.dark_matter_void !== undefined)
      ? `<span class="theater-void">VOID: CLASS ${board.dark_matter_void}</span>` : '';
    return `<div class="theater-player ${isActive ? 'active' : ''}">
      <div class="theater-player-name"><span>${escapeHtml(name)} · hand ${board.hand_sizes[pi] ?? '-'}</span>${voidTag}</div>
      ${systemsHtml}
    </div>`;
  }).join('');
  document.getElementById('theater-frame-nav').textContent =
    `turn ${frame.board.turn} · move ${idx + 1} / ${game.frames.length}` + (game.winner_idx !== undefined && idx === game.frames.length - 1 ? ` · winner: ${game.players[game.winner_idx]}` : '');
}

function updateTheater(spectatorGame) {
  if (!spectatorGame || !spectatorGame.frames || !spectatorGame.frames.length) {
    document.getElementById('theater-caption').textContent = 'Awaiting first spectator game - lands after the first win-rate eval completes.';
    document.getElementById('theater-board').innerHTML = '';
    document.getElementById('theater-frame-nav').textContent = '';
    return;
  }
  // Key on frame count + winner so a genuinely new game restarts playback
  // from frame 0, but a re-poll of the SAME still-latest game (nothing
  // changed remotely) doesn't jank the animation back to the start.
  const key = spectatorGame.frames.length + ':' + spectatorGame.winner_idx + ':' + spectatorGame.players.join(',');
  if (key === theaterGameKey) return;
  theaterGameKey = key;
  theaterFrameIdx = 0;
  renderTheaterFrame(spectatorGame, 0);
  clearInterval(theaterTimer);
  theaterTimer = setInterval(() => {
    theaterFrameIdx = (theaterFrameIdx + 1) % spectatorGame.frames.length;
    renderTheaterFrame(spectatorGame, theaterFrameIdx);
  }, 2200);
}

// ---- stellar spectrum: card-kind mix over epochs (stacked bar) ----
const SPECTRUM_CATEGORIES = [
  { key: 'star', label: 'Star', color: ACCENT },
  { key: 'discovery', label: 'Discovery', color: GOOD },
  { key: 'flare', label: 'Flare', color: WARN },
  { key: 'omen', label: 'Omen', color: ACCENT2 },
  { key: 'fracture', label: 'Fracture', color: CRIT },
];

function renderSpectrumChart(log) {
  document.getElementById('spectrum-legend').innerHTML = SPECTRUM_CATEGORIES.map(c =>
    `<span><span class="legend-swatch" style="background:${c.color}"></span>${c.label}</span>`
  ).join('');
  const el = document.getElementById('spectrum-chart');
  if (!log || !log.length) {
    el.innerHTML = '<p class="empty-note">no spectrum data yet&hellip;</p>';
    return;
  }
  const w = 1140, h = 170, padL = 30, padR = 10, padT = 10, padB = 20;
  const barGap = 4;
  const barW = Math.max(4, (w - padL - padR) / log.length - barGap);
  const totals = log.map(row => SPECTRUM_CATEGORIES.reduce((s, c) => s + (row[c.key] || 0), 0));
  const maxTotal = Math.max(...totals, 1);

  let svg = `<svg class="chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  for (let i = 0; i <= 2; i++) {
    const gy = padT + i * (h - padT - padB) / 2;
    const val = maxTotal - i * maxTotal / 2;
    svg += `<line class="grid-line" x1="${padL}" y1="${gy}" x2="${w-padR}" y2="${gy}" />`;
    svg += `<text class="axis-label" x="${padL - 6}" y="${gy + 3}" text-anchor="end">${Math.round(val)}</text>`;
  }
  log.forEach((row, i) => {
    const x = padL + i * (barW + barGap);
    let yCursor = h - padB;
    for (const cat of SPECTRUM_CATEGORIES) {
      const v = row[cat.key] || 0;
      if (v <= 0) continue;
      const segH = (h - padT - padB) * (v / maxTotal);
      svg += `<rect x="${x.toFixed(1)}" y="${(yCursor - segH).toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(segH - 1, 0).toFixed(1)}" fill="${cat.color}" opacity="0.9" />`;
      yCursor -= segH;
    }
    if (i === log.length - 1 || i % Math.ceil(log.length / 10 || 1) === 0) {
      svg += `<text class="axis-label" x="${(x + barW/2).toFixed(1)}" y="${h - 4}" text-anchor="middle">${row.epoch}</text>`;
    }
  });
  svg += `</svg>`;
  el.innerHTML = svg;
}

// ---- propulsion (thruster): particle exhaust reacting to loss trend ----
const thrusterCanvas = document.getElementById('thruster');
const thrusterCtx = thrusterCanvas.getContext('2d');
let thrusterParticles = [];
let thrusterState = { trend: 'idle', running: false };

function thrusterFrame() {
  const w = thrusterCanvas.width, h = thrusterCanvas.height;
  thrusterCtx.clearRect(0, 0, w, h);
  const shipX = 22, shipY = h / 2;
  thrusterCtx.fillStyle = '#C7D0E0';
  thrusterCtx.beginPath();
  thrusterCtx.moveTo(shipX - 12, shipY);
  thrusterCtx.lineTo(shipX + 8, shipY - 7);
  thrusterCtx.lineTo(shipX + 8, shipY + 7);
  thrusterCtx.closePath();
  thrusterCtx.fill();

  const trend = thrusterState.trend;
  const active = thrusterState.running && trend !== 'idle';
  if (active && Math.random() < (trend === 'worsening' ? 0.5 : 0.85)) {
    const jitter = trend === 'worsening' ? 3.2 : trend === 'flat' ? 1.6 : 0.6;
    thrusterParticles.push({
      x: shipX + 8, y: shipY + (Math.random() - 0.5) * 6,
      vx: 1.4 + Math.random() * 1.2, vy: (Math.random() - 0.5) * jitter,
      life: 0, maxLife: 24 + Math.random() * 14,
    });
  }
  const color = trend === 'improving' ? ACCENT : trend === 'worsening' ? CRIT : trend === 'flat' ? WARN : '#3A4360';
  for (let i = thrusterParticles.length - 1; i >= 0; i--) {
    const p = thrusterParticles[i];
    p.x += p.vx; p.y += p.vy; p.life++;
    if (p.life > p.maxLife || p.x > w) { thrusterParticles.splice(i, 1); continue; }
    const frac = 1 - p.life / p.maxLife;
    thrusterCtx.globalAlpha = frac * 0.8;
    thrusterCtx.fillStyle = color;
    thrusterCtx.beginPath();
    thrusterCtx.arc(p.x, p.y, 1.6 * frac + 0.4, 0, Math.PI * 2);
    thrusterCtx.fill();
  }
  thrusterCtx.globalAlpha = 1;
  requestAnimationFrame(thrusterFrame);
}
requestAnimationFrame(thrusterFrame);

// Loss slope (recent heartbeats, falling back to epoch-over-epoch) drives
// the exhaust: a clean descent burns steady, a flat loss flickers, a rising
// loss sputters - turns the least-glanceable number into something alive.
function updateThruster(d) {
  const hbs = d.heartbeats || [];
  const eps = d.epochs || [];
  let trend = 'idle';
  const running = d.status === 'running';
  if (hbs.length >= 2) {
    const delta = hbs[hbs.length - 1].policy_recent - hbs[hbs.length - 2].policy_recent;
    trend = delta < -0.01 ? 'improving' : delta > 0.01 ? 'worsening' : 'flat';
  } else if (eps.length >= 2) {
    const delta = eps[eps.length - 1].policy - eps[eps.length - 2].policy;
    trend = delta < -0.01 ? 'improving' : delta > 0.01 ? 'worsening' : 'flat';
  } else if (running) {
    trend = 'improving';
  }
  thrusterState = { trend, running };
  const label = !running ? 'BURN: STANDBY'
    : trend === 'improving' ? 'BURN: CLEAN'
    : trend === 'worsening' ? 'BURN: SPUTTERING'
    : 'BURN: FLICKERING';
  document.getElementById('thruster-label').textContent = label;
}

// ---- mission patches (win-rate badges) + milestone banner ----
const WIN_THRESHOLDS = [
  { at: 0.10, name: 'First Contact' },
  { at: 0.25, name: 'Established' },
  { at: 0.40, name: 'Competitive' },
  { at: 0.55, name: 'Dominant' },
];
const celebratedThresholds = new Set();
let seenBestAcc = null;
let firstRender = true;

function showBanner(text) {
  const el = document.getElementById('milestone-banner');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(showBanner._t);
  showBanner._t = setTimeout(() => el.classList.remove('show'), 3800);
}

function renderPatches(epochs) {
  const rates = epochs.map(e => e.win_rate).filter(w => w !== null && w !== undefined);
  const bestWr = rates.length ? Math.max(...rates) : -1;
  const el = document.getElementById('patches');
  el.innerHTML = WIN_THRESHOLDS.map(t => {
    const lit = bestWr >= t.at;
    const isNew = lit && !celebratedThresholds.has(t.at);
    if (lit) celebratedThresholds.add(t.at);
    if (isNew && !firstRender) {
      showBanner(`⬡ Mission Patch Earned - ${t.name} (win rate ≥ ${(t.at * 100).toFixed(0)}%)`);
      spawnBurst(window.innerWidth * (0.3 + Math.random() * 0.4), window.innerHeight * (0.18 + Math.random() * 0.15), ACCENT2);
    }
    return `<div class="patch ${lit ? 'lit' : ''} ${isNew && !firstRender ? 'new' : ''}"><span class="patch-dot"></span>${t.name} · ${(t.at * 100).toFixed(0)}%</div>`;
  }).join('');
}

// ---- procedural comms chatter - ambient flavor seeded by real metrics ----
const CHATTER_TEMPLATES = [
  d => `[COMMS] Sector scan nominal - GPU relay holding at ${d.gpu_util !== null ? Math.round(d.gpu_util) : '-'}% throughput.`,
  d => `[COMMS] Probe reports ${d.best_val_acc !== null ? (d.best_val_acc * 100).toFixed(0) : '-'}% chart-recognition confidence.`,
  d => `[COMMS] Flight recorder: epoch ${d.current_epoch || '-'} trajectory within tolerance.`,
  d => `[COMMS] Dark Matter Void containment steady. No anomalies detected.`,
  d => `[COMMS] Win-rate telemetry: ${d.best_val_acc !== null ? 'signal strong' : 'awaiting first contact'}.`,
  d => `[COMMS] Star Chart uplink holding at ${d.gpu_temp !== null ? d.gpu_temp : '-'}°C core temp.`,
];
function chatterLine(d) {
  if (!d.latest_heartbeat) return null;
  const seed = (d.current_epoch || 0) * 97 + (d.latest_heartbeat.batch || 0);
  return CHATTER_TEMPLATES[seed % CHATTER_TEMPLATES.length](d);
}

let lastData = null, lastPollAt = null, etaSeed = null;

// Real-time push (2026-07-15): the server pushes a new reading over SSE
// the instant its background poller's SSH round-trip completes, instead
// of this page fetching on its own timer - see dashboard_server.py's
// module doc. EventSource auto-reconnects on its own if the connection
// drops (a flaky Tailscale link, the server restarting), so no manual
// retry loop is needed here.
function connectStream() {
  const es = new EventSource('/api/stream');
  es.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch (e) { return; }
    if (d.error) return;
    lastData = d;
    lastPollAt = performance.now();
    render(d);
  };
  es.onopen = () => { document.getElementById('foot').dataset.live = '1'; };
  es.onerror = () => {
    document.getElementById('status-text').textContent = 'uplink lost';
    // EventSource retries on its own; nothing else to do here.
  };
}

// One-time fallback fetch so the page isn't blank while the SSE
// connection establishes (or in the rare case EventSource is unsupported).
async function pollOnce() {
  try {
    const r = await fetch('/api/data', { cache: 'no-store' });
    const d = await r.json();
    if (!d.error && d.generated_at) { lastData = d; lastPollAt = performance.now(); render(d); }
  } catch (e) { /* SSE connection below will take over once it's up */ }
}

function dash(x) { return x === null || x === undefined; }
function pct(x) { return dash(x) ? '-' : (x * 100).toFixed(2) + '%'; }
function num(x, d) { return dash(x) ? '-' : Number(x).toFixed(d); }
function cell(label, value, sub) {
  return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
}
function decisionCell(group, key, label, note) {
  const b = (group && group[key]) || {};
  const bits = [];
  if (note) bits.push(note);
  if (b.n) bits.push('n=' + Number(b.n).toLocaleString());
  return cell(label, dash(b.acc) ? '-' : (b.acc * 100).toFixed(2) + '%', bits.join(' · '));
}
function bucketRow(title, items) {
  if (!items || !items.length) return '';
  return `<div class="diag-bins"><div class="label">${title}</div>` + items.map(b =>
    `<div class="diag-bin"><span>${b.label}</span><b>${dash(b.acc) ? '-' : (b.acc * 100).toFixed(1) + '%'}</b><em>n=${Number(b.n).toLocaleString()}</em></div>`
  ).join('') + '</div>';
}
function renderDiag(g) {
  const el = document.getElementById('diag');
  if (!el) return;
  if (!g) {
    el.innerHTML = '<p class="empty-note">Fills in at the end of the next epoch.</p>';
    return;
  }
  const grp = g.groups || {};
  el.innerHTML = `
    <h3>Policy</h3>
    <div class="stats">
      ${cell('Top-1', pct(g.top1), g.n ? 'n=' + Number(g.n).toLocaleString() : '')}
      ${cell('Top-3', pct(g.top3))}
      ${cell('Top-5', pct(g.top5))}
      ${cell('Teacher probability', pct(g.teacher_p))}
      ${cell('Teacher mean rank', num(g.teacher_rank, 2))}
      ${cell('Teacher margin', num(g.teacher_margin, 3))}
      ${cell('Normalized entropy', num(g.entropy, 3))}
    </div>
    <h3>By decision</h3>
    <div class="stats">
      ${decisionCell(grp, 'play', 'Play', 'incl. pass')}
      ${decisionCell(grp, 'chain', 'Chain')}
      ${decisionCell(grp, 'decide', 'Decide')}
      ${decisionCell(grp, 'target', 'Target')}
      ${decisionCell(grp, 'discard', 'Discard')}
    </div>
    <h3>Value</h3>
    <div class="stats">
      ${cell('MSE', num(g.mse, 4))}
      ${cell('MAE', num(g.mae, 4))}
      ${cell('Mean V', num(g.mean_v, 3))}
      ${cell('Winner V', num(g.winner_v, 3), g.winner_n ? 'n=' + Number(g.winner_n).toLocaleString() : '')}
      ${cell('Loser V', num(g.loser_v, 3), g.loser_n ? 'n=' + Number(g.loser_n).toLocaleString() : '')}
      ${cell('AUC', num(g.auc, 3))}
    </div>
    <h3>Difficulty</h3>
    ${bucketRow('Accuracy × legal actions', g.legal)}
    ${bucketRow('Accuracy × history length', g.history)}
  `;
}

function render(d) {
  const pill = document.getElementById('status-pill');
  const dot = pill.querySelector('.dot');
  const statusText = document.getElementById('status-text');
  dot.className = 'dot ' + d.status;
  const statusLabel = { running: 'training', crashed: 'anomaly', idle: 'standby', stopped: 'stopped' }[d.status] || d.status;
  statusText.textContent = statusLabel;

  document.getElementById('node-name').textContent = d.ssh_host || '-';
  document.getElementById('foot-host').textContent = d.ssh_host || '-';
  document.getElementById('subtitle').textContent =
    `${d.games_per_epoch ? d.games_per_epoch.toLocaleString() + ' games/epoch × ' + d.players + ' players' : 'astra7 training node'} · ${d.gpu_name || 'GPU unknown'} · sync ${new Date(d.generated_at).toLocaleTimeString()}`;

  const hb = d.latest_heartbeat;
  const stats = [
    { label: 'Epoch', value: d.current_epoch && d.total_epochs ? `${d.current_epoch} / ${d.total_epochs}` : '-' },
    { label: 'Batch', value: hb ? `${hb.batch.toLocaleString()} / ${hb.total_batches.toLocaleString()}` : '-' },
    { label: 'Best val acc', value: d.best_val_acc !== null ? (d.best_val_acc*100).toFixed(1) + '%' : '-', accent: true },
    { label: 'Policy loss', value: hb ? fmt(hb.policy, 3) : '-' },
    { label: 'Learning rate', value: hb ? hb.lr.toExponential(2) : '-' },
  ];
  document.getElementById('stats').innerHTML = stats.map(s =>
    `<div class="stat"><div class="label">${s.label}</div><div class="value${s.accent ? ' accent' : ''}">${s.value}</div>${s.sub ? `<div class="sub">${s.sub}</div>` : ''}</div>`
  ).join('') + `<div class="stat"><div class="label">Orbital Insertion</div><div class="value accent" id="tminus-value">-</div></div>`;
  etaSeed = d.eta_s;

  renderDiag(d.diag);
  updateConstellation(d.epochs || []);
  renderPatches(d.epochs || []);
  updateTheater(d.spectator_game);
  renderSpectrumChart(d.card_spectrum_log || []);
  updateThruster(d);
  if (d.best_val_acc !== null && d.best_val_acc !== undefined) {
    if (seenBestAcc !== null && d.best_val_acc > seenBestAcc + 1e-6) {
      showBanner(`◆ New Best Discovered - Val Acc ${(d.best_val_acc * 100).toFixed(1)}%`);
      spawnBurst(window.innerWidth * (0.3 + Math.random() * 0.4), window.innerHeight * (0.15 + Math.random() * 0.2), ACCENT);
    }
    seenBestAcc = d.best_val_acc;
  }
  firstRender = false;

  radialGauge(document.getElementById('gpu-gauge'), d.gpu_util);
  const vramPct = (d.gpu_mem_used !== null && d.gpu_mem_total) ? (d.gpu_mem_used / d.gpu_mem_total * 100) : 0;
  document.getElementById('vram-fill').style.width = vramPct.toFixed(1) + '%';
  document.getElementById('vram-text').textContent = (d.gpu_mem_used !== null)
    ? `${(d.gpu_mem_used/1024).toFixed(1)}G / ${(d.gpu_mem_total/1024).toFixed(1)}G VRAM` : 'VRAM -';
  document.getElementById('gpu-temp-text').textContent = d.gpu_temp !== null && d.gpu_temp !== undefined
    ? `${Math.round(d.gpu_temp)}°C`
    : (d.gpu_power_w !== null && d.gpu_power_w !== undefined ? `${Math.round(d.gpu_power_w)}W` : '-°C');

  // Windowed (recent-batches) figures, not the cumulative epoch-so-far
  // average - the cumulative one is structurally flat-looking late in a
  // large epoch and reads as a false plateau (2026-07-10 finding).
  lineChart(document.getElementById('loss-chart'), [
    { ys: d.heartbeats.map(h => h.policy_recent), color: ACCENT },
    { ys: d.heartbeats.map(h => h.value_recent), color: ACCENT2 },
  ], { yDec: 2 });

  lineChart(document.getElementById('acc-chart'), [
    { ys: d.epochs.map(e => e.val_acc), color: GOOD },
  ], { yDec: 3 });

  lineChart(document.getElementById('winrate-chart'), [
    { ys: d.epochs.filter(e => e.win_rate !== null && e.win_rate !== undefined).map(e => e.win_rate), color: ACCENT2 },
  ], { yDec: 3 });

  const tbody = document.querySelector('#epoch-table tbody');
  tbody.innerHTML = d.epochs.slice().reverse().map(e => `
    <tr class="${e.best ? 'best' : ''}">
      <td>${e.epoch} / ${e.total_epochs}${e.best ? '<span class="best-badge">BEST</span>' : ''}</td>
      <td>${fmt(e.policy, 4)}</td>
      <td>${fmt(e.value, 4)}</td>
      <td>${(e.val_acc*100).toFixed(2)}%</td>
      <td>${fmt(e.val_ce, 4)}</td>
      <td>${e.win_rate !== null && e.win_rate !== undefined ? (e.win_rate*100).toFixed(1) + '%' : '-'}</td>
      <td>${fmtDur(e.seconds)}</td>
    </tr>`).join('') || '<tr><td colspan="7" class="empty-note">no epoch evals completed yet</td></tr>';

  const logEl = document.getElementById('log-tail');
  const chatter = chatterLine(d);
  logEl.innerHTML = (d.tail_lines || []).map(l =>
    l.toLowerCase().includes('panic') || l.toLowerCase().includes('error')
      ? `<span class="crit-line">${escapeHtml(l)}</span>`
      : escapeHtml(l)
  ).join('\\n') + (chatter ? '\\n<span class="chatter-line">' + escapeHtml(chatter) + '</span>' : '') + '<span class="cursor"></span>';
  logEl.scrollTop = logEl.scrollHeight;

  if (d.status === 'crashed' && d.panic_excerpt) {
    document.getElementById('foot').innerHTML =
      `<span style="color:var(--crit)">ANOMALY: ${escapeHtml(d.panic_excerpt)}</span>`;
  }
}

// MET clock + T-minus countdown tick client-side every second between
// polls, seeded from the last known elapsed_total_s/eta_s so they don't
// sit frozen for the 6s gap.
function tickClock() {
  if (lastData && lastData.elapsed_total_s !== undefined && lastData.status === 'running') {
    const drift = (performance.now() - lastPollAt) / 1000;
    document.getElementById('met-clock').textContent = fmtClock(lastData.elapsed_total_s + drift);
    const tminus = document.getElementById('tminus-value');
    if (tminus) {
      tminus.textContent = etaSeed !== null && etaSeed !== undefined
        ? 'T-' + fmtClock(Math.max(0, etaSeed - drift)) : '-';
    }
  } else if (lastData && lastData.elapsed_total_s !== undefined) {
    document.getElementById('met-clock').textContent = fmtClock(lastData.elapsed_total_s);
    const tminus = document.getElementById('tminus-value');
    if (tminus) tminus.textContent = etaSeed !== null && etaSeed !== undefined ? 'T-' + fmtClock(etaSeed) : '-';
  }
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

pollOnce();
connectStream();
setInterval(tickClock, 1000);
</script>

</body>
</html>
"""


# ── Background poller + push (real-time, see module doc) ───────────────────
#
# One thread does the (single, multiplexed) SSH round-trip on
# POLL_INTERVAL_S; every /api/stream (SSE) client is notified via a
# Condition the instant a new result lands, instead of each browser tab
# separately triggering its own SSH call on its own timer. `/api/data`
# stays as a synchronous fallback (curl-friendly, and what a client falls
# back to if EventSource ever isn't available) - it just serves whatever
# the cache currently holds rather than blocking on a fresh SSH call, so
# it's always instant too.
_cache_lock = threading.Condition()
_cache = {"data": None, "generation": 0}


def _poller_loop():
    while True:
        try:
            data = fake_get_data() if FAKE_MODE else get_data()
        except Exception as e:
            data = {"error": str(e)}
        with _cache_lock:
            _cache["data"] = data
            _cache["generation"] += 1
            _cache_lock.notify_all()
        time.sleep(POLL_INTERVAL_S)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        # BaseHTTP answers HEAD with 501 unless this exists. Caddy and curl -I use it.
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            with _cache_lock:
                data = _cache["data"] if _cache["data"] is not None else {}
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_sent_gen = 0
            try:
                while True:
                    with _cache_lock:
                        # Wake on either a new poll result or a 15s
                        # heartbeat timeout - the timeout keeps a comment/
                        # ping flowing so intermediary proxies (and the
                        # browser) don't treat a quiet stream as dead.
                        self._cache_notified = _cache_lock.wait_for(
                            lambda: _cache["generation"] != last_sent_gen, timeout=15
                        )
                        gen = _cache["generation"]
                        data = _cache["data"]
                    if gen != last_sent_gen and data is not None:
                        payload = json.dumps(data)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last_sent_gen = gen
                    else:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client navigated away / tab closed - not an error
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # daemon_threads: SSE connections are long-lived (one thread each, for
    # as long as a browser tab stays open) - without this the process
    # would hang on exit waiting for them.
    socketserver.ThreadingTCPServer.daemon_threads = True
    poller = threading.Thread(target=_poller_loop, daemon=True)
    poller.start()
    with socketserver.ThreadingTCPServer((DASHBOARD_HOST, PORT), Handler) as httpd:
        mode = "FAKE DATA (demo mode)" if FAKE_MODE else (f"local probe on {SSH_HOST}" if LOCAL_MODE else f"ssh to {SSH_HOST}")
        print(f"Astra-7 dashboard live at http://localhost:{PORT} (bind {DASHBOARD_HOST}, {mode}, every {POLL_INTERVAL_S}s, pushed via SSE)")
        httpd.serve_forever()
