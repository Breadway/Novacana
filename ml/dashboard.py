#!/usr/bin/env python3
"""Astra-6 training dashboard.

Live view of ALL training activity: BC pretrain runs (sweep points and
warm-start/candidate-only runs), RL (PPO) segments, and AZ runs - parsed from
learning_curve_out/*.log and logs/*.log. Stdlib only.

Usage:  python3 dashboard.py [--port 7750]
"""

import argparse
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))

JOINT_RE = re.compile(
    r"epoch (\d+)/(\d+): policy=([\d.]+) value=([\d.]+)"
    r"(?:\s+val_acc=([\d.]+) val_ce=([\d.]+))?")
LOSS_RE    = re.compile(r"epoch (\d+)/(\d+): loss=([\d.]+)")
GEN_RE     = re.compile(r"(\d+) samples in ([\d.]+)s")
RESTORE_RE = re.compile(r"Restoring best-policy checkpoint \(val_acc=([\d.]+)\)")
ITER_RE    = re.compile(r"^Iter\s+(\d+)\s+new=\s*(\d+)\s+buf=\s*(\d+)\s+pl=([+-][\d.]+)\s+vl=([\d.]+).*\(([\d.]+)s\)", re.MULTILINE)
EVAL_AT_RE = re.compile(r"Eval @ iter (\d+): ([\d.]+)%")
RL_HDR_RE  = re.compile(r"RL \(PPO\+PER\): (\d+) iters")
AZ_HDR_RE  = re.compile(r"AlphaZero: (\d+) iters")
AGREE_RE   = re.compile(r"agree=\s*([\d.]+)%\s+top2=\s*([\d.]+)%.*?mean_CE=([\d.]+)")
EVAL_LN_RE = re.compile(r"Eval \(\d+ games.*?\):\s+([\d.]+)%")


def parse_log(path, name):
    """Classify and parse one training log. Returns a bc run, rl run, or None."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    if not text.strip():
        return None
    mtime = os.path.getmtime(path)

    hdr = RL_HDR_RE.search(text) or AZ_HDR_RE.search(text)
    if hdr:
        run = {"kind": "rl", "name": name, "mtime": mtime,
               "planned_iters": int(hdr.group(1)),
               "az": bool(AZ_HDR_RE.search(text)),
               "iters": [], "evals": [], "crashed": False}
        for m in ITER_RE.finditer(text):
            run["iters"].append({
                "iter": int(m.group(1)), "pl": float(m.group(4)),
                "vl": float(m.group(5)), "secs": float(m.group(6)),
            })
        for m in EVAL_AT_RE.finditer(text):
            run["evals"].append({"iter": int(m.group(1)), "wr": float(m.group(2))})
        run["complete"] = "training complete" in text
        run["crashed"]  = ("SIGSEGV" in text or "free()" in text
                           or "precompiled NUM_THREADS exceeded" in text)
        return run

    # BC/pretrain-style log: scan sequentially, tracking phase.
    run = {"kind": "bc", "name": name, "mtime": mtime, "phase": "generating",
           "gen_samples": None, "warm_start": "Warm start" in text,
           "joint": [], "value": [], "cand": [], "best_acc": None}
    phase = "joint"
    for line in text.splitlines():
        if "Value fine-tune" in line:
            phase = "value"; run["phase"] = "value"; continue
        if "Candidate pretraining" in line:
            phase = "cand"; run["phase"] = "candidate"; continue
        if "Pretrain complete" in line:
            run["phase"] = "done"; continue
        m = GEN_RE.search(line)
        if m and run["gen_samples"] is None:
            run["gen_samples"] = int(m.group(1)); run["phase"] = "joint"; continue
        m = RESTORE_RE.search(line)
        if m:
            run["best_acc"] = float(m.group(1)); continue
        m = JOINT_RE.search(line)
        if m and phase == "joint":
            run["joint"].append({
                "epoch": int(m.group(1)), "total": int(m.group(2)),
                "policy": float(m.group(3)), "value": float(m.group(4)),
                "val_acc": float(m.group(5)) if m.group(5) else None,
            })
            run["phase"] = "joint"
            continue
        m = LOSS_RE.search(line)
        if m and phase in ("value", "cand"):
            run[phase].append({"epoch": int(m.group(1)), "total": int(m.group(2)),
                               "loss": float(m.group(3))})
    if not (run["joint"] or run["value"] or run["cand"] or run["gen_samples"]):
        return None
    accs = [e["val_acc"] for e in run["joint"] if e["val_acc"] is not None]
    if run["best_acc"] is None and accs:
        run["best_acc"] = max(accs)
    return run


def parse_fidelity_log(path):
    out = {}
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return out
    ca_part, _, rc_part = text.partition("--- respond_to_chain ---")
    m = AGREE_RE.search(ca_part)
    if m:
        out["choose_agree"], out["choose_top2"] = float(m.group(1)), float(m.group(2))
    m = AGREE_RE.search(rc_part)
    if m:
        out["respond_agree"] = float(m.group(1))
    return out


def parse_eval_log(path):
    try:
        with open(path) as f:
            m = EVAL_LN_RE.search(f.read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


def sysinfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            mem = dict(line.split(":", 1) for line in f)
        info["mem_avail_gb"] = int(mem["MemAvailable"].split()[0]) / 1048576
    except (OSError, KeyError):
        pass
    try:
        with open("/proc/loadavg") as f:
            info["load1"] = float(f.read().split()[0])
    except OSError:
        pass
    try:
        subprocess.run(["pgrep", "-x", "novacana-ml"], capture_output=True, check=True)
        info["training_running"] = True
    except subprocess.CalledProcessError:
        info["training_running"] = False
    return info


def collect():
    bc, rl = [], []
    sweep_dir = os.path.join(BASE, "learning_curve_out")
    logs_dir  = os.path.join(BASE, "logs")

    if os.path.isdir(sweep_dir):
        for fname in sorted(os.listdir(sweep_dir)):
            m = re.fullmatch(r"train_(\d+)\.log", fname)
            if not m:
                continue
            games = int(m.group(1))
            run = parse_log(os.path.join(sweep_dir, fname), f"{games} games")
            if not run:
                continue
            run["games"] = games
            run.update(parse_fidelity_log(os.path.join(sweep_dir, f"fidelity_{games}.log")))
            run["eval_winrate"] = parse_eval_log(os.path.join(sweep_dir, f"eval_{games}.log"))
            bc.append(run)
        bc.sort(key=lambda r: r["games"])

    if os.path.isdir(logs_dir):
        for fname in sorted(os.listdir(logs_dir)):
            if not fname.endswith(".log"):
                continue
            run = parse_log(os.path.join(logs_dir, fname), fname[:-4])
            if not run:
                continue
            (rl if run["kind"] == "rl" else bc).append(run)

    sys_ = sysinfo()
    # Current activity: the most recently modified run.
    activity = None
    all_runs = bc + rl
    if all_runs:
        latest = max(all_runs, key=lambda r: r["mtime"])
        if sys_.get("training_running"):
            if latest["kind"] == "rl":
                li = latest["iters"][-1]["iter"] if latest["iters"] else 0
                kind = "AZ" if latest.get("az") else "RL"
                activity = f"{kind} {latest['name']} - iter {li}/{latest['planned_iters']}"
            else:
                ph = latest["phase"]
                n = len(latest.get({"joint": "joint", "value": "value",
                                    "candidate": "cand"}.get(ph, "joint"), []))
                activity = f"BC {latest['name']} - {ph} {n and f'epoch {n}' or ''}"
        else:
            activity = "idle"
    return {"bc": bc, "rl": rl, "sys": sys_, "activity": activity, "ts": time.time()}


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Astra-6 Training</title>
<style>
:root {
  --surface:   #fcfcfb; --page: #f9f9f7;
  --ink:       #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid:      #e1e0d9; --axis: #c3c2b7;
  --border:    rgba(11,11,11,0.10);
  --good:      #006300;
  --s1: #2a78d6; --s2: #1baf7a; --s3: #eda100; --s4: #008300;
  --s5: #4a3aa7; --s6: #e34948; --s7: #e87ba4; --s8: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface:   #1a1a19; --page: #0d0d0d;
    --ink:       #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid:      #2c2c2a; --axis: #383835;
    --border:    rgba(255,255,255,0.10);
    --good:      #0ca30c;
    --s1: #3987e5; --s2: #199e70; --s3: #c98500; --s4: #008300;
    --s5: #9085e9; --s6: #e66767; --s7: #d55181; --s8: #d95926;
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 20px; max-width: 1060px; margin: 0 auto;
}
h1 { font-size: 17px; font-weight: 650; }
header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px; }
#status { font-size: 12px; color: var(--ink2); }
#status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
               background: var(--muted); margin-right: 5px; }
#status.live .dot { background: var(--good); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 10px; margin-bottom: 14px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 10px 14px; }
.tile .k { font-size: 11px; color: var(--muted); text-transform: uppercase;
           letter-spacing: .04em; }
.tile .v { font-size: 20px; font-weight: 650; margin-top: 2px; }
.tile .v small { font-size: 12px; font-weight: 400; color: var(--ink2); }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
@media (max-width: 820px) { .charts { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 12px 14px; position: relative; margin-bottom: 0; }
.card h2 { font-size: 12px; font-weight: 600; color: var(--ink2); margin-bottom: 6px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px;
          color: var(--ink2); margin-top: 6px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
              margin-right: 5px; vertical-align: -1px; }
svg { display: block; width: 100%; height: auto; }
svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
svg .dl { font-weight: 600; }
.tip { position: absolute; pointer-events: none; background: var(--surface);
       border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px;
       font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none;
       white-space: nowrap; z-index: 2; }
.tip b { font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: right; font-size: 11px; color: var(--muted); font-weight: 600;
     text-transform: uppercase; letter-spacing: .04em; padding: 6px 10px;
     border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid);
     font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td .ph { color: var(--muted); }
td.evals { white-space: normal; max-width: 280px; font-size: 12px; line-height: 1.7; }
#tbl_bc, #tbl_rl { overflow-x: auto; }
details { margin-top: 8px; }
details summary { font-size: 12px; color: var(--muted); cursor: pointer; padding: 4px 0; }
details table { margin-top: 4px; }
.stack { display: grid; gap: 10px; margin-bottom: 14px; }
footer { margin-top: 10px; font-size: 11px; color: var(--muted); }
</style></head><body>
<header>
  <h1>Astra-6 Training</h1>
  <div id="status"><span class="dot"></span><span id="stxt">connecting…</span></div>
</header>
<div class="tiles" id="tiles"></div>
<div class="charts">
  <div class="card"><h2>BC - held-out policy accuracy per epoch</h2>
    <div id="c_acc"></div><div class="legend" id="lg_acc"></div><div class="tip" id="t_acc"></div></div>
  <div class="card"><h2>BC - train policy cross-entropy per epoch</h2>
    <div id="c_pce"></div><div class="legend" id="lg_pce"></div><div class="tip" id="t_pce"></div></div>
  <div class="card"><h2>Candidate-head cross-entropy per epoch</h2>
    <div id="c_cand"></div><div class="legend" id="lg_cand"></div><div class="tip" id="t_cand"></div></div>
  <div class="card"><h2>RL - eval win rate vs iteration</h2>
    <div id="c_rl"></div><div class="legend" id="lg_rl"></div><div class="tip" id="t_rl"></div></div>
</div>
<div class="stack">
  <div class="card"><h2>BC runs</h2><div id="tbl_bc"></div></div>
  <div class="card"><h2>RL runs</h2><div id="tbl_rl"></div></div>
</div>
<footer id="foot"></footer>
<script>
const COLORS = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const fmt  = (v, d=3) => v == null ? '·' : v.toFixed(d);

// Shorten run names for labels: "8000 games" → "8K", strip the constant prefix
// (keep version prefixes - stripping them collided v3_rl and v4_rl into "rl").
function short(n) {
  const m = n.match(/^(\d+) games$/);
  if (m) return (+m[1] >= 1000 ? (+m[1] / 1000) + 'K' : m[1]);
  return n.replace(/^astra6_/, '').slice(0, 14);
}

function lineChart(el, tipEl, series, {ymin, ymax, yfmt, xlabel, refs}) {
  const W = 470, Hpx = 240, m = {t: 10, r: 78, b: 24, l: 40};
  const pw = W - m.l - m.r, ph = Hpx - m.t - m.b;
  series = series.filter(s => s.pts.length);
  if (!series.length) { el.innerHTML = '<svg viewBox="0 0 470 240"><text x="235" y="120" text-anchor="middle">no data yet</text></svg>'; return; }
  const allX = series.flatMap(s => s.pts.map(p => p[0]));
  const xmax = Math.max(...allX), xmin = Math.min(...allX);
  const allY = series.flatMap(s => s.pts.map(p => p[1]));
  (refs || []).forEach(r => allY.push(r.y));
  if (ymin == null) ymin = Math.min(...allY);
  if (ymax == null) ymax = Math.max(...allY);
  if (ymax - ymin < 1e-6) { ymax += 0.01; ymin -= 0.01; }
  const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
  const X = x => m.l + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * pw;
  const Y = y => m.t + (1 - (y - ymin) / (ymax - ymin)) * ph;
  let g = '';
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4, y = Y(v);
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    g += `<text x="${m.l - 6}" y="${y + 3.5}" text-anchor="end">${yfmt(v)}</text>`;
  }
  const xstep = Math.max(1, Math.ceil((xmax - xmin) / 8));
  for (let x = Math.ceil(xmin); x <= xmax; x += xstep)
    g += `<text x="${X(x)}" y="${Hpx - 8}" text-anchor="middle">${x}</text>`;
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${m.t + ph}" y2="${m.t + ph}" stroke="var(--axis)" stroke-width="1"/>`;
  (refs || []).forEach(r => {
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="var(--axis)" stroke-width="1" stroke-dasharray="4 4"/>`;
    g += `<text x="${W - m.r + 4}" y="${Y(r.y) + 3.5}">${r.label}</text>`;
  });
  series.forEach((s, i) => {
    const c = `var(${COLORS[i % COLORS.length]})`;
    const d = s.pts.map((p, j) => (j ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join('');
    g += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
    if (s.pts.length === 1) g += `<circle cx="${X(s.pts[0][0])}" cy="${Y(s.pts[0][1])}" r="3.5" fill="${c}"/>`;
    const lp = s.pts[s.pts.length - 1];
    g += `<circle cx="${X(lp[0])}" cy="${Y(lp[1])}" r="3" fill="${c}"/>`;
  });
  // End labels live in a right-hand gutter column and are pushed apart so they
  // can never overlap, even when the curves converge.
  const labs = series.map((s, i) => ({ i, name: s.name, y: Y(s.pts[s.pts.length - 1][1]) }))
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < labs.length; k++)
    if (labs[k].y - labs[k - 1].y < 13) labs[k].y = labs[k - 1].y + 13;
  for (const L of labs) {
    const c = `var(${COLORS[L.i % COLORS.length]})`;
    g += `<circle cx="${W - m.r + 8}" cy="${L.y - 3.5}" r="3" fill="${c}"/>`;
    g += `<text class="dl" x="${W - m.r + 15}" y="${L.y}" fill="var(--ink2)">${L.name}</text>`;
  }
  g += `<line id="xh" y1="${m.t}" y2="${m.t + ph}" stroke="var(--axis)" stroke-width="1" visibility="hidden"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${Hpx}">${g}</svg>`;
  const svg = el.querySelector('svg'), xh = svg.querySelector('#xh');
  svg.onmousemove = ev => {
    const r = svg.getBoundingClientRect();
    const mx = (ev.clientX - r.left) * W / r.width;
    const ep = Math.round(xmin + (mx - m.l) / pw * (xmax - xmin));
    if (ep < xmin || ep > xmax) { xh.setAttribute('visibility', 'hidden'); tipEl.style.display = 'none'; return; }
    xh.setAttribute('x1', X(ep)); xh.setAttribute('x2', X(ep)); xh.setAttribute('visibility', 'visible');
    let rows = `${xlabel || 'x'} ${ep}`;
    series.forEach((s, i) => {
      let best = null;
      for (const q of s.pts) if (best === null || Math.abs(q[0] - ep) < Math.abs(best[0] - ep)) best = q;
      if (best && Math.abs(best[0] - ep) <= Math.max(1, xstep))
        rows += `<br><span class="sw" style="background:var(${COLORS[i % COLORS.length]});display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px"></span>${s.name} (${best[0]}): <b>${yfmt(best[1])}</b>`;
    });
    tipEl.innerHTML = rows; tipEl.style.display = 'block';
    const cardR = el.parentElement.getBoundingClientRect();
    let tx = ev.clientX - cardR.left + 14;
    if (tx > cardR.width - 170) tx -= 190;
    tipEl.style.left = tx + 'px';
    tipEl.style.top = (ev.clientY - cardR.top + 8) + 'px';
  };
  svg.onmouseleave = () => { xh.setAttribute('visibility', 'hidden'); tipEl.style.display = 'none'; };
}

function legend(el, series) {
  const live = series.filter(s => s.pts.length);
  // The gutter labels already identify every series; the legend row only adds
  // value when some series got dropped from the gutter (none currently) or on
  // narrow screens. Keep it for ≥2 series per the accessibility rule.
  el.innerHTML = live.length < 2 ? '' : live.map((s, i) =>
    `<span><span class="sw" style="background:var(${COLORS[i % COLORS.length]})"></span>${s.name}</span>`).join('');
}

const tile = (k, v, sub) =>
  `<div class="tile"><div class="k">${k}</div><div class="v">${v}${sub ? ` <small>${sub}</small>` : ''}</div></div>`;
const pct = v => v == null ? '<span class="ph">·</span>' : v.toFixed(1) + '%';

async function refresh() {
  let d;
  try { d = await (await fetch('data.json')).json(); }
  catch { document.getElementById('stxt').textContent = 'server unreachable'; return; }
  const st = document.getElementById('status');
  st.classList.toggle('live', d.sys.training_running);
  document.getElementById('stxt').textContent = d.activity || (d.sys.training_running ? 'training' : 'idle');

  const rlSorted = d.rl.slice().sort((a, b) => a.mtime - b.mtime);
  const active   = rlSorted.length ? rlSorted[rlSorted.length - 1] : null;
  const prior    = rlSorted.slice(0, -1);
  const prevPeak = Math.max(0, ...prior.flatMap(r => r.evals.map(e => e.wr)));
  const lastWr   = active && active.evals.length ? active.evals[active.evals.length - 1].wr : null;
  const li0      = active && active.iters.length ? active.iters[active.iters.length - 1].iter : 0;
  document.getElementById('tiles').innerHTML =
    tile('Active run', active ? short(active.name) : '·', active ? `iter ${li0}/${active.planned_iters}` : '') +
    tile('Latest eval', lastWr != null ? lastWr.toFixed(1) + '%' : '·', 'parity 25%') +
    tile('Best val acc (BC)', Math.max(0, ...d.bc.map(r => r.best_acc || 0)).toFixed(3)) +
    tile('RAM available', d.sys.mem_avail_gb ? d.sys.mem_avail_gb.toFixed(1) : '?', 'GB') +
    tile('Load (1m)', d.sys.load1 != null ? d.sys.load1.toFixed(1) : '?', '/ 16');

  const sAcc = d.bc.map(r => ({ name: short(r.name),
    pts: r.joint.filter(e => e.val_acc != null).map(e => [e.epoch, e.val_acc]) }));
  const sPce = d.bc.map(r => ({ name: short(r.name), pts: r.joint.map(e => [e.epoch, e.policy]) }));
  const sCand = d.bc.map(r => ({ name: short(r.name), pts: r.cand.map(e => [e.epoch, e.loss]) }));
  // Chart only the ACTIVE run - twelve overlaid historical curves were noise.
  // Prior runs' high-water mark becomes a reference line instead.
  const sRl = active ? [{ name: short(active.name), pts: active.evals.map(e => [e.iter, e.wr]) }] : [];

  lineChart(document.getElementById('c_acc'), document.getElementById('t_acc'),
            sAcc, { yfmt: v => v.toFixed(3), xlabel: 'epoch' });
  lineChart(document.getElementById('c_pce'), document.getElementById('t_pce'),
            sPce, { ymin: 0, yfmt: v => v.toFixed(2), xlabel: 'epoch' });
  lineChart(document.getElementById('c_cand'), document.getElementById('t_cand'),
            sCand, { ymin: 0, yfmt: v => v.toFixed(2), xlabel: 'epoch' });
  const rlRefs = [{ y: 25, label: 'parity' }];
  if (prevPeak > 0) rlRefs.push({ y: prevPeak, label: 'prev peak' });
  lineChart(document.getElementById('c_rl'), document.getElementById('t_rl'),
            sRl, { yfmt: v => v.toFixed(1) + '%', xlabel: 'iter', refs: rlRefs });
  legend(document.getElementById('lg_acc'), sAcc);
  legend(document.getElementById('lg_pce'), sPce);
  legend(document.getElementById('lg_cand'), sCand);
  legend(document.getElementById('lg_rl'), sRl);

  document.getElementById('tbl_bc').innerHTML =
    `<table><tr><th>Run</th><th>Samples</th><th>Best val acc</th><th>Cand CE</th>
      <th>Choose agree</th><th>Top-2</th><th>Respond</th><th>Eval win</th><th>Phase</th></tr>` +
    d.bc.map(r => `<tr><td>${short(r.name)}${r.warm_start ? ' <span class="ph">(warm)</span>' : ''}</td>
      <td>${r.gen_samples ? r.gen_samples.toLocaleString() : '<span class="ph">·</span>'}</td>
      <td>${r.best_acc ? fmt(r.best_acc, 4) : '<span class="ph">·</span>'}</td>
      <td>${r.cand.length ? fmt(r.cand[r.cand.length - 1].loss, 3) : '<span class="ph">·</span>'}</td>
      <td>${pct(r.choose_agree)}</td><td>${pct(r.choose_top2)}</td>
      <td>${pct(r.respond_agree)}</td><td>${pct(r.eval_winrate)}</td>
      <td>${r.phase === 'done' ? '✓' : r.phase}</td></tr>`).join('') + '</table>';

  const rlRow = r => {
    const li = r.iters.length ? r.iters[r.iters.length - 1] : null;
    const spi = r.iters.length ? (r.iters.reduce((a, x) => a + x.secs, 0) / r.iters.length).toFixed(1) : '·';
    const best = r.evals.length ? Math.max(...r.evals.map(e => e.wr)) : null;
    const evs = r.evals.map(e => `<span class="ph">${e.iter}&hairsp;→</span>&thinsp;${e.wr.toFixed(1)}%`).join(' &nbsp; ') || '<span class="ph">·</span>';
    const status = r.crashed ? 'crashed' : r.complete ? '✓' : (li ? `iter ${li.iter}/${r.planned_iters}` : 'starting');
    return `<tr><td>${short(r.name)}${r.az ? ' <span class="ph">(AZ)</span>' : ''}</td>
      <td>${r.iters.length}</td><td>${spi}</td>
      <td>${best != null ? best.toFixed(1) + '%' : '·'}</td>
      <td class="evals">${evs}</td><td>${status}</td></tr>`;
  };
  const recent = rlSorted.slice(-3).reverse();
  const older  = rlSorted.slice(0, -3).reverse();
  const hdr = `<tr><th>Run</th><th>Iters</th><th>s/iter</th><th>Best eval</th><th>Evals</th><th>Status</th></tr>`;
  let rlHtml = `<table>${hdr}${recent.map(rlRow).join('')}</table>`;
  if (older.length) {
    const open = window._rlArchOpen ? ' open' : '';
    rlHtml += `<details id="rl_arch"${open}><summary>${older.length} archived runs</summary>
      <table>${hdr}${older.map(rlRow).join('')}</table></details>`;
  }
  document.getElementById('tbl_rl').innerHTML = rlHtml;
  const arch = document.getElementById('rl_arch');
  if (arch) arch.ontoggle = () => { window._rlArchOpen = arch.open; };

  document.getElementById('foot').textContent =
    'updated ' + new Date(d.ts * 1000).toLocaleTimeString();
}
refresh();
setInterval(refresh, 3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif self.path == "/data.json":
            body = json.dumps(collect()).encode()
            ctype = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7750)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard → http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
