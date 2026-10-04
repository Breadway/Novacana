@echo off
REM Astra-7 dashboard - local mode. Double-click to run.
REM
REM Launches (or attaches to) a tmux session inside WSL2/Ubuntu running
REM dashboard_server.py directly on this machine (ASTRA7_LOCAL=1 - reads
REM the training log/GPU/checkpoint files locally instead of over SSH).
REM Binds 0.0.0.0, so it's reachable two ways at once:
REM   - on this PC:        http://localhost:8420
REM   - from the tailnet:  http://<this machine's tailscale ip>:8420
REM         (run "tailscale ip -4" in the Ubuntu window to find it, or
REM         check `tailscale status` from hermes/hestia once joined)
REM Independent of start-training.bat - can run whether or not training is
REM currently active (shows a "no training process" status either way).

setlocal
set DISTRO=Ubuntu-22.04

echo Checking for an already-running dashboard session...
wsl -d %DISTRO% -- bash -lc "tmux has-session -t astra7-dash 2>/dev/null"
if %ERRORLEVEL% EQU 0 (
    echo Already running - attaching to it. Press Ctrl+b then d to detach
    echo without stopping the dashboard ^(NOT Ctrl+c, that would kill it^).
    wsl -d %DISTRO% -- bash -lc "tmux attach -t astra7-dash"
) else (
    echo Starting the dashboard...
    wsl -d %DISTRO% -- bash -lc "tmux new-session -d -s astra7-dash 'ASTRA7_LOCAL=1 python3 ~/novacana-engine/tools/dashboard_server.py'"
    echo Started. Open http://localhost:8420 in a browser on this PC.
    echo For the tailnet URL, run: wsl -d %DISTRO% -- bash -lc "tailscale ip -4"
)
pause
