@echo off
REM Astra-7 training - start/resume. Double-click to run.
REM
REM Launches (or attaches to, if already running) a tmux session inside
REM WSL2/Ubuntu that runs launch-bc.sh. Because it's tmux, closing this
REM window or even closing Ubuntu entirely does NOT stop training - it
REM keeps running in the background. Use stop-training.bat to actually
REM pause it (which saves a checkpoint first).

setlocal
set DISTRO=Ubuntu-22.04

echo Checking for an already-running training session...
wsl -d %DISTRO% -- bash -lc "tmux has-session -t astra7 2>/dev/null"
if %ERRORLEVEL% EQU 0 (
    echo Already running - attaching to it. Press Ctrl+b then d to detach
    echo without stopping training ^(NOT Ctrl+c, that would kill it^).
    wsl -d %DISTRO% -- bash -lc "tmux attach -t astra7"
) else (
    echo Starting a fresh training session...
    wsl -d %DISTRO% -- bash -lc "tmux new-session -d -s astra7 'bash ~/novacana-engine/brothers-pc/linux/launch-bc.sh 2>&1 | tee -a ~/novacana-engine/training.log'"
    echo Started. Training is now running in the background even if you
    echo close this window. Run this script again any time to watch live
    echo progress ^(Ctrl+b then d to detach without stopping it^).
)
pause
