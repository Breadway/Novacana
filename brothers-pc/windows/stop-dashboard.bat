@echo off
REM Astra-7 dashboard - stop. Double-click to run.
REM
REM Just kills the tmux session; the dashboard has no state to save (it
REM only reads the training log/checkpoint files, never writes them), so
REM this is safe to run any time, independent of training's own state.

setlocal
set DISTRO=Ubuntu-22.04

wsl -d %DISTRO% -- bash -lc "tmux kill-session -t astra7-dash 2>/dev/null"
echo Dashboard stopped.
pause
