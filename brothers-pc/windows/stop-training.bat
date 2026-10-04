@echo off
REM Astra-7 training - graceful stop. Double-click to run.
REM
REM Sends SIGTERM to the training process (the same signal Ctrl+C sends).
REM The Rust process catches it, finishes the batch it's currently on,
REM saves a checkpoint + resume-metadata sidecar, and exits cleanly -
REM verified end-to-end (2026-07-15) to take well under 2 seconds even
REM mid-self-play-generation. This is NOT a hard kill: closing the Ubuntu
REM window or shutting down WSL without running this first can lose up to
REM --checkpoint-every batches of progress (a few minutes at most, see
REM launch-bc.sh's comments) rather than being lost entirely, but this
REM script is still the clean way to pause.
REM
REM After this finishes, start-training.bat will automatically resume
REM from the saved checkpoint next time it's run.

setlocal
set DISTRO=Ubuntu-22.04

echo Sending stop signal to training...
wsl -d %DISTRO% -- bash -lc "pkill -TERM -f target/release/novacana-ml7"
if %ERRORLEVEL% NEQ 0 (
    echo No training process was running.
    pause
    exit /b 0
)

echo Waiting for it to save a checkpoint and exit...
:waitloop
wsl -d %DISTRO% -- bash -lc "pgrep -f target/release/novacana-ml7 >/dev/null"
if %ERRORLEVEL% EQU 0 (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo Stopped cleanly. Checkpoint saved - safe to close everything, use the
echo PC for something else, and resume later with start-training.bat.
pause
