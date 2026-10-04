# Astra-7 brother's-PC bootstrap, step 1 of 2 (Windows side).
#
# Run this ONCE, in an ELEVATED PowerShell (right-click PowerShell -> "Run
# as administrator"). It installs WSL2 + Ubuntu, which is where the actual
# Rust/CUDA build and training happen - the CUDA-linking fix this project
# needs (ml7/build.rs, ASTRA7.md §5) relies on GNU-ld-style `-Wl,` link
# args that only make sense on Linux, so this can't be built with native
# Windows MSVC. WSL2 + GPU passthrough gives Linux userspace with full
# access to the RTX 3080 without dual-booting.
#
# Usage:  powershell -ExecutionPolicy Bypass -File setup-wsl2.ps1

$ErrorActionPreference = "Stop"

function Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

Section "Checking for admin rights"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script must be run as Administrator. Right-click PowerShell -> Run as administrator, then re-run this script." -ForegroundColor Red
    exit 1
}

Section "Checking NVIDIA driver (Windows side)"
# WSL2 CUDA passthrough uses the WINDOWS driver via libcuda passthrough -
# do NOT install a separate NVIDIA driver inside WSL2/Ubuntu later, that's
# a common source of "CUDA not found" confusion. Only the Windows driver
# needs to be current.
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "nvidia-smi found, driver info:"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
    Write-Host ""
    Write-Host "If the driver version is older than ~560.x, update it from" -ForegroundColor Yellow
    Write-Host "https://www.nvidia.com/Download/index.aspx (RTX 3080 -> Game Ready or Studio driver)" -ForegroundColor Yellow
    Write-Host "before continuing - CUDA 12.6 (what this project targets) needs a reasonably recent driver."
} else {
    Write-Host "nvidia-smi not found on PATH. Install/update the NVIDIA driver from" -ForegroundColor Red
    Write-Host "https://www.nvidia.com/Download/index.aspx before continuing." -ForegroundColor Red
    Write-Host "(Pick the RTX 3080, Game Ready or Studio driver - either works.)"
}

Section "Installing WSL2 + Ubuntu 22.04"
# wsl --install pulls the WSL2 kernel, sets WSL2 as default, and installs
# the named distro in one shot on a modern (post-2021) Windows 10/11.
wsl --install -d Ubuntu-22.04

Write-Host ""
Write-Host "If this is the FIRST TIME WSL has been installed on this machine, Windows" -ForegroundColor Yellow
Write-Host "likely needs a REBOOT now. After rebooting, Ubuntu will finish setup and" -ForegroundColor Yellow
Write-Host "ask you to create a Linux username/password on its own - do that, then" -ForegroundColor Yellow
Write-Host "come back to this folder and continue with the README's step 2." -ForegroundColor Yellow
Write-Host ""
Write-Host "If Ubuntu opened a window and asked for a username/password just now," -ForegroundColor Yellow
Write-Host "no reboot was needed - go ahead and continue with the README's step 2." -ForegroundColor Yellow

Section "Next: copy .wslconfig"
Write-Host "Copy windows\dot-wslconfig to C:\Users\<your-username>\.wslconfig"
Write-Host "(this caps how much of the 16GB host RAM WSL2 is allowed to use, so"
Write-Host "Windows itself doesn't get starved while training runs)."
Write-Host ""
Write-Host "Then run:  wsl --shutdown   (from any PowerShell, to apply it)"
Write-Host "Then open Ubuntu from the Start menu and continue with the README's step 2."
