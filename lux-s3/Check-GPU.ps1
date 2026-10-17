param([ValidateSet(1,4,8,16,32,64)][int]$Batch=4, [ValidateSet(32,64,128,256)][int]$Steps=32)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Wsl-Lux.ps1"
$linuxRoot = ConvertTo-LuxWslPath -Value $PSScriptRoot
Invoke-LuxGpuPython -Script 'check_backend.py' -ScriptArguments @('--require-gpu', '--batch', "$Batch", '--steps', "$Steps", '--progress', '--output', "$linuxRoot/artifacts/backend-gpu.json")
