$ErrorActionPreference = 'Stop'
if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    py -3.12 -m venv "$PSScriptRoot\.venv"
    if ($LASTEXITCODE) { throw 'Could not create isolated game venv' }
}
& "$PSScriptRoot\.venv\Scripts\python.exe" -B "$PSScriptRoot\..\tools\bootstrap_upstream.py" --target kaggriculture
if ($LASTEXITCODE) { throw 'Official source bootstrap failed' }
& "$PSScriptRoot\.venv\Scripts\python.exe" -m pip install -r "$PSScriptRoot\requirements-lock.txt"
if ($LASTEXITCODE) { throw 'Game dependency installation failed' }
& "$PSScriptRoot\.venv\Scripts\python.exe" -m pip check
if ($LASTEXITCODE) { throw 'Game dependency check failed' }
