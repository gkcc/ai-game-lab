$ErrorActionPreference = 'Stop'
if (-not (Test-Path "$PSScriptRoot\learning\.venv\Scripts\python.exe")) {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -B -m venv "$PSScriptRoot\learning\.venv"
    if ($LASTEXITCODE) { throw 'Could not create learning venv' }
}
& "$PSScriptRoot\learning\.venv\Scripts\python.exe" -m pip install -r "$PSScriptRoot\learning\requirements-lock.txt" --disable-pip-version-check
if ($LASTEXITCODE) { throw 'Could not install isolated optimizer requirements' }
& "$PSScriptRoot\learning\.venv\Scripts\python.exe" -m pip check
if ($LASTEXITCODE) { throw 'Learning dependency check failed' }
