param([int[]]$Seeds = @(0, 1, 7, 42, 2026, 314159))
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\crossplay.py" --seeds $Seeds
if ($LASTEXITCODE -ne 0) { throw "Crossplay failed (exit $LASTEXITCODE). See printed artifact directory." }
