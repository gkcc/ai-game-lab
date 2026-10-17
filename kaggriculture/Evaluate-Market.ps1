param([int[]]$Seeds = @(0, 1, 7, 42, 2026, 314159))
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\market_evaluate.py" --seeds $Seeds
if ($LASTEXITCODE -ne 0) { throw "Market evaluation failed (exit $LASTEXITCODE). See artifact directory." }
