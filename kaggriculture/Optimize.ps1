param([ValidateRange(5,32)][int]$Trials = 8)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\learning\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\learning\optimize.py" --trials $Trials
if ($LASTEXITCODE -ne 0) { throw "Optimization failed (exit $LASTEXITCODE). Inspect printed artifacts." }
