param([string]$Bot = 'main.py', [string]$Opponent = 'starter', [int]$Seed = 42, [ValidateSet(0,1)][int]$Side = 0)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\run_match.py" --bot $Bot --opponent $Opponent --seed $Seed --side $Side
exit $LASTEXITCODE
