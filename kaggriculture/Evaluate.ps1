param(
    [int[]]$Seeds = @(0, 1, 7, 42, 2026, 314159),
    [string]$Bot = 'bots/heuristic_v2.py',
    [string]$Baseline = 'bots/heuristic_v1.py',
    [string]$Opponent = 'starter'
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\evaluate.py" --bot $Bot --baseline $Baseline --opponent $Opponent --seeds @Seeds
exit $LASTEXITCODE
