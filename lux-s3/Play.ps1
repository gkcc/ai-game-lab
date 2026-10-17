param([string]$Bot='bot/main.py', [string]$Opponent='bot/main.py', [int]$Seed=42, [ValidateSet('json','html')][string]$Format='json', [ValidateSet('CPU','GPU')][string]$Backend='CPU')
$ErrorActionPreference = 'Stop'
if ($Backend -eq 'GPU') {
    . "$PSScriptRoot\Wsl-Lux.ps1"
    $gpuBot = ConvertTo-LuxWslPath -Value $Bot
    $gpuOpponent = ConvertTo-LuxWslPath -Value $Opponent
    Invoke-LuxGpuPython -Script 'run_match.py' -ScriptArguments @('--bot', $gpuBot, '--opponent', $gpuOpponent,
        '--seed', "$Seed", '--format', $Format, '--require-gpu')
    exit 0
}
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -X utf8 "$PSScriptRoot\run_match.py" --bot $Bot --opponent $Opponent --seed $Seed --format $Format
exit $LASTEXITCODE
