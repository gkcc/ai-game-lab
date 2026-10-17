$ErrorActionPreference = 'Stop'
if (-not (Test-Path "$PSScriptRoot\wsl-config.json")) { throw 'Copy and configure wsl-config.example.json locally first' }
if (-not (Test-Path "$PSScriptRoot\artifacts\setup\luxai_s3-0.2.1-py3-none-any.whl")) { throw 'Run the CPU source-wheel setup first' }
. "$PSScriptRoot\Wsl-Lux.ps1"
$config = Get-Content -Raw "$PSScriptRoot\wsl-config.json" | ConvertFrom-Json
$linuxRoot = ConvertTo-LuxWslPath -Value $PSScriptRoot
$venvRoot = $config.python.Substring(0, $config.python.Length - '/bin/python'.Length)
Invoke-LuxWsl -LinuxArguments @('python3', '-m', 'venv', $venvRoot)
Invoke-LuxWsl -LinuxArguments @($config.python, '-m', 'pip', 'install', '--require-hashes', '--no-deps', '-r', "$linuxRoot/uv-bootstrap-hashed.txt")
Invoke-LuxWsl -LinuxArguments @("$venvRoot/bin/uv", 'pip', 'install', '--require-hashes', '--python', $config.python, '-r', "$linuxRoot/requirements-gpu-hashed.txt")
Invoke-LuxWsl -LinuxArguments @($config.python, '-m', 'pip', 'install', '--no-deps', "$linuxRoot/artifacts/setup/luxai_s3-0.2.1-py3-none-any.whl")
Invoke-LuxWsl -LinuxArguments @($config.python, '-B', "$linuxRoot/../tools/verify_lux_source.py", '--gpu')
Invoke-LuxWsl -LinuxArguments @($config.python, '-m', 'pip', 'check')
