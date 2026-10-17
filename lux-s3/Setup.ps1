$ErrorActionPreference = 'Stop'
$python = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3.12 -m venv "$PSScriptRoot\.venv"
    if ($LASTEXITCODE) { throw 'Could not create isolated Lux venv' }
}
& $python -B "$PSScriptRoot\..\tools\bootstrap_upstream.py" --target lux
if ($LASTEXITCODE) { throw 'Official Lux source bootstrap failed' }
& $python -m pip install -r "$PSScriptRoot\requirements-lock.txt"
if ($LASTEXITCODE) { throw 'Lux dependency installation failed' }
& $python -m pip wheel --no-deps --wheel-dir "$PSScriptRoot\artifacts\setup" "$PSScriptRoot\upstream\src"
if ($LASTEXITCODE) { throw 'Pinned official Lux source wheel build failed' }
& $python -m pip install --no-deps --force-reinstall "$PSScriptRoot\artifacts\setup\luxai_s3-0.2.1-py3-none-any.whl"
if ($LASTEXITCODE) { throw 'Official Lux wheel installation failed' }
& $python -B "$PSScriptRoot\..\tools\verify_lux_source.py"
if ($LASTEXITCODE) { throw 'Official Lux source verification failed' }
& $python -m pip check
if ($LASTEXITCODE) { throw 'Lux dependency check failed' }
