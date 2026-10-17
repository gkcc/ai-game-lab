function Invoke-LuxWsl {
    param([string[]]$LinuxArguments)
    $config = Get-Content -Raw -LiteralPath "$PSScriptRoot\wsl-config.json" | ConvertFrom-Json
    & wsl.exe -d $config.distro -u $config.user --exec @LinuxArguments
    if ($LASTEXITCODE) { throw "WSL command failed with exit code $LASTEXITCODE" }
}

function ConvertTo-LuxWslPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        $converted = Invoke-LuxWsl -LinuxArguments @('wslpath', '-a', '-u', $Value)
        return ($converted -join "`n").Trim()
    }
    return $Value.Replace('\', '/')
}

function Invoke-LuxGpuPython {
    param([string]$Script, [string[]]$ScriptArguments)
    $config = Get-Content -Raw -LiteralPath "$PSScriptRoot\wsl-config.json" | ConvertFrom-Json
    $linuxRoot = ConvertTo-LuxWslPath -Value $PSScriptRoot
    $arguments = @('env', 'JAX_PLATFORMS=cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
        "JAX_COMPILATION_CACHE_DIR=$($config.compilation_cache)", 'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1', 'PYTHONUTF8=1', $config.python, '-u', '-B', "$linuxRoot/$Script")
    Invoke-LuxWsl -LinuxArguments ($arguments + $ScriptArguments)
}
