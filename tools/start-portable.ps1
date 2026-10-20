$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $studioPython = Join-Path $PSScriptRoot 'runtime\python.exe'
    if (-not (Test-Path -LiteralPath $studioPython)) {
        throw 'The bundled runtime is missing. Extract the entire ZIP before running start.bat.'
    }
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = '1'
    $env:MPLBACKEND = 'Agg'
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib-cache'
    & $studioPython -I .\sionna-rt-gui\scripts\run.py --config .\polyu-stylized-config.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Studio exited with an error. See the traceback above.' }
} catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally { Pop-Location }
