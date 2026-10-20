$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $studioPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $studioReady = Join-Path $PSScriptRoot '.venv\.studio-ready'
    if (-not (Test-Path -LiteralPath $studioPython) -or -not (Test-Path -LiteralPath $studioReady)) {
        Write-Host 'First-run setup: installing dependencies. Internet access is required.'
        & (Join-Path $PSScriptRoot 'setup.ps1')
        if (-not (Test-Path -LiteralPath $studioReady)) {
            throw 'Setup did not complete. Check the error above and run start.bat again.'
        }
    }
    $env:PYTHONNOUSERSITE = '1'
    $env:MPLBACKEND = 'Agg'
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib-cache'
    & $studioPython .\sionna-rt-gui\scripts\run.py --config .\polyu-stylized-config.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Studio exited with an error. See the traceback above.' }
} catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally { Pop-Location }
