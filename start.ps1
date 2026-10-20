$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $studioPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $studioPython)) {
        throw 'Run setup.ps1 first (Python 3.12 and Internet required).'
    }
    $env:PYTHONNOUSERSITE = '1'
    $env:MPLBACKEND = 'Agg'
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib-cache'
    & $studioPython .\sionna-rt-gui\scripts\run.py --config .\polyu-stylized-config.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Studio exited with an error. See the traceback above.' }
} finally { Pop-Location }
