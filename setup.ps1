$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit), then retry.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -c requirements-tested.txt -e .\sionna-rt-gui
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Review the error above.' }
    Write-Host 'Ready. Run start.bat.'
} finally { Pop-Location }
