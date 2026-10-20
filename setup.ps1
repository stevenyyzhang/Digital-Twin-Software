$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $studioReady = Join-Path $PSScriptRoot '.venv\.studio-ready'
    if (Test-Path -LiteralPath $studioReady) { Remove-Item -LiteralPath $studioReady }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Install Python 3.12 (64-bit), including the Python Launcher, then run start.bat again.'
        }
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit), then retry.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -c requirements-tested.txt -e .\sionna-rt-gui
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Review the error above.' }
    Set-Content -LiteralPath $studioReady -Value 'Dependencies installed successfully.' -Encoding ascii
    Write-Host 'Setup complete.'
} finally { Pop-Location }
