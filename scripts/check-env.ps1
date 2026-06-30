$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $ProjectRoot "backend"
$PythonExe = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Backend virtual environment is missing. Run scripts/start-backend.ps1 first." -ForegroundColor Yellow
    exit 1
}

Push-Location $BackendDir
try {
    & $PythonExe -m app.dev_check
}
finally {
    Pop-Location
}
