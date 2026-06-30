$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $ProjectRoot "backend"
$PythonExe = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating backend virtual environment..."
    Push-Location $BackendDir
    try {
        python -m venv .venv
    }
    finally {
        Pop-Location
    }
}

Push-Location $BackendDir
try {
    & $PythonExe -m pip install -e ".[dev]"
    & $PythonExe -m app.dev_check
    & $PythonExe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
