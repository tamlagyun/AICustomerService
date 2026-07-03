$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $ProjectRoot "backend"
$PythonExe = Join-Path $BackendDir ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $name, $value = $line.Split("=", 2)
        if ($name -match "^(APP_HOST|APP_PORT)$") {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

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
    $HostValue = if ($env:APP_HOST) { $env:APP_HOST } else { "127.0.0.1" }
    $PortValue = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }
    Write-Host "Starting backend on ${HostValue}:${PortValue}"
    & $PythonExe -m uvicorn app.main:app --reload --host $HostValue --port $PortValue
}
finally {
    Pop-Location
}
