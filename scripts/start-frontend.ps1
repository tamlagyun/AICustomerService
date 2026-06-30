$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $ProjectRoot "frontend"
$NodeModulesDir = Join-Path $FrontendDir "node_modules"
$EnvFile = Join-Path $ProjectRoot ".env"

if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $name, $value = $line.Split("=", 2)
        if ($name -match "^(FRONTEND_HOST|FRONTEND_PORT|BACKEND_ORIGIN)$") {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm is missing. Install Node.js before starting the frontend." -ForegroundColor Red
    exit 1
}

Push-Location $FrontendDir
try {
    if (-not (Test-Path -LiteralPath $NodeModulesDir)) {
        npm install
    }

    $HostValue = if ($env:FRONTEND_HOST) { $env:FRONTEND_HOST } else { "127.0.0.1" }
    $PortValue = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "5173" }
    Write-Host "Starting frontend on ${HostValue}:${PortValue}"
    npm run dev
}
finally {
    Pop-Location
}
