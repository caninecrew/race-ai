<# Simple bootstrap script to create .venv and install requirements. #>

param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$venvPath = ".venv"

try {
    Get-Command $Python | Out-Null
} catch {
    Write-Error "Python is not available on PATH."
    exit 1
}

if (-not (Test-Path $venvPath)) {
    & $Python -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts/python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "Virtual environment ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
