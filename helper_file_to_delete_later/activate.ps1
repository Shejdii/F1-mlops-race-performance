# activate.ps1
# Aktywuje globalne środowisko template'u dla tego projektu

$venv = "D:\Kurs\MLOps\VS\mlops-template-GPU\.venv\Scripts\Activate.ps1"

if (Test-Path $venv) {
    & $venv
    Write-Host "`n[OK] Template venv activated:"
    python -c "import sys; print(sys.executable)"
} else {
    Write-Host "[ERROR] Interpreter not found:"
    Write-Host $venv
}

