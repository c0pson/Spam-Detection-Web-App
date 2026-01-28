# PowerShell script to run tests before starting the Flask app
# Usage: .\run_with_tests.ps1

Write-Host "=" * 70
Write-Host "Activating virtual environment and running test suite..."
Write-Host "=" * 70

# Activate the virtual environment
. .\.venv\Scripts\Activate.ps1

# Run pytest
Write-Host "`nRunning pytest..." -ForegroundColor Cyan
python -m pytest -q

# Check if tests passed
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nTests failed! Application startup aborted." -ForegroundColor Red
    exit 1
}

Write-Host "`n" + ("=" * 70)
Write-Host "All tests passed! Starting Flask application..." -ForegroundColor Green
Write-Host "=" * 70 + "`n"

# Start the Flask app
python -m spam_detection
