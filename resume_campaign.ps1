# IC-PINOT Autonomous 24/7 Lab Runner with Power-Cut Recovery
# Automatically resumes execution from the exact dataset, seed, and pillar where interrupted.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = "C:\ProgramData\anaconda3\python.exe"

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "   IC-PINOT: Autonomous Resilient 24/7 Training Engine" -ForegroundColor Green
Write-Host "   Power-Cut and Crash-Resistant Auto-Resume Protocol" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan

while ($true) {
    $Now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Now] Running / Resuming Master Campaign..." -ForegroundColor Cyan

    & $PythonExe -u run_master_campaign.py --all --epochs 300 --seeds 42 1337 2026

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n===============================================================================" -ForegroundColor Green
        Write-Host "   ALL 5 PILLARS COMPLETED SUCCESSFULLY! ALL ARTIFACTS SAVED IN results/" -ForegroundColor Green
        Write-Host "===============================================================================" -ForegroundColor Green
        break
    } else {
        Write-Host "`n[WARNING] Process exited with code $LASTEXITCODE. Waiting 10s before auto-resuming..." -ForegroundColor Red
        Start-Sleep -Seconds 10
    }
}
