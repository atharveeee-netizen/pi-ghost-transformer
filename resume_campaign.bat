@echo off
title IC-PINOT Autonomous 24/7 Lab Runner with Power-Cut Recovery
echo ===============================================================================
echo   IC-PINOT: Autonomous Resilient 24/7 Training Engine
echo   Power-Cut and Crash-Resistant Auto-Resume Protocol
echo ===============================================================================
cd /d "%~dp0"

:LOOP
echo [%DATE% %TIME%] Starting / Resuming IC-PINOT Master Campaign...
"C:\ProgramData\anaconda3\python.exe" -u run_master_campaign.py --all --epochs 300 --seeds 42 1337 2026

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo   ALL 5 PILLARS COMPLETED SUCCESSFULLY! RESULTS PERMANENTLY SAVED.
    echo ===============================================================================
    pause
    exit /b 0
) else (
    echo.
    echo [WARNING] Execution interrupted (Power cut, error, or reboot).
    echo Automatically waiting 10 seconds before resuming from exact checkpoint...
    timeout /t 10 /nobreak >nul
    goto LOOP
)
