@echo off
REM Microgrid Project Launcher
REM This script runs the project using the virtual environment

echo ========================================
echo Microgrid Forecasting Project
echo ========================================
echo.

REM Run using virtual environment Python
".venv\Scripts\python.exe" main.py

echo.
echo ========================================
echo Press any key to exit...
pause >nul
