
@echo off
cd /d "%~dp0"
title Hotspot Hunter - AI 2
color 07

set "PYTHON=C:\Users\debae\OneDrive - Hogeschool VIVES\drone applications\jaar 2\challenge 3\herkansing\wildfire_YOLO\.venv\Scripts\python.exe"

if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python-omgeving niet gevonden.
    pause
    exit /b 1
)

:menu
cls
echo.
echo ==========================================
echo              HOTSPOT HUNTER
echo ==========================================
echo.
echo Brecht de Baets
echo Yoren Dedeurwaerder
echo Drone Applications - AI 2
echo.
echo RGB ANALYSIS
echo.
echo  1. Live YOLO demonstratie
echo  2. Analyseer alle videos
echo  3. Open originele videos
echo  4. Open RGB resultaten
echo.
echo RESEARCH
echo.
echo  5. Start resolutie-experiment
echo  6. Open experimentresultaten
echo.
echo THERMAL ANALYSIS
echo.
echo  7. Start thermische analyse
echo  8. Open thermische resultaten
echo.
echo PROJECT
echo.
echo  9. Open projectmap
echo 10. Open Visual Studio Code
echo 11. Open Cheat Sheet
echo 12. Afsluiten
echo.
echo ==========================================
echo.

set "choice="
set /p choice=Maak een keuze:

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto batch
if "%choice%"=="3" goto videos
if "%choice%"=="4" goto results
if "%choice%"=="5" goto experiment
if "%choice%"=="6" goto experiment_results
if "%choice%"=="7" goto thermal
if "%choice%"=="8" goto thermal_results
if "%choice%"=="9" goto folder
if "%choice%"=="10" goto vscode
if "%choice%"=="11" goto sheet
if "%choice%"=="12" exit /b

goto menu

:demo
"%PYTHON%" hotspot_hunter.py
pause
goto menu

:batch
"%PYTHON%" hotspot_hunter.py --all
pause
goto menu

:videos
explorer "%~dp0demo_video"
goto menu

:results
explorer "%~dp0AI_resultaten"
goto menu

:experiment
"%PYTHON%" experiment.py
pause
goto menu

:experiment_results
explorer "%~dp0AI_resultaten\experiment"
goto menu

:thermal
"%PYTHON%" thermal.py
pause
goto menu

:thermal_results
explorer "%~dp0AI_resultaten\thermal"
goto menu

:folder
explorer "%~dp0"
goto menu

:vscode
call code "%~dp0."
goto menu

:sheet
start "" notepad.exe "%~dp0Sheet.txt"
goto menu
