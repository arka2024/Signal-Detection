@echo off
REM SUMO Installation and Setup Script for Windows

echo ====================================================
echo SUMO Installation and Setup Script
echo ====================================================
echo.

REM Check if SUMO is already installed
if defined SUMO_HOME (
    echo SUMO_HOME is already set to: %SUMO_HOME%
    echo Checking if directory exists...
    if exist "%SUMO_HOME%" (
        echo SUMO appears to be already installed.
        echo You can skip installation if SUMO is working correctly.
    ) else (
        echo SUMO_HOME is set but directory does not exist.
    )
    echo.
)

echo This script will help you install SUMO (Simulation of Urban MObility).
echo.
echo Options:
echo 1. Download and install SUMO (requires administrator privileges)
echo 2. Set SUMO_HOME environment variable (after installation)
echo 3. Test SUMO installation
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Downloading SUMO...
    echo.
    echo Please visit: https://sumo.dlr.de/docs/Downloads.php
    echo Download the Windows installer and run it.
    echo.
    echo After installation is complete, return to this script and select option 2.
    pause
    goto :eof
)

if "%choice%"=="2" (
    echo.
    echo Setting SUMO_HOME environment variable...
    echo.
    echo Common SUMO installation paths:
    echo 1. C:\Program Files (x86)\Eclipse\Sumo
    echo 2. C:\Program Files\Eclipse\Sumo
    echo 3. Enter custom path
    echo.
    
    set /p path_choice="Enter your choice (1-3): "
    
    if "%path_choice%"=="1" (
        set SUMO_PATH=C:\Program Files (x86)\Eclipse\Sumo
    ) else if "%path_choice%"=="2" (
        set SUMO_PATH=C:\Program Files\Eclipse\Sumo
    ) else if "%path_choice%"=="3" (
        set /p SUMO_PATH="Enter full path to SUMO installation: "
    ) else (
        echo Invalid choice.
        goto :eof
    )
    
    echo.
    echo Setting SUMO_HOME to: %SUMO_PATH%
    
    REM Set for current session
    set SUMO_HOME=%SUMO_PATH%
    
    REM Set permanently (requires admin privileges)
    echo.
    echo To set SUMO_HOME permanently, this script needs admin privileges.
    echo If you get an access denied error, please run this script as administrator.
    echo.
    setx SUMO_HOME "%SUMO_PATH%" /m
    
    echo.
    echo SUMO_HOME environment variable has been set.
    echo Adding SUMO to PATH...
    
    REM Add SUMO bin to PATH
    set PATH=%PATH%;%SUMO_HOME%\bin
    
    echo Done.
    echo.
    pause
)

if "%choice%"=="3" (
    echo.
    echo Testing SUMO installation...
    echo.
    
    if not defined SUMO_HOME (
        echo SUMO_HOME is not set. Please set it first using option 2.
        pause
        goto :eof
    )
    
    echo Testing sumo-gui...
    where sumo-gui > nul 2>&1
    if %errorlevel% equ 0 (
        echo sumo-gui found in PATH.
    ) else (
        echo sumo-gui not found in PATH. Checking in SUMO_HOME...
        if exist "%SUMO_HOME%\bin\sumo-gui.exe" (
            echo sumo-gui.exe found in SUMO_HOME\bin.
            echo Consider adding %SUMO_HOME%\bin to your PATH.
        ) else (
            echo Error: sumo-gui.exe not found.
        )
    )
    
    echo.
    echo Testing Python traci module...
    python -c "import traci; print('TraCI module imported successfully!')" 2>nul
    if %errorlevel% neq 0 (
        echo Failed to import traci module. Make sure Python can find the SUMO tools.
        echo You may need to run:
        echo python -c "import os, sys; sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))"
    )
    
    echo.
    pause
)

if "%choice%"=="4" (
    echo Exiting...
    goto :eof
)

echo Invalid choice. Please run the script again and select a valid option.
