@echo off
setlocal
cd /d "%~dp0"
where csv-screen >nul 2>&1
if %ERRORLEVEL% equ 0 goto :use_csv_screen
goto :py_module

:use_csv_screen
if "%~1"=="" csv-screen -h & exit /b %ERRORLEVEL%
csv-screen %*
exit /b %ERRORLEVEL%

:py_module
if "%~1"=="" python -m two_stage_screen screen -h & exit /b %ERRORLEVEL%
python -m two_stage_screen screen %*
exit /b %ERRORLEVEL%
