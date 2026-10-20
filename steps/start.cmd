@echo off
cd /d "%~dp0.."
python steps\serve.py --open
if errorlevel 1 pause
