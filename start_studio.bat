@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "zed_studio.py"
