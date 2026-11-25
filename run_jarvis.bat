@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe jarvis_hybrid.py %*
pause
