@echo off
title AI Desktop Agent - Student Web UI
echo ========================================================
echo   AI-Powered Autonomous Desktop Agent - Web Frontend
echo   URL: http://127.0.0.1:8000
echo ========================================================
echo.

start "" http://127.0.0.1:8000
.\.venv\Scripts\python.exe web_app.py
pause
