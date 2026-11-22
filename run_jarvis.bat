@echo off
title Autonomous AI Desktop Agent - Live Jarvis Assistant
echo ========================================================
echo   Autonomous AI Desktop Agent: Full End-to-End Jarvis
echo   STT (74M) + Brain (LLM) + TTS (82M) Live Assistant
echo ========================================================
echo.

:: Ensure local Ollama daemon is active in background
curl.exe -s http://localhost:11434/ >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo [*] Launching local Ollama neural engine...
        start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
        timeout /t 2 /nobreak >nul
    )
)

.\.venv\Scripts\python.exe jarvis_live.py
pause
