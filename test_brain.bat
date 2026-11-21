@echo off
echo ========================================================
echo   Autonomous AI Desktop Agent - Module 2 (Brain / LLM)
echo ========================================================
echo.

:: Auto-start local Ollama engine if not already active
curl.exe -s http://localhost:11434/ >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo [*] Starting local Ollama engine in background...
        start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
        timeout /t 2 /nobreak >nul
    )
)

.\.venv\Scripts\python.exe brain_module\chat.py
pause
