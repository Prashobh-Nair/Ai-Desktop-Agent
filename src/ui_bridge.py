"""
Real-time UI Event Bridge for JARVIS Voice-First Frontend.

Runs a lightweight FastAPI + SSE server in a background daemon thread and
broadcasts real backend events (orb state, mic RMS, TTS amplitude, task steps)
to the browser/app frontend with zero impact on 8 GB RAM laptops.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

VALID_STATES = ("IDLE", "LISTENING", "THINKING", "SPEAKING", "EXECUTING")


def _humanize_step(step: Dict[str, Any]) -> str:
    """Converts a Brain Plan step dict into a clean, user-friendly progress label."""
    action = (step.get("action") or "").strip().lower()
    target = (step.get("target") or "").strip()
    query = (step.get("query") or "").strip()
    value = (step.get("value") or "").strip()

    if action == "open_app":
        return f"Opening {target.title() or 'application'}"
    if action == "close_app":
        return f"Closing {target.title() or 'application'}"
    if action == "open_url":
        return f"Navigating to {value or target or 'website'}"
    if action == "search":
        site = target.title() if target else "Web"
        return f"Searching {site} for \"{query}\"" if query else f"Searching {site}"
    if action == "play":
        return f"Playing \"{query}\" on {target.title() or 'media'}" if query else f"Controlling playback on {target.title() or 'media'}"
    if action == "type_text":
        preview = (query[:32] + "...") if len(query) > 32 else query
        return f"Typing \"{preview}\""
    if action == "hotkey":
        return f"Pressing shortcut {value or query or target}"
    if action == "click_element":
        return f"Clicking {query or target or 'element'}"
    if action == "mouse":
        return f"Mouse action: {value or 'click'}"
    if action == "browser":
        return f"Browser: {(value or 'action').replace('_', ' ')}"
    if action == "see_screen":
        return "Analyzing current screen"
    if action == "verify_screen":
        return "Verifying result on screen"
    if action == "focus_window":
        return f"Focusing {target or 'window'}"
    if action == "wait":
        return f"Waiting for {target or 'page'}"
    if action == "plugin_op":
        return f"Running {target or 'plugin'}: {value or 'task'}"
    if action == "run_terminal":
        return "Running terminal command"
    if action == "file_op":
        return f"File operation: {value or 'list'}"
    if action == "git_op":
        return f"Git: {value or 'status'}"
    if action == "calendar_op":
        return "Checking calendar"
    if action == "web_fetch":
        return f"Fetching {target or 'web'} info"
    return f"Executing {action.replace('_', ' ')}"


class JarvisUIBridge:
    """Thread-safe state + SSE broadcaster for the JARVIS frontend."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._seq = 0
        self._state: Dict[str, Any] = {
            "state": "IDLE",
            "mic_level": 0.0,
            "tts_level": 0.0,
            "status_text": "Ready",
            "task": {
                "active": False,
                "completed": False,
                "title": "",
                "steps": [],
            },
            "ts": time.time(),
        }
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _bump(self) -> None:
        self._seq += 1
        self._state["ts"] = time.time()

    def set_state(self, state: str, status_text: Optional[str] = None) -> None:
        st = state.upper()
        if st not in VALID_STATES:
            return
        with self._lock:
            self._state["state"] = st
            if st != "LISTENING":
                self._state["mic_level"] = 0.0
            if st != "SPEAKING":
                self._state["tts_level"] = 0.0
            if status_text is not None:
                self._state["status_text"] = status_text
            else:
                defaults = {
                    "IDLE": "Ready",
                    "LISTENING": "Listening...",
                    "THINKING": "Thinking...",
                    "SPEAKING": "Speaking...",
                    "EXECUTING": "Executing...",
                }
                self._state["status_text"] = defaults.get(st, "Ready")
            self._bump()

    def set_mic_level(self, level: float) -> None:
        clamped = max(0.0, min(1.0, float(level)))
        with self._lock:
            self._state["mic_level"] = round(clamped, 3)
            self._bump()

    def set_tts_level(self, level: float) -> None:
        clamped = max(0.0, min(1.0, float(level)))
        with self._lock:
            self._state["tts_level"] = round(clamped, 3)
            self._bump()

    def start_task(self, title: str, plan_steps: List[Dict[str, Any]]) -> None:
        steps_ui = [
            {"label": _humanize_step(s), "status": "pending"}
            for s in (plan_steps or [])
        ]
        with self._lock:
            self._state["state"] = "EXECUTING"
            self._state["status_text"] = "Executing..."
            self._state["mic_level"] = 0.0
            self._state["tts_level"] = 0.0
            self._state["task"] = {
                "active": True,
                "completed": False,
                "title": title,
                "steps": steps_ui,
            }
            self._bump()

    def update_step(self, index: int, status: str, label: Optional[str] = None) -> None:
        with self._lock:
            steps = self._state["task"].get("steps") or []
            if 0 <= index < len(steps):
                steps[index]["status"] = status
                if label:
                    steps[index]["label"] = label
                self._bump()

    def finish_task(self, had_error: bool = False) -> None:
        with self._lock:
            self._state["task"]["completed"] = True
            self._state["task"]["active"] = True
            self._bump()

        def _hide_later():
            time.sleep(2.2)
            with self._lock:
                if self._state["task"].get("completed"):
                    self._state["task"]["active"] = False
                    self._bump()

        threading.Thread(target=_hide_later, daemon=True).start()

    def build_app(self) -> FastAPI:
        app = FastAPI(title="JARVIS UI Bridge")

        @app.get("/health")
        async def health():
            return {"ok": True, "state": self._state["state"]}

        @app.get("/state")
        async def get_state():
            return self.snapshot()

        @app.get("/events")
        async def sse_events():
            async def event_stream():
                last_seq = -1
                while True:
                    with self._lock:
                        cur_seq = self._seq
                        payload = json.dumps(self._state) if cur_seq != last_seq else None
                    if payload is not None:
                        last_seq = cur_seq
                        yield f"data: {payload}\n\n"
                    await asyncio.sleep(0.033)  # ~30 Hz max update rate

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if FRONTEND_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

            @app.get("/")
            async def index():
                return FileResponse(str(FRONTEND_DIR / "index.html"))

        return app

    def start(self, open_window: bool = True) -> None:
        """Starts the FastAPI server in a daemon thread and optionally opens Chrome app window."""
        if self._thread and self._thread.is_alive():
            return
        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait briefly for server to bind
        time.sleep(0.35)
        if open_window:
            self.launch_frontend_window()

    def launch_frontend_window(self) -> None:
        """Opens the JARVIS frontend in a clean frameless Chrome --app window (or browser fallback)."""
        url = f"http://{self.host}:{self.port}/"
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        chrome_exe = next((p for p in chrome_paths if os.path.isfile(p)), None)
        if not chrome_exe:
            chrome_exe = shutil.which("chrome") or shutil.which("google-chrome")

        if chrome_exe:
            try:
                subprocess.Popen(
                    [
                        chrome_exe,
                        f"--app={url}",
                        "--window-size=960,720",
                    ],
                    creationflags=subprocess.DETACHED_PROCESS,
                )
                return
            except Exception:
                pass
        webbrowser.open(url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
