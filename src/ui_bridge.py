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
import queue
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
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

VALID_STATES = ("IDLE", "LISTENING", "THINKING", "SPEAKING", "EXECUTING", "COMPLETED", "ERROR", "CLARIFICATION")


class _CommandPayload(BaseModel):
    text: str


class _ConfirmPayload(BaseModel):
    approved: bool


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
    if action == "volume":
        return f"Adjusting volume ({value or 'change'})"
    if action == "brightness":
        return f"Adjusting brightness ({value or 'change'})"
    if action == "wallpaper":
        return f"Changing wallpaper ({query or 'new'})"
    if action == "screenshot":
        return "Copying screenshot to clipboard"
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
            "error_message": "",
            "recognized_text": "",
            "reply_text": "",
            "tts_muted": False,
            "island_visible": False,
            "confirmation": {"active": False, "question": ""},
            "services": {"brain": "ready", "stt": "ready", "mic": "ready", "tts": "ready"},
            "history": [],
            "task": {
                "active": False,
                "completed": False,
                "title": "",
                "steps": [],
            },
            "ts": time.time(),
        }
        self.command_queue: "queue.Queue[str]" = queue.Queue()
        self.mic_cancel_event = threading.Event()
        self.mic_trigger_event = threading.Event()
        self._confirm_event = threading.Event()
        self._confirm_result: Optional[bool] = None
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._island = None

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
                    "COMPLETED": "Done",
                    "ERROR": "Something went wrong",
                    "CLARIFICATION": "Waiting for your answer...",
                }
                self._state["status_text"] = defaults.get(st, "Ready")
            if st != "ERROR":
                self._state["error_message"] = ""
            self._bump()

    def set_recognized_text(self, text: str) -> None:
        with self._lock:
            self._state["recognized_text"] = text
            self._bump()

    def set_reply_text(self, text: str) -> None:
        with self._lock:
            self._state["reply_text"] = text
            self._bump()

    def add_history(self, role: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            hist = self._state.setdefault("history", [])
            hist.append({"role": role, "text": text})
            if len(hist) > 30:
                del hist[:-30]
            self._bump()

    def clear_history(self) -> None:
        with self._lock:
            self._state["history"] = []
            self._bump()

    def set_services(self, **kwargs: str) -> None:
        with self._lock:
            self._state["services"].update(kwargs)
            self._bump()

    def is_tts_muted(self) -> bool:
        with self._lock:
            return bool(self._state.get("tts_muted", False))

    def toggle_tts_mute(self) -> bool:
        with self._lock:
            self._state["tts_muted"] = not bool(self._state.get("tts_muted", False))
            self._bump()
            return self._state["tts_muted"]

    def begin_confirmation(self, question: str) -> None:
        self._confirm_event.clear()
        self._confirm_result = None
        with self._lock:
            self._state["state"] = "CLARIFICATION"
            self._state["status_text"] = "Confirmation needed"
            self._state["confirmation"] = {"active": True, "question": question}
            self._bump()

    def resolve_confirmation(self, approved: bool) -> None:
        self._confirm_result = bool(approved)
        self._confirm_event.set()
        with self._lock:
            self._state["confirmation"] = {"active": False, "question": ""}
            self._bump()

    def poll_confirmation(self, timeout: float = 0.1) -> Optional[bool]:
        if self._confirm_event.wait(timeout=timeout):
            return self._confirm_result
        return None

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

    def set_island_visible(self, visible: bool) -> None:
        with self._lock:
            self._state["island_visible"] = bool(visible)
            self._bump()

    def start_task(self, title: str, plan_steps: List[Dict[str, Any]]) -> None:
        steps_ui = [
            {"label": _humanize_step(s), "status": "pending"}
            for s in (plan_steps or [])
        ]
        with self._lock:
            self._state["state"] = "EXECUTING"
            self._state["status_text"] = "Executing..."
            self._state["error_message"] = ""
            self._state["mic_level"] = 0.0
            self._state["tts_level"] = 0.0
            self._state["island_visible"] = True
            self._state["task"] = {
                "active": True,
                "completed": False,
                "title": title,
                "steps": steps_ui,
            }
            self._bump()
        # Automatically hide/minimize the main JARVIS window so user's app stays visible
        self.minimize_main_window()

    def update_step(self, index: int, status: str, label: Optional[str] = None) -> None:
        with self._lock:
            steps = self._state["task"].get("steps") or []
            if 0 <= index < len(steps):
                steps[index]["status"] = status
                if label:
                    steps[index]["label"] = label
                self._bump()

    def finish_task(self, had_error: bool = False, error_msg: str = "") -> None:
        with self._lock:
            self._state["task"]["completed"] = True
            self._state["task"]["active"] = True
            if had_error:
                self._state["state"] = "ERROR"
                self._state["status_text"] = "Error"
                self._state["error_message"] = (error_msg or "Task failed")[:80]
            else:
                self._state["state"] = "COMPLETED"
                self._state["status_text"] = "Done"
            self._bump()

        def _hide_later():
            time.sleep(2.0)
            with self._lock:
                if self._state["task"].get("completed"):
                    self._state["task"]["active"] = False
                    if self._state["state"] in ("COMPLETED", "ERROR"):
                        self._state["state"] = "IDLE"
                        self._state["status_text"] = "Ready"
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

        @app.post("/command")
        async def post_command(payload: _CommandPayload):
            txt = (payload.text or "").strip()
            if not txt:
                return {"ok": False, "error": "Empty command"}
            self.command_queue.put(txt)
            self.mic_cancel_event.set()
            return {"ok": True}

        @app.post("/confirm")
        async def post_confirm(payload: _ConfirmPayload):
            self.resolve_confirmation(payload.approved)
            self.mic_cancel_event.set()
            return {"ok": True, "approved": payload.approved}

        @app.post("/mic/toggle")
        async def post_mic_toggle():
            if self._state.get("state") == "LISTENING":
                self.mic_cancel_event.set()
            else:
                self.mic_trigger_event.set()
            return {"ok": True}

        @app.post("/tts/mute")
        async def post_tts_mute():
            muted = self.toggle_tts_mute()
            return {"ok": True, "muted": muted}

        @app.post("/history/clear")
        async def post_history_clear():
            self.clear_history()
            return {"ok": True}

        @app.post("/window/minimize")
        async def post_window_minimize():
            self.minimize_main_window()
            return {"ok": True}

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

    def start(self, open_window: bool = True, start_island: bool = True) -> None:
        """Starts the FastAPI server, native Dynamic Island overlay, and full UI window."""
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

        time.sleep(0.3)
        if start_island:
            try:
                from src.dynamic_island import DynamicIslandOverlay
                self._island = DynamicIslandOverlay(self)
                self._island.start()
            except Exception:
                self._island = None
        if open_window:
            self.launch_frontend_window()

    def minimize_main_window(self) -> None:
        """Minimizes the full JARVIS Chrome --app window if open so desktop apps stay visible."""
        if os.name != "nt":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            SW_MINIMIZE = 6

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _lp):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        # Exact title of frontend/index.html is "JARVIS"
                        if buf.value.strip() == "JARVIS":
                            user32.ShowWindow(hwnd, SW_MINIMIZE)
                return True

            user32.EnumWindows(_cb, 0)
            time.sleep(0.12)
        except Exception:
            pass

    def restore_main_window(self) -> None:
        """Restores the full JARVIS Chrome --app window for conversational Q&A."""
        if os.name != "nt":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            SW_RESTORE = 9

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _lp):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if buf.value.strip() == "JARVIS" and user32.IsIconic(hwnd):
                            user32.ShowWindow(hwnd, SW_RESTORE)
                return True

            user32.EnumWindows(_cb, 0)
            time.sleep(0.12)
        except Exception:
            pass

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
                ui_profile = PROJECT_ROOT / "memory" / "ui_chrome_profile"
                ui_profile.mkdir(parents=True, exist_ok=True)
                subprocess.Popen(
                    [
                        chrome_exe,
                        f"--app={url}",
                        f"--user-data-dir={ui_profile}",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--window-size=960,720",
                    ],
                    creationflags=subprocess.DETACHED_PROCESS,
                )
                return
            except Exception:
                pass
        webbrowser.open(url)

    def stop(self) -> None:
        if self._island is not None:
            try:
                self._island.stop()
            except Exception:
                pass
        if self._server is not None:
            self._server.should_exit = True
