"""
Windows OS & Browser Action Executor for the Autonomous AI Desktop Agent.

Executes validated Plan steps produced by Module 2 (Brain) on Windows:
- Application launching & closing (desktop apps and web apps)
- Full browser automation (URLs, multi-engine search, YouTube auto-play, tabs, scrolling, zoom, history)
- Media & volume controls via native Windows virtual key events
- Unicode text typing via Windows SendInput
- Safe system power operations (lock, sleep, restart, shutdown)
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Dict, List, Optional, Tuple

import psutil

from src.tools import calendar_op, file_op, git_op, web_fetch


# ---------------------------------------------------------------------------
# Windows Virtual Key Codes & Input Helpers (Zero external dependencies)
# ---------------------------------------------------------------------------

VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12  # Alt
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22   # Page Down
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_F5 = 0x74
VK_F11 = 0x7A
VK_OEM_PLUS = 0xBB
VK_OEM_MINUS = 0xBD

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1
MOUSEEVENTF_WHEEL = 0x0800


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT_UNION)]


def _press_keys(*vkeys: int) -> None:
    """Presses and releases a combination of virtual keys (e.g., Ctrl+T)."""
    user32 = ctypes.windll.user32
    for vk in vkeys:
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.02)
    for vk in reversed(vkeys):
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.02)


def _scroll_wheel(delta: int) -> None:
    """Scrolls the mouse wheel vertically (positive = up, negative = down)."""
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, delta, 0)


def _focus_browser_window() -> bool:
    """Finds an open browser window (Chrome/Edge/Brave/Firefox) and brings it to focus."""
    user32 = ctypes.windll.user32
    found_hwnd = None
    browser_keywords = ("google chrome", "microsoft edge", "mozilla firefox", "brave", "opera")

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _lparam):
        nonlocal found_hwnd
        if found_hwnd is not None:
            return False
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.lower()
                if any(k in title for k in browser_keywords):
                    found_hwnd = hwnd
                    return False
        return True

    user32.EnumWindows(enum_cb, 0)
    if found_hwnd:
        SW_RESTORE = 9
        if user32.IsIconic(found_hwnd):
            user32.ShowWindow(found_hwnd, SW_RESTORE)
        user32.SetForegroundWindow(found_hwnd)
        time.sleep(0.12)
        return True
    return False


def _type_unicode(text: str) -> None:
    """Types arbitrary Unicode text into the active window using SendInput."""
    user32 = ctypes.windll.user32
    extra = ctypes.c_ulong(0)
    for ch in text:
        code = ord(ch)
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = _INPUT(
                type=INPUT_KEYBOARD,
                union=_INPUT_UNION(
                    ki=_KEYBDINPUT(
                        wVk=0,
                        wScan=code,
                        dwFlags=flags,
                        time=0,
                        dwExtraInfo=ctypes.pointer(extra),
                    )
                ),
            )
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# App, Website & Browser Mappings
# ---------------------------------------------------------------------------

WEB_APPS: Dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "instagram": "https://www.instagram.com",
    "whatsapp": "https://web.whatsapp.com",
    "netflix": "https://www.netflix.com",
    "prime video": "https://www.primevideo.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
    "wynk music": "https://wynk.in/music",
    "gaana": "https://gaana.com",
    "jiosaavn": "https://www.jiosaavn.com",
}

DESKTOP_LAUNCH: Dict[str, str] = {
    "chrome": "chrome",
    "edge": "msedge",
    "notepad": "notepad",
    "calculator": "calc",
    "vscode": "code",
    "file explorer": "explorer",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "paint": "mspaint",
    "cmd": "cmd",
    "terminal": "wt",
    "powershell": "powershell",
    "task manager": "taskmgr",
    "settings": "ms-settings:",
    "spotify": "spotify:",
    "telegram": "tg:",
}

CLOSE_PROCESS_MAP: Dict[str, str] = {
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",
    "vscode": "Code.exe",
    "word": "WINWORD.EXE",
    "excel": "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "paint": "mspaint.exe",
    "spotify": "Spotify.exe",
    "whatsapp": "WhatsApp.exe",
    "telegram": "Telegram.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "Taskmgr.exe",
}

SEARCH_URL_TEMPLATES: Dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "chrome": "https://www.google.com/search?q={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={q}",
    "github": "https://github.com/search?q={q}",
    "spotify": "https://open.spotify.com/search/{q}",
    "bing": "https://www.bing.com/search?q={q}",
}

BROWSER_HOTKEYS: Dict[str, Tuple[int, ...]] = {
    "new_tab": (VK_CONTROL, ord("T")),
    "close_tab": (VK_CONTROL, ord("W")),
    "reopen_tab": (VK_CONTROL, VK_SHIFT, ord("T")),
    "next_tab": (VK_CONTROL, VK_TAB),
    "prev_tab": (VK_CONTROL, VK_SHIFT, VK_TAB),
    "refresh": (VK_CONTROL, ord("R")),
    "back": (VK_MENU, VK_LEFT),
    "forward": (VK_MENU, VK_RIGHT),
    "scroll_top": (VK_HOME,),
    "scroll_bottom": (VK_END,),
    "zoom_in": (VK_CONTROL, VK_OEM_PLUS),
    "zoom_out": (VK_CONTROL, VK_OEM_MINUS),
    "zoom_reset": (VK_CONTROL, ord("0")),
    "incognito": (VK_CONTROL, VK_SHIFT, ord("N")),
    "history": (VK_CONTROL, ord("H")),
    "downloads": (VK_CONTROL, ord("J")),
    "bookmark": (VK_CONTROL, ord("D")),
    "fullscreen": (VK_F11,),
}


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Executes Brain Plan steps on Windows desktop and web browsers."""

    def _find_first_youtube_video(self, query: str) -> Optional[str]:
        """Fetches the first YouTube video URL for a search query."""
        encoded = urllib.parse.quote_plus(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded}"
        try:
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            if match:
                return f"https://www.youtube.com/watch?v={match.group(1)}&autoplay=1"
        except Exception:
            pass
        return None

    def _is_process_running(self, target: str) -> bool:
        t = (target or "").strip().lower()
        exe = CLOSE_PROCESS_MAP.get(t, f"{t}.exe" if not t.endswith(".exe") else t).lower()
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name == exe or t in name:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def open_app(self, target: str) -> str:
        t = (target or "").strip().lower()
        if t in WEB_APPS:
            webbrowser.open(WEB_APPS[t])
            return f"Opened {t} in browser."

        cmd = DESKTOP_LAUNCH.get(t, t)
        try:
            if cmd.endswith(":"):
                os.startfile(cmd)
            else:
                subprocess.Popen(f'start "" "{cmd}"', shell=True)
            time.sleep(0.35)
            verified = self._is_process_running(t)
            return f"Launched {t} (verified active)." if verified else f"Launched {t}."
        except Exception as e:
            return f"Failed to open {t}: {e}"

    def close_app(self, target: str) -> str:
        t = (target or "").strip().lower()
        if t in WEB_APPS:
            _press_keys(*BROWSER_HOTKEYS["close_tab"])
            return f"Closed current browser tab for {t}."

        exe = CLOSE_PROCESS_MAP.get(t, f"{t}.exe" if not t.endswith(".exe") else t)
        res = subprocess.run(
            ["taskkill", "/f", "/im", exe],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            verified = not self._is_process_running(t)
            return f"Closed {t} (verified stopped)." if verified else f"Closed {t}."
        return f"{t} is not currently running."

    def open_url(self, url: str) -> str:
        if not url:
            return "No URL provided."
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        webbrowser.open(url)
        return f"Opened {url}."

    def search_web(self, target: str, query: str) -> str:
        t = (target or "google").strip().lower()
        template = SEARCH_URL_TEMPLATES.get(t, SEARCH_URL_TEMPLATES["google"])
        url = template.format(q=urllib.parse.quote_plus(query or ""))
        webbrowser.open(url)
        return f"Searched {t} for '{query}'."

    def play_media(self, target: str, query: Optional[str]) -> str:
        t = (target or "youtube").strip().lower()
        if not query:
            _press_keys(VK_MEDIA_PLAY_PAUSE)
            return f"Toggled playback on {t}."

        if t == "youtube":
            video_url = self._find_first_youtube_video(query)
            if video_url:
                webbrowser.open(video_url)
                return f"Playing '{query}' on YouTube."
        return self.search_web(t, query)

    def control_browser(self, command: str) -> str:
        cmd = (command or "").strip().lower()
        _focus_browser_window()
        if cmd == "scroll_down":
            _scroll_wheel(-480)
            return "Scrolled down."
        if cmd == "scroll_up":
            _scroll_wheel(480)
            return "Scrolled up."

        keys = BROWSER_HOTKEYS.get(cmd)
        if keys:
            _press_keys(*keys)
            return f"Executed browser command: {cmd}."
        return f"Unknown browser command: {cmd}."

    def control_volume(self, value: str) -> str:
        v = (value or "").strip().lower()
        if v == "up":
            for _ in range(5):
                _press_keys(VK_VOLUME_UP)
            return "Increased volume."
        if v == "down":
            for _ in range(5):
                _press_keys(VK_VOLUME_DOWN)
            return "Decreased volume."
        if v in ("mute", "unmute"):
            _press_keys(VK_VOLUME_MUTE)
            return f"Volume {v}d."
        return f"Unknown volume action: {v}."

    def control_system(self, value: str) -> str:
        v = (value or "").strip().lower()
        if v == "lock":
            ctypes.windll.user32.LockWorkStation()
            return "Locked workstation."
        if v == "sleep":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            return "Putting system to sleep."
        if v == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "5"])
            return "Shutting down in 5 seconds."
        if v == "restart":
            subprocess.Popen(["shutdown", "/r", "/t", "5"])
            return "Restarting in 5 seconds."
        return f"Unknown system command: {v}."

    def control_mouse(self, value: str, target: str = "") -> str:
        user32 = ctypes.windll.user32
        v = (value or "click").strip().lower()
        if v == "move" and "," in (target or ""):
            try:
                x, y = [int(p.strip()) for p in target.split(",", 1)]
                user32.SetCursorPos(x, y)
                return f"Moved mouse to ({x}, {y})."
            except ValueError:
                return f"Invalid coordinates: {target}"
        if v in ("click", "left_click"):
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            return "Clicked mouse."
        if v == "double_click":
            for _ in range(2):
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.05)
            return "Double-clicked mouse."
        if v == "right_click":
            user32.mouse_event(0x0008, 0, 0, 0, 0)
            user32.mouse_event(0x0010, 0, 0, 0, 0)
            return "Right-clicked mouse."
        return f"Unknown mouse action: {v}"

    def run_terminal(self, command: str) -> str:
        cmd = (command or "").strip()
        if not cmd:
            return "No terminal command provided."
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            out = (res.stdout or res.stderr or "").strip()
            status = "OK" if res.returncode == 0 else f"exit {res.returncode}"
            return f"[{status}] {out[:300]}" if out else f"[{status}] Command completed."
        except Exception as e:
            return f"Terminal command failed: {e}"

    def execute_plan(self, plan: Dict, memory=None) -> List[str]:
        """Executes all steps in a Brain Plan dictionary and returns status logs."""
        steps = plan.get("steps") or []
        if not steps:
            return []

        logs: List[str] = []
        actions = [s.get("action") for s in steps]
        has_play = "play" in actions
        has_search = "search" in actions

        pending_query: Optional[str] = None
        for step in steps:
            action = step.get("action")
            target = step.get("target") or ""
            query = step.get("query") or ""
            value = step.get("value") or ""

            if action == "open_app":
                # Skip opening bare web-app homepage if we are about to search/play on it
                if (has_search or has_play) and target.lower() in WEB_APPS:
                    continue
                logs.append(self.open_app(target))
            elif action == "close_app":
                logs.append(self.close_app(target))
            elif action == "open_url":
                logs.append(self.open_url(value))
            elif action == "search":
                if has_play:
                    pending_query = query
                    continue
                logs.append(self.search_web(target, query))
            elif action == "play":
                logs.append(self.play_media(target, pending_query or query))
                pending_query = None
            elif action == "browser":
                logs.append(self.control_browser(value))
            elif action in ("pause", "resume"):
                _press_keys(VK_MEDIA_PLAY_PAUSE)
                logs.append(f"Media {action}d.")
            elif action == "next":
                _press_keys(VK_MEDIA_NEXT_TRACK)
                logs.append("Skipped to next track.")
            elif action == "previous":
                _press_keys(VK_MEDIA_PREV_TRACK)
                logs.append("Went to previous track.")
            elif action == "volume":
                logs.append(self.control_volume(value))
            elif action == "type_text":
                _type_unicode(query)
                logs.append(f"Typed text: '{query}'.")
            elif action == "system":
                logs.append(self.control_system(value))
            elif action == "mouse":
                logs.append(self.control_mouse(value, target))
            elif action == "run_terminal":
                logs.append(self.run_terminal(query or value))
            elif action == "file_op":
                logs.append(file_op(value or "list", target or ".", query))
            elif action == "git_op":
                logs.append(git_op(value or "status", target))
            elif action == "calendar_op":
                logs.append(calendar_op(value or "list", query or target, target))
            elif action == "web_fetch":
                logs.append(web_fetch(query or target, target or "weather"))
            elif action == "memory_op" and memory is not None:
                op = (value or "list").lower()
                if op == "remember":
                    logs.append(memory.remember(query or target))
                elif op == "forget":
                    logs.append(memory.forget(query or target))
                else:
                    items = memory.list_memories()
                    logs.append(
                        "Saved memories: " + "; ".join(i["fact"] for i in items[-5:])
                        if items else "No saved personal memories yet."
                    )

        return logs
