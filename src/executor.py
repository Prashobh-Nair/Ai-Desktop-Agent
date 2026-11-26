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
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Dict, List, Optional, Tuple

import psutil

from src.tools import calendar_op, file_op, git_op, web_fetch


# ---------------------------------------------------------------------------
# Chrome Tab-Reuse Engine
# Always open URLs in a new tab in the existing Chrome window.
# Never uses Edge or the system default browser.
# ---------------------------------------------------------------------------

_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _find_chrome_exe() -> str:
    """Return the path to chrome.exe, or raise RuntimeError if not found."""
    for path in _CHROME_PATHS:
        if os.path.isfile(path):
            return path
    # Fallback: search PATH
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    raise RuntimeError("Google Chrome not found on this machine.")


def _is_chrome_running() -> bool:
    """Return True if any chrome.exe process is currently running."""
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "chrome.exe":
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _focus_chrome_window() -> bool:
    """
    Bring the Chrome window to the foreground.
    Returns True if a Chrome window was found and focused.
    """
    user32 = ctypes.windll.user32
    found_hwnd = None

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum_cb(hwnd, _lparam):
        nonlocal found_hwnd
        if found_hwnd is not None:
            return False
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.lower()
                if "google chrome" in title or "chrome" in title:
                    found_hwnd = hwnd
                    return False
        return True

    user32.EnumWindows(_enum_cb, 0)
    if found_hwnd:
        SW_RESTORE = 9
        if user32.IsIconic(found_hwnd):
            user32.ShowWindow(found_hwnd, SW_RESTORE)
        user32.SetForegroundWindow(found_hwnd)
        time.sleep(0.15)
        return True
    return False


def _chrome_open_url(url: str) -> str:
    """
    Open a URL in Chrome.
    - If Chrome is already running: focus it, open a new tab, navigate there.
    - If Chrome is not running: launch it with the URL directly.
    Never opens a new Chrome window for every request.
    Returns a human-readable result message.
    """
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        chrome_exe = _find_chrome_exe()
    except RuntimeError:
        # Last-resort fallback to system default browser
        webbrowser.open(url)
        return f"Opened {url} (Chrome not found — used default browser)."

    if _is_chrome_running():
        # Chrome is running — focus the window, open a new tab, type the URL
        focused = _focus_chrome_window()
        if focused:
            time.sleep(0.1)
            # Open new tab with Ctrl+T
            ctypes.windll.user32.keybd_event(0x11, 0, 0, 0)        # Ctrl down
            ctypes.windll.user32.keybd_event(ord("T"), 0, 0, 0)    # T down
            ctypes.windll.user32.keybd_event(ord("T"), 0, 2, 0)    # T up
            ctypes.windll.user32.keybd_event(0x11, 0, 2, 0)        # Ctrl up
            time.sleep(0.3)
            # Navigate using the address bar (Ctrl+L then type URL and Enter)
            ctypes.windll.user32.keybd_event(0x11, 0, 0, 0)        # Ctrl down
            ctypes.windll.user32.keybd_event(ord("L"), 0, 0, 0)    # L down
            ctypes.windll.user32.keybd_event(ord("L"), 0, 2, 0)    # L up
            ctypes.windll.user32.keybd_event(0x11, 0, 2, 0)        # Ctrl up
            time.sleep(0.2)
            # Type the URL character by character via SendInput
            _chrome_type_address(url)
            time.sleep(0.1)
            # Press Enter
            ctypes.windll.user32.keybd_event(0x0D, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0x0D, 0, 2, 0)
            return f"Opened {url} in a new Chrome tab."
        # Window focus failed — fall through to subprocess launch with new-tab flag
    # Chrome not running (or focus failed) → launch it
    subprocess.Popen([chrome_exe, url], creationflags=subprocess.DETACHED_PROCESS)
    time.sleep(0.4)
    return f"Launched Chrome and opened {url}."


def _chrome_type_address(text: str) -> None:
    """Type a URL into the Chrome address bar using SendInput (Unicode)."""
    user32 = ctypes.windll.user32
    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002

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
        time.sleep(0.003)


# ---------------------------------------------------------------------------
# Windows Virtual Key Codes & Input Helpers (Zero external dependencies)
# ---------------------------------------------------------------------------

VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12  # Alt
VK_LWIN = 0x5B
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22   # Page Down
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E
VK_F1 = 0x70
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
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


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
    """
    Finds an open Chrome window and brings it to focus.
    Falls back to any browser window if Chrome is not open.
    """
    # Prefer Chrome window first
    if _focus_chrome_window():
        return True
    # Fallback: any browser
    user32 = ctypes.windll.user32
    found_hwnd = None
    browser_keywords = ("mozilla firefox", "brave", "opera")

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


def _type_unicode(text: str, replace: bool = False, press_enter: bool = False) -> None:
    """Types arbitrary Unicode text (with newline/tab support) into the active window."""
    if replace:
        _press_keys(VK_CONTROL, ord("A"))
        time.sleep(0.04)
        _press_keys(VK_BACK)
        time.sleep(0.04)

    # For longer text, clipboard paste is much faster and avoids dropped characters
    if len(text) > 120 and _set_clipboard(text):
        _press_keys(VK_CONTROL, ord("V"))
        time.sleep(0.08)
        if press_enter:
            _press_keys(VK_RETURN)
        return

    user32 = ctypes.windll.user32
    extra = ctypes.c_ulong(0)
    for ch in text:
        if ch == "\n":
            _press_keys(VK_RETURN)
            continue
        if ch == "\t":
            _press_keys(VK_TAB)
            continue
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
        time.sleep(0.004)
    if press_enter:
        _press_keys(VK_RETURN)


def _set_clipboard(text: str) -> bool:
    """Copies Unicode text to the Windows clipboard via ctypes (64-bit safe, no pyperclip needed)."""
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = ctypes.c_bool
        kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
        kernel32.GlobalFree.restype = ctypes.c_void_p
        user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        user32.OpenClipboard.restype = ctypes.c_bool
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p

        data = (text + "\0").encode("utf-16-le")
        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not h_mem:
                return False
            ptr = kernel32.GlobalLock(h_mem)
            if not ptr:
                kernel32.GlobalFree(h_mem)
                return False
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(h_mem)
            if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
                kernel32.GlobalFree(h_mem)
                return False
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


_KEY_NAME_MAP: Dict[str, int] = {
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "shift": VK_SHIFT,
    "alt": VK_MENU,
    "win": VK_LWIN,
    "windows": VK_LWIN,
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "tab": VK_TAB,
    "backspace": VK_BACK,
    "back": VK_BACK,
    "del": VK_DELETE,
    "delete": VK_DELETE,
    "esc": VK_ESCAPE,
    "escape": VK_ESCAPE,
    "space": VK_SPACE,
    "up": VK_UP,
    "down": VK_DOWN,
    "left": VK_LEFT,
    "right": VK_RIGHT,
    "home": VK_HOME,
    "end": VK_END,
    "pgup": VK_PRIOR,
    "pageup": VK_PRIOR,
    "pgdn": VK_NEXT,
    "pagedown": VK_NEXT,
    "plus": VK_OEM_PLUS,
    "minus": VK_OEM_MINUS,
}
for _i in range(1, 13):
    _KEY_NAME_MAP[f"f{_i}"] = VK_F1 + (_i - 1)


def _press_hotkey(combo: str) -> bool:
    """Parses and presses a shortcut like 'ctrl+a', 'ctrl+shift+s', 'enter', 'alt+tab'."""
    parts = [p.strip().lower() for p in (combo or "").split("+") if p.strip()]
    if not parts:
        return False
    vkeys: List[int] = []
    for part in parts:
        if part in _KEY_NAME_MAP:
            vkeys.append(_KEY_NAME_MAP[part])
        elif len(part) == 1 and part.isalnum():
            vkeys.append(ord(part.upper()))
        else:
            return False
    _press_keys(*vkeys)
    return True


WINDOW_TITLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "notepad": ("notepad", "untitled"),
    "word": ("word", "document", "winword"),
    "excel": ("excel",),
    "powerpoint": ("powerpoint",),
    "vscode": ("visual studio code", "code"),
    "chrome": ("google chrome", "chrome"),
    "edge": ("microsoft edge", "edge"),
    "file explorer": ("file explorer", "explorer"),
    "terminal": ("windows powershell", "powershell", "command prompt", "terminal"),
    "paint": ("paint",),
    "calculator": ("calculator",),
}


def _find_window(target: str) -> Optional[Tuple[int, str]]:
    """Finds the first visible top-level window matching target or its aliases."""
    user32 = ctypes.windll.user32
    t = (target or "").strip().lower()
    if not t:
        return None
    keywords = WINDOW_TITLE_ALIASES.get(t, (t,))
    result: Optional[Tuple[int, str]] = None

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _lparam):
        nonlocal result
        if result is not None:
            return False
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title_lower = buf.value.lower()
                if any(k in title_lower for k in keywords):
                    result = (int(hwnd), buf.value)
                    return False
        return True

    user32.EnumWindows(enum_cb, 0)
    return result


def _bring_hwnd_to_front(hwnd: int) -> bool:
    """Restores and brings a window handle to the foreground reliably."""
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        fg_hwnd = user32.GetForegroundWindow()
        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
        cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        attached = False
        if fg_tid and fg_tid != cur_tid:
            attached = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(cur_tid, fg_tid, False)
        time.sleep(0.15)
        return True
    except Exception:
        return False


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
    "word": "winword /w",
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
            return _chrome_open_url(WEB_APPS[t])

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
        return _chrome_open_url(url)

    def search_web(self, target: str, query: str) -> str:
        t = (target or "google").strip().lower()
        template = SEARCH_URL_TEMPLATES.get(t, SEARCH_URL_TEMPLATES["google"])
        url = template.format(q=urllib.parse.quote_plus(query or ""))
        return _chrome_open_url(url)

    def play_media(self, target: str, query: Optional[str]) -> str:
        t = (target or "youtube").strip().lower()
        if not query:
            _press_keys(VK_MEDIA_PLAY_PAUSE)
            return f"Toggled playback on {t}."

        if t == "youtube":
            video_url = self._find_first_youtube_video(query)
            if video_url:
                return _chrome_open_url(video_url)
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

    def focus_window(self, target: str) -> str:
        """Finds a running window matching target and brings it to foreground."""
        found = _find_window(target)
        if not found:
            return f"Window '{target}' not found."
        hwnd, title = found
        if _bring_hwnd_to_front(hwnd):
            return f"Focused window: {title}."
        return f"Found '{title}' but could not focus it."

    def wait_for_window(self, target: str, timeout: float = 6.0) -> bool:
        """Waits until a window matching target appears, then focuses it."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = _find_window(target)
            if found:
                _bring_hwnd_to_front(found[0])
                time.sleep(0.25)
                return True
            time.sleep(0.2)
        return False

    def click_element(self, description: str, click_type: str = "click", vision=None) -> str:
        """Uses Vision to locate a UI element on screen and clicks it (with 1 retry)."""
        if vision is None:
            return "Vision engine is not attached; cannot locate UI element."
        if not description:
            return "No UI element description provided."

        loc = vision.locate_element(description)
        if not loc.get("found"):
            time.sleep(0.5)
            loc = vision.locate_element(description)
        if not loc.get("found"):
            return f"Could not find '{description}' on screen ({loc.get('reason', 'not visible')})."

        x, y = int(loc["x"]), int(loc["y"])
        act = (click_type or "click").strip().lower()
        if act not in ("click", "left_click", "double_click", "right_click"):
            act = "click"
        self.control_mouse(act, f"{x},{y}")
        label = loc.get("label") or description
        return f"Clicked '{label}' at ({x}, {y})."

    def control_mouse(self, value: str, target: str = "") -> str:
        user32 = ctypes.windll.user32
        v = (value or "click").strip().lower()

        if v == "scroll_up":
            amt = int(target) if (target or "").strip().isdigit() else 480
            _scroll_wheel(abs(amt))
            return "Scrolled up."
        if v == "scroll_down":
            amt = int(target) if (target or "").strip().isdigit() else 480
            _scroll_wheel(-abs(amt))
            return "Scrolled down."

        if v == "drag" and "->" in (target or ""):
            try:
                p1, p2 = target.split("->", 1)
                x1, y1 = [int(p.strip()) for p in p1.split(",", 1)]
                x2, y2 = [int(p.strip()) for p in p2.split(",", 1)]
                user32.SetCursorPos(x1, y1)
                time.sleep(0.04)
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                steps = 12
                for i in range(1, steps + 1):
                    ix = int(x1 + (x2 - x1) * i / steps)
                    iy = int(y1 + (y2 - y1) * i / steps)
                    user32.SetCursorPos(ix, iy)
                    time.sleep(0.015)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return f"Dragged from ({x1}, {y1}) to ({x2}, {y2})."
            except ValueError:
                return f"Invalid drag coordinates: {target}"

        # Optional "x,y" target moves cursor before clicking
        moved_msg = ""
        if "," in (target or ""):
            try:
                x, y = [int(p.strip()) for p in target.split(",", 1)]
                user32.SetCursorPos(x, y)
                time.sleep(0.04)
                moved_msg = f" at ({x}, {y})"
                if v == "move":
                    return f"Moved mouse to ({x}, {y})."
            except ValueError:
                if v == "move":
                    return f"Invalid coordinates: {target}"

        if v in ("click", "left_click"):
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return f"Clicked mouse{moved_msg}."
        if v == "double_click":
            for _ in range(2):
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.05)
            return f"Double-clicked mouse{moved_msg}."
        if v == "right_click":
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            return f"Right-clicked mouse{moved_msg}."
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

    def execute_plan(self, plan: Dict, memory=None, vision=None, plugin_manager=None) -> List[str]:
        """Executes all steps in a Brain Plan dictionary in order and returns status logs."""
        steps = plan.get("steps") or []
        if not steps:
            return []

        logs: List[str] = []
        actions = [s.get("action") for s in steps]
        has_play = "play" in actions
        has_search = "search" in actions

        pending_query: Optional[str] = None
        last_opened_app: Optional[str] = None
        last_typed_step: Optional[Dict] = None
        total = len(steps)
        for idx, step in enumerate(steps):
            action = step.get("action")
            target = step.get("target") or ""
            query = step.get("query") or ""
            value = step.get("value") or ""

            if action == "open_app":
                # Skip opening bare web-app homepage if we are about to search/play on it
                if (has_search or has_play) and target.lower() in WEB_APPS:
                    continue
                logs.append(self.open_app(target))
                last_opened_app = target
                # If followed by interactive steps, wait for window & focus it automatically
                if idx + 1 < total:
                    next_act = steps[idx + 1].get("action")
                    if next_act in ("type_text", "hotkey", "click_element", "mouse"):
                        self.wait_for_window(target, timeout=6.0)
                    elif next_act not in ("wait", "focus_window"):
                        time.sleep(0.8)
            elif action == "focus_window":
                logs.append(self.focus_window(target or last_opened_app or ""))
            elif action == "wait":
                if target:
                    ok = self.wait_for_window(target, timeout=6.0)
                    logs.append(f"{target} is ready." if ok else f"Waited for {target}.")
                else:
                    try:
                        secs = max(0.1, min(10.0, float(value or "1.0")))
                    except ValueError:
                        secs = 1.0
                    time.sleep(secs)
                    logs.append(f"Waited {secs:.1f}s.")
            elif action == "close_app":
                logs.append(self.close_app(target))
            elif action == "open_url":
                logs.append(self.open_url(value))
                if idx + 1 < total:
                    time.sleep(0.8)
            elif action == "see_screen":
                if vision is not None:
                    logs.append(vision.analyze_screen(query or "What is on my screen?"))
                else:
                    logs.append("Vision engine is not attached.")
            elif action == "click_element":
                logs.append(self.click_element(query or target, value or "click", vision=vision))
                if idx + 1 < total:
                    time.sleep(0.2)
            elif action == "verify_screen":
                if vision is not None and query:
                    ok, summary = vision.verify_screen(query)
                    # Retry once if verification failed right after typing
                    if not ok and last_typed_step is not None:
                        if last_opened_app:
                            self.focus_window(last_opened_app)
                        _type_unicode(
                            last_typed_step.get("query") or "",
                            replace=(last_typed_step.get("value") or "").lower() == "replace",
                            press_enter=(last_typed_step.get("value") or "").lower() == "enter",
                        )
                        time.sleep(0.4)
                        ok, summary = vision.verify_screen(query)
                    logs.append(f"Verified: {summary}" if ok else f"Screen check: {summary}")
                else:
                    logs.append("Verification skipped (vision unavailable).")
            elif action == "get_time":
                from datetime import datetime
                logs.append("It's " + datetime.now().strftime("%I:%M %p").lstrip("0") + ".")
            elif action == "get_date":
                from datetime import datetime
                logs.append("Today is " + datetime.now().strftime("%A, %d %B %Y") + ".")
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
            elif action == "hotkey":
                combo = value or query or target
                if _press_hotkey(combo):
                    logs.append(f"Pressed {combo}.")
                    time.sleep(0.12)
                else:
                    logs.append(f"Unknown shortcut: {combo}.")
            elif action == "type_text":
                v_low = (value or "").strip().lower()
                _type_unicode(
                    query,
                    replace=(v_low == "replace"),
                    press_enter=(v_low == "enter"),
                )
                last_typed_step = step
                logs.append(f"Typed: '{query}'.")
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
            elif action == "plugin_op":
                if plugin_manager is not None:
                    logs.append(plugin_manager.handle_action(value or "list", target, query))
                else:
                    logs.append("Plugin manager is not attached.")
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
