"""
Phase 5: External Tools, File Management, GitHub, Calendar & Web APIs.

Provides safe, structured tool functions invoked by ActionExecutor and AutonomousRunner:
- File management (list, read, create folder/file, search inside workspace)
- Git & GitHub status/history inspection
- Local Calendar & Reminders (stored in memory/calendar.json) + Gmail/Calendar browser launch
- Live Weather & Web/API fetch
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALENDAR_FILE = PROJECT_ROOT / "memory" / "calendar.json"


def _safe_workspace_path(rel_path: str) -> Path:
    """Resolves a path safely within the project workspace."""
    clean = (rel_path or ".").strip().strip('"').strip("'")
    target = (PROJECT_ROOT / clean).resolve()
    if PROJECT_ROOT.resolve() not in target.parents and target != PROJECT_ROOT.resolve():
        raise ValueError("Path escapes the project workspace.")
    return target


# ---------------------------------------------------------------------------
# File Management
# ---------------------------------------------------------------------------

def file_op(op: str, target: str = ".", content: str = "") -> str:
    op = (op or "list").strip().lower()
    try:
        path = _safe_workspace_path(target)
        if op == "list":
            if not path.exists():
                return f"Folder '{target}' does not exist."
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            preview = ", ".join(entries[:20])
            more = f" (+{len(entries) - 20} more)" if len(entries) > 20 else ""
            return f"Files in {target}: {preview}{more}"
        if op == "read":
            if not path.is_file():
                return f"File '{target}' not found."
            text = path.read_text(encoding="utf-8", errors="ignore")[:800]
            return f"Contents of {target}:\n{text}"
        if op == "mkdir":
            path.mkdir(parents=True, exist_ok=True)
            return f"Created folder '{target}'."
        if op == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content or "", encoding="utf-8")
            return f"Saved file '{target}'."
        if op == "search":
            matches = [
                str(p.relative_to(PROJECT_ROOT))
                for p in PROJECT_ROOT.rglob(f"*{target}*")
                if ".venv" not in p.parts and ".git" not in p.parts
            ][:10]
            return "Matches: " + (", ".join(matches) if matches else "none found.")
        return f"Unknown file operation: {op}"
    except Exception as e:
        return f"File operation error: {e}"


# ---------------------------------------------------------------------------
# Git & GitHub Integration
# ---------------------------------------------------------------------------

def git_op(op: str = "status", target: str = "") -> str:
    op = (op or "status").strip().lower()
    try:
        if op == "status":
            out = subprocess.check_output(["git", "status", "-s"], cwd=str(PROJECT_ROOT), text=True).strip()
            return "Git working tree is clean." if not out else f"Git changes:\n{out}"
        if op == "log":
            out = subprocess.check_output(["git", "log", "-5", "--oneline"], cwd=str(PROJECT_ROOT), text=True).strip()
            return f"Recent commits:\n{out}"
        if op == "open_github":
            webbrowser.open("https://github.com/Prashobh-Nair/Ai-Desktop-Agent")
            return "Opened GitHub repository in browser."
        return f"Unknown git operation: {op}"
    except Exception as e:
        return f"Git error: {e}"


# ---------------------------------------------------------------------------
# Calendar & Gmail Integration
# ---------------------------------------------------------------------------

def _load_events() -> List[Dict]:
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CALENDAR_FILE.exists():
        return []
    try:
        return json.loads(CALENDAR_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_events(events: List[Dict]) -> None:
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR_FILE.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")


def calendar_op(op: str = "list", title: str = "", when: str = "") -> str:
    op = (op or "list").strip().lower()
    if op == "open_gmail":
        webbrowser.open("https://mail.google.com")
        return "Opened Gmail in your browser."
    if op == "open_calendar":
        webbrowser.open("https://calendar.google.com")
        return "Opened Google Calendar in your browser."
    if op == "add":
        events = _load_events()
        events.append({
            "id": int(time.time() * 1000),
            "title": title or "Reminder",
            "when": when or "Today",
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
        })
        _save_events(events)
        return f"Added calendar event: '{title}' ({when or 'Today'})."
    events = _load_events()
    if not events:
        return "Your local calendar has no upcoming events."
    summary = "; ".join(f"{e['title']} ({e['when']})" for e in events[-5:])
    return f"Upcoming events: {summary}"


# ---------------------------------------------------------------------------
# External Web / Weather APIs
# ---------------------------------------------------------------------------

def web_fetch(query: str = "", target: str = "weather") -> str:
    t = (target or "weather").strip().lower()
    try:
        if t == "weather":
            loc = urllib.parse.quote(query or "")
            url = f"https://wttr.in/{loc}?format=3"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                return resp.read().decode("utf-8", errors="ignore").strip()
        if t == "wikipedia":
            slug = urllib.parse.quote((query or "").strip().replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
            req = urllib.request.Request(url, headers={"User-Agent": "JarvisAgent/1.0"})
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read())
                return data.get("extract", "No Wikipedia summary found.")[:350]
        return f"Unsupported web_fetch target: {t}"
    except Exception as e:
        return f"External API fetch failed: {e}"
