"""
Phase 4: Multi-Tier Memory Manager for JARVIS.

Manages:
1. Short-term conversation memory (per-session recent turns)
2. Long-term personal memory (persisted in memory/long_term.json)
3. Project-specific memory (auto-indexed workspace summary in memory/project_memory.json)
4. Keyword/recency retrieval & user-controlled remember/forget operations
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
LONG_TERM_FILE = MEMORY_DIR / "long_term.json"
PROJECT_MEMORY_FILE = MEMORY_DIR / "project_memory.json"


class MemoryManager:
    """Handles short-term, long-term personal, and project-specific memory."""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.long_term_path = self.memory_dir / "long_term.json"
        self.project_path = self.memory_dir / "project_memory.json"
        self.short_term: Dict[str, List[Dict[str, str]]] = {}
        self._ensure_project_memory()

    # ------------------------------------------------------------------
    # Short-Term Conversation Memory
    # ------------------------------------------------------------------
    def add_turn(self, session_id: str, user_text: str, assistant_reply: str, max_turns: int = 8) -> None:
        hist = self.short_term.setdefault(session_id, [])
        hist.append({"role": "user", "content": user_text})
        hist.append({"role": "assistant", "content": assistant_reply})
        if len(hist) > max_turns * 2:
            self.short_term[session_id] = hist[-(max_turns * 2):]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self.short_term.get(session_id, []))

    # ------------------------------------------------------------------
    # Long-Term Personal Memory
    # ------------------------------------------------------------------
    def _load_long_term(self) -> List[Dict]:
        if not self.long_term_path.exists():
            return []
        try:
            return json.loads(self.long_term_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_long_term(self, items: List[Dict]) -> None:
        self.long_term_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def remember(self, fact: str, category: str = "personal") -> str:
        clean = (fact or "").strip()
        if not clean:
            return "Nothing specified to remember."
        items = self._load_long_term()
        # Avoid exact duplicates
        if any(i.get("fact", "").lower() == clean.lower() for i in items):
            return f"I already remember that: {clean}"
        items.append({
            "id": int(time.time() * 1000),
            "category": category,
            "fact": clean,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save_long_term(items)
        return f"Got it, I will remember: {clean}"

    def forget(self, query: str) -> str:
        q = (query or "").strip().lower()
        items = self._load_long_term()
        if not q or q in ("all", "everything"):
            self._save_long_term([])
            return "Cleared all long-term personal memories."
        kept = [i for i in items if q not in i.get("fact", "").lower()]
        removed = len(items) - len(kept)
        self._save_long_term(kept)
        if removed:
            return f"Forgot {removed} matching memory item(s)."
        return f"No saved memory matched '{query}'."

    def list_memories(self) -> List[Dict]:
        return self._load_long_term()

    def retrieve_relevant(self, query: str, limit: int = 4) -> List[str]:
        """Retrieves the most relevant long-term & project memories for a query."""
        items = self._load_long_term()
        q_words = {w for w in (query or "").lower().split() if len(w) > 2}

        scored = []
        for idx, item in enumerate(items):
            fact = item.get("fact", "")
            f_words = set(fact.lower().split())
            overlap = len(q_words & f_words)
            scored.append((overlap, idx, fact))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        results = [f"- {fact}" for score, _, fact in scored[:limit] if score > 0 or len(items) <= limit]

        # Include brief project context if query mentions project/code/git/repo
        if q_words & {"project", "repo", "code", "git", "github", "jarvis", "stt", "tts", "brain", "file", "work"}:
            proj = self.get_project_summary()
            if proj:
                results.append(f"- Active Project: {proj}")
        return results

    # ------------------------------------------------------------------
    # Project-Specific Memory
    # ------------------------------------------------------------------
    def _ensure_project_memory(self) -> None:
        if self.project_path.exists():
            return
        self.refresh_project_memory()

    def refresh_project_memory(self) -> Dict:
        branch = "main"
        last_commit = ""
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(PROJECT_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            last_commit = subprocess.check_output(
                ["git", "log", "-1", "--oneline"],
                cwd=str(PROJECT_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass

        data = {
            "project_name": "AI-Powered Autonomous Desktop Agent",
            "workspace": str(PROJECT_ROOT),
            "git_branch": branch,
            "last_commit": last_commit,
            "modules": {
                "stt": "Prashobh Nair (Local Faster-Whisper 244M/74M)",
                "brain": "Sahaj Patel (Hybrid Rules + Gemini 3.1 Flash Lite + Ollama)",
                "tts": "Zeel Vasoya (Kokoro-82M + SAPI5)",
                "automation": "Mayank (Windows OS, Browser & SmolVLM-256M Vision)",
            },
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.project_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def get_project_summary(self) -> str:
        try:
            data = json.loads(self.project_path.read_text(encoding="utf-8"))
            return (
                f"{data.get('project_name')} on branch '{data.get('git_branch')}' "
                f"(latest: {data.get('last_commit')})"
            )
        except Exception:
            return "AI-Powered Autonomous Desktop Agent"
