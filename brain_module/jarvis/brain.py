"""The Brain. Text in -> validated Plan out (with a spoken reply).

mode "model"  : YOUR trained model decides; rules are only a safety net
                (used when the model is offline or returns an invalid plan)
mode "rules"  : rule-based planner only (the baseline for the report)
mode "hybrid" : rules for very clear commands, the model for everything else
mode "auto"   : same as "model" (default)
"""
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional

from .llm import BadModelOutput, GeminiModel, JarvisModel, ModelUnavailable
from .planner import RulesPlanner
from .schema import Action, Intent, Plan, enforce_safety

NEEDS_APP = (Action.PAUSE, Action.RESUME, Action.NEXT, Action.PREVIOUS)
_EXPLICIT_SEARCH_RE = re.compile(r"\b(?:search|google|look\s*up|browse|dhundo|dhoondo)\b", re.IGNORECASE)
_CHAT_QUESTION_RE = re.compile(
    r"^\s*(?:tell\s+me|what\s+(?:is|are|was|were)|who\s+(?:is|are|was|were)|"
    r"why\s+|how\s+(?:does|do|did|is|are|to)|explain\b|define\b|describe\b|write\b|can\s+you\s+tell\b)",
    re.IGNORECASE,
)


def make_reply(plan: Plan) -> str:
    """A short sentence for the TTS module to speak."""
    if plan.intent in (Intent.CLARIFY, Intent.UNKNOWN) and plan.clarification_question:
        return plan.clarification_question
    if plan.reply and plan.intent == Intent.CHAT:
        return plan.reply          # the model wrote the answer itself
    if not plan.steps:
        return plan.reply or "Okay."
    first = plan.steps[0]
    if first.action == Action.GET_TIME:
        return "It's " + datetime.now().strftime("%I:%M %p").lstrip("0") + "."
    if first.action == Action.GET_DATE:
        return "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."
    if plan.needs_confirmation:
        return f"Do you really want me to {first.value} the laptop?"
    play = next((s for s in plan.steps if s.action == Action.PLAY), None)
    if play:
        q = next((s.query for s in plan.steps if s.action == Action.SEARCH), None)
        return f"Playing {q} on {play.target}." if q else f"Playing on {play.target}."
    search = next((s for s in plan.steps if s.action == Action.SEARCH), None)
    if search:
        return f"Searching {search.target} for {search.query}."
    if first.action == Action.OPEN_APP:
        return f"Opening {first.target}."
    if first.action == Action.CLOSE_APP:
        return f"Closing {first.target}."
    if first.action == Action.OPEN_URL:
        return f"Opening {first.value}."
    if first.action == Action.VOLUME:
        return f"Volume {first.value}."
    if first.action == Action.BROWSER:
        labels = {
            "new_tab": "Opening a new tab.",
            "close_tab": "Closing the current tab.",
            "reopen_tab": "Reopening the closed tab.",
            "next_tab": "Switching to the next tab.",
            "prev_tab": "Switching to the previous tab.",
            "refresh": "Refreshing the page.",
            "back": "Going back.",
            "forward": "Going forward.",
            "scroll_down": "Scrolling down.",
            "scroll_up": "Scrolling up.",
            "scroll_top": "Scrolling to the top.",
            "scroll_bottom": "Scrolling to the bottom.",
            "zoom_in": "Zooming in.",
            "zoom_out": "Zooming out.",
            "zoom_reset": "Resetting zoom.",
            "incognito": "Opening an incognito window.",
            "history": "Opening browser history.",
            "downloads": "Opening downloads.",
            "bookmark": "Bookmarking this page.",
            "fullscreen": "Toggling fullscreen.",
        }
        return labels.get(first.value or "", "Done.")
    return "Done."


def ensure_ollama_running() -> bool:
    """Checks if Ollama is listening on localhost:11434; if not, automatically launches it."""
    import socket
    import subprocess

    for host in ("127.0.0.1", "localhost"):
        try:
            with socket.create_connection((host, 11434), timeout=0.25):
                return True
        except (OSError, TimeoutError):
            pass

    ollama_exe = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if os.path.exists(ollama_exe):
        try:
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen([ollama_exe, "serve"], creationflags=DETACHED_PROCESS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(15):
                time.sleep(0.2)
                try:
                    with socket.create_connection(("127.0.0.1", 11434), timeout=0.25):
                        return True
                except (OSError, TimeoutError):
                    pass
        except Exception:
            pass
    return False


class Brain:
    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or os.getenv("BRAIN_MODE", "auto")).lower()
        self.rules = RulesPlanner()
        self.gemini = GeminiModel()
        self.model = JarvisModel()
        self.sessions: Dict[str, Dict] = {}   # remembers the last app per user/session

        self.gemini_available = self.gemini.is_configured
        # Only start heavy local Ollama if Gemini is not configured (saves ~1.6GB RAM!)
        if self.gemini_available:
            self.ollama_available = False
        else:
            ensure_ollama_running()
            self.ollama_available = self.model.is_reachable()

        if self.mode == "auto":
            self.mode = "hybrid" if (self.gemini_available or self.ollama_available) else "rules"

    def _active_llm(self):
        """Returns (backend_name, llm_client) preferring Gemini, with Ollama fallback."""
        if self.gemini_available:
            return "gemini", self.gemini
        if self.ollama_available:
            return "model", self.model
        return None, None

    def plan(self, text: str, session_id: str = "default") -> Dict:
        start = time.perf_counter()
        ctx = self.sessions.setdefault(session_id, {})
        history = ctx.setdefault("history", [])
        used, note = "rules", None

        if self.mode == "rules" or (not self.gemini_available and not self.ollama_available):
            plan = self.rules.plan(text, ctx)
            used = "rules"
            note = None
        else:
            plan = None
            if self.mode == "hybrid":
                fast = self.rules.plan(text, ctx)
                if fast.intent != Intent.UNKNOWN and fast.confidence >= 0.9:
                    plan = fast
            if plan is None:
                backend_name, llm = self._active_llm()
                if llm is not None:
                    try:
                        if backend_name == "gemini":
                            plan, used = llm.plan(text, history=history), "gemini"
                        elif _CHAT_QUESTION_RE.match(text) and not _EXPLICIT_SEARCH_RE.search(text):
                            answer = llm.chat_answer(text)
                            plan, used = Plan(intent=Intent.CHAT, steps=[], reply=answer, confidence=1.0), backend_name
                        else:
                            plan, used = llm.plan(text), backend_name

                        has_search_step = any(s.action == Action.SEARCH for s in plan.steps)
                        if (plan.intent == Intent.SEARCH_WEB or has_search_step) and not _EXPLICIT_SEARCH_RE.search(text):
                            answer = llm.chat_answer(text, history=history) if backend_name == "gemini" else llm.chat_answer(text)
                            plan = Plan(intent=Intent.CHAT, steps=[], reply=answer, confidence=1.0)
                    except (ModelUnavailable, BadModelOutput) as e:
                        # Automatic offline fallback to Ollama if Gemini fails (e.g., no internet)
                        if backend_name == "gemini":
                            ensure_ollama_running()
                            if self.model.is_reachable():
                                self.ollama_available = True
                                try:
                                    if _CHAT_QUESTION_RE.match(text) and not _EXPLICIT_SEARCH_RE.search(text):
                                        ans = self.model.chat_answer(text)
                                        plan, used = Plan(intent=Intent.CHAT, steps=[], reply=ans, confidence=1.0), "model"
                                    else:
                                        plan, used = self.model.plan(text), "model"
                                except (ModelUnavailable, BadModelOutput) as e2:
                                    plan, note = self.rules.plan(text, ctx), f"models failed, used rules: {e2}"
                            else:
                                plan, note = self.rules.plan(text, ctx), f"gemini failed ({e}), used rules"
                        else:
                            plan, note = self.rules.plan(text, ctx), f"model failed, used rules: {e}"
                else:
                    plan = self.rules.plan(text, ctx)

        plan = enforce_safety(plan)           # shutdown etc. always need confirmation

        # "pause it" / "next song": fill in the app from earlier in the conversation
        for s in plan.steps:
            if s.action in NEEDS_APP and not s.target:
                s.target = ctx.get("last_app")

        plan.reply = make_reply(plan)

        # Save turn to session memory (max 10 messages = 5 turns)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": plan.reply or ""})
        if len(history) > 10:
            ctx["history"] = history[-10:]

        for s in plan.steps:                  # remember the app in use
            if s.action in (Action.OPEN_APP, Action.PLAY) and s.target:
                ctx["last_app"] = s.target

        return {
            "plan": plan.model_dump(mode="json"),
            "backend": used,
            "note": note,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }