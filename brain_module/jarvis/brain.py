"""The Brain. Text in -> validated Plan out (with a spoken reply).

Pure AI-model routing (no RulesPlanner):
- Primary (API): Google Gemini (fast, high-intelligence, zero local RAM)
- Fallback (Local): Fine-tuned Ollama model (`jarvis:latest`) when offline
"""
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .llm import BadModelOutput, GeminiModel, GroqBrainModel, JarvisModel, ModelUnavailable
from .schema import Action, Intent, Plan, enforce_safety
from src.memory_manager import MemoryManager

NEEDS_APP = (Action.PAUSE, Action.RESUME, Action.NEXT, Action.PREVIOUS)
_EXPLICIT_SEARCH_RE = re.compile(r"\b(?:search|google|look\s*up|browse|dhundo|dhoondo)\b", re.IGNORECASE)
_CHAT_QUESTION_RE = re.compile(
    r"^\s*(?:tell\s+me|what\s+(?:is|are|was|were)|who\s+(?:is|are|was|were)|"
    r"why\s+|how\s+(?:does|do|did|is|are|to)|explain\b|define\b|describe\b|write\b|can\s+you\s+tell\b)",
    re.IGNORECASE,
)


def make_reply(plan: Plan, preserve_llm_reply: bool = False) -> str:
    """A short sentence for the TTS module to speak."""
    if plan.intent in (Intent.CLARIFY, Intent.UNKNOWN) and plan.clarification_question:
        return plan.clarification_question
    if plan.reply and plan.intent == Intent.CHAT:
        return plan.reply          # the model wrote the answer itself
    if not plan.steps:
        return plan.reply or "Got it."
    first = plan.steps[0]
    if first.action == Action.GET_TIME:
        return "It's " + datetime.now().strftime("%I:%M %p").lstrip("0") + "."
    if first.action == Action.GET_DATE:
        return "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."
    if plan.needs_confirmation:
        if preserve_llm_reply and plan.reply:
            return plan.reply
        return f"Just checking — do you want me to {first.value or 'run that'}?"
    if preserve_llm_reply and plan.reply:
        return plan.reply
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
    if first.action == Action.BRIGHTNESS:
        return f"Brightness {first.value}."
    if first.action == Action.WALLPAPER:
        return f"Changing wallpaper to {first.query}." if first.query else "Changing your wallpaper."
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
        return labels.get(first.value or "", plan.reply or "Done.")
    return plan.reply or "Done."


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
        self.groq = GroqBrainModel()
        self.gemini = GeminiModel()
        self.model = JarvisModel()
        self.memory = MemoryManager()
        self.sessions: Dict[str, Dict] = {}   # remembers the last app per user/session

        self.groq_available = self.groq.is_configured
        self.gemini_available = self.gemini.is_configured
        # Only start heavy local Ollama if no cloud LLM is configured (saves ~1.6GB RAM!)
        if (self.groq_available or self.gemini_available) and self.mode != "model":
            self.ollama_available = False
        else:
            ensure_ollama_running()
            self.ollama_available = self.model.is_reachable()

        if self.mode == "auto":
            if self.groq_available:
                self.mode = "groq"
            elif self.gemini_available:
                self.mode = "gemini"
            else:
                self.mode = "model"

    def _active_llm(self):
        """Returns (backend_name, llm_client) preferring Groq (fastest), then Gemini, then Ollama."""
        if self.mode != "model":
            if self.groq_available:
                return "groq", self.groq
            if self.gemini_available:
                return "gemini", self.gemini
        if self.ollama_available:
            return "model", self.model
        return None, None

    @staticmethod
    def _unavailable_plan(reason: str) -> Plan:
        msg = "I couldn't reach Gemini or local Ollama right now. Please check your internet or start Ollama."
        return Plan(intent=Intent.CLARIFY, steps=[], clarification_question=msg, reply=msg, confidence=0.0)

    def plan(self, text: str, session_id: str = "default") -> Dict:
        start = time.perf_counter()
        ctx = self.sessions.setdefault(session_id, {})
        history = self.memory.get_history(session_id)
        mem_context = self.memory.retrieve_relevant(text)
        used, note = "unavailable", None

        backend_name, llm = self._active_llm()
        if llm is None:
            ensure_ollama_running()
            if self.model.is_reachable():
                self.ollama_available = True
                backend_name, llm = "model", self.model

        if llm is not None:
            try:
                if backend_name in ("groq", "gemini"):
                    plan, used = llm.plan(text, history=history, memory_context=mem_context), backend_name
                elif _CHAT_QUESTION_RE.match(text) and not _EXPLICIT_SEARCH_RE.search(text):
                    answer = llm.chat_answer(text)
                    plan, used = Plan(intent=Intent.CHAT, steps=[], reply=answer, confidence=1.0), backend_name
                else:
                    plan, used = llm.plan(text), backend_name

                has_search_step = any(s.action == Action.SEARCH for s in plan.steps)
                has_play_step = any(s.action == Action.PLAY for s in plan.steps)
                # Only suppress bare web-search if NO play step (search+play is intentional multi-step)
                if (plan.intent == Intent.SEARCH_WEB or has_search_step) and not has_play_step and not _EXPLICIT_SEARCH_RE.search(text):
                    answer = llm.chat_answer(text, history=history, memory_context=mem_context) if backend_name in ("groq", "gemini") else llm.chat_answer(text)
                    plan = Plan(intent=Intent.CHAT, steps=[], reply=answer, confidence=1.0)
            except (ModelUnavailable, BadModelOutput) as e:
                # Fallback chain: Groq -> Gemini -> Local Ollama
                if backend_name == "groq" and self.gemini_available:
                    try:
                        plan, used = self.gemini.plan(text, history=history, memory_context=mem_context), "gemini"
                        has_search_step = any(s.action == Action.SEARCH for s in plan.steps)
                        has_play_step = any(s.action == Action.PLAY for s in plan.steps)
                        if (plan.intent == Intent.SEARCH_WEB or has_search_step) and not has_play_step and not _EXPLICIT_SEARCH_RE.search(text):
                            answer = self.gemini.chat_answer(text, history=history, memory_context=mem_context)
                            plan = Plan(intent=Intent.CHAT, steps=[], reply=answer, confidence=1.0)
                    except (ModelUnavailable, BadModelOutput) as eg:
                        e = eg
                        backend_name = "gemini"
                if used == "unavailable":
                    if backend_name in ("groq", "gemini"):
                        ensure_ollama_running()
                        if self.model.is_reachable():
                            self.ollama_available = True
                            try:
                                if _CHAT_QUESTION_RE.match(text) and not _EXPLICIT_SEARCH_RE.search(text):
                                    ans = self.model.chat_answer(text)
                                    plan, used = Plan(intent=Intent.CHAT, steps=[], reply=ans, confidence=1.0), "model"
                                else:
                                    try:
                                        plan, used = self.model.plan(text), "model"
                                    except BadModelOutput:
                                        ans = self.model.chat_answer(text)
                                        plan, used = Plan(intent=Intent.CHAT, steps=[], reply=ans, confidence=1.0), "model"
                            except (ModelUnavailable, BadModelOutput) as e2:
                                plan, note = self._unavailable_plan(str(e2)), f"gemini & ollama failed: {e2}"
                        else:
                            plan, note = self._unavailable_plan(str(e)), f"gemini failed ({e}) and ollama unreachable"
                    elif isinstance(e, BadModelOutput):
                        try:
                            ans = llm.chat_answer(text)
                            plan, used = Plan(intent=Intent.CHAT, steps=[], reply=ans, confidence=1.0), backend_name
                        except (ModelUnavailable, BadModelOutput) as e2:
                            plan, note = self._unavailable_plan(str(e2)), f"model failed: {e2}"
                    else:
                        plan, note = self._unavailable_plan(str(e)), f"model failed: {e}"
        else:
            plan, note = self._unavailable_plan("no LLM available"), "no LLM backend available"

        plan = enforce_safety(plan)           # shutdown etc. always need confirmation

        # "pause it" / "next song": fill in the app from earlier in the conversation
        for s in plan.steps:
            if s.action in NEEDS_APP and not s.target:
                s.target = ctx.get("last_app")

        plan.reply = make_reply(plan, preserve_llm_reply=(used != "rules"))

        # Save turn to session memory
        self.memory.add_turn(session_id, text, plan.reply or "")

        for s in plan.steps:                  # remember the app in use
            if s.action in (Action.OPEN_APP, Action.PLAY) and s.target:
                ctx["last_app"] = s.target

        return {
            "plan": plan.model_dump(mode="json"),
            "backend": used,
            "note": note,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }