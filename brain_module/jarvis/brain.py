"""The Brain. Text in -> validated Plan out (with a spoken reply).

mode "model"  : YOUR trained model decides; rules are only a safety net
                (used when the model is offline or returns an invalid plan)
mode "rules"  : rule-based planner only (the baseline for the report)
mode "hybrid" : rules for very clear commands, the model for everything else
mode "auto"   : same as "model" (default)
"""
import os
import time
from datetime import datetime
from typing import Dict, Optional

from .llm import BadModelOutput, JarvisModel, ModelUnavailable
from .planner import RulesPlanner
from .schema import Action, Intent, Plan, enforce_safety

NEEDS_APP = (Action.PAUSE, Action.RESUME, Action.NEXT, Action.PREVIOUS)


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
    return "Done."


class Brain:
    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or os.getenv("BRAIN_MODE", "auto")).lower()
        if self.mode == "auto":
            self.mode = "model"
        self.rules = RulesPlanner()
        self.model = JarvisModel()
        self.sessions: Dict[str, Dict] = {}   # remembers the last app per user/session

    def plan(self, text: str, session_id: str = "default") -> Dict:
        start = time.perf_counter()
        ctx = self.sessions.setdefault(session_id, {})
        used, note = "rules", None

        if self.mode == "rules":
            plan = self.rules.plan(text, ctx)
        else:
            plan = None
            if self.mode == "hybrid":
                fast = self.rules.plan(text, ctx)
                if fast.intent != Intent.UNKNOWN and fast.confidence >= 0.9:
                    plan = fast
            if plan is None:
                try:
                    plan, used = self.model.plan(text), "model"
                except (ModelUnavailable, BadModelOutput) as e:
                    plan, note = self.rules.plan(text, ctx), f"model failed, used rules: {e}"

        plan = enforce_safety(plan)           # shutdown etc. always need confirmation

        # "pause it" / "next song": fill in the app from earlier in the conversation
        for s in plan.steps:
            if s.action in NEEDS_APP and not s.target:
                s.target = ctx.get("last_app")

        plan.reply = make_reply(plan)

        for s in plan.steps:                  # remember the app in use
            if s.action in (Action.OPEN_APP, Action.PLAY) and s.target:
                ctx["last_app"] = s.target

        return {
            "plan": plan.model_dump(mode="json"),
            "backend": used,
            "note": note,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }