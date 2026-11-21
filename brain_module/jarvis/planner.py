"""Stage 3: rule-based planner.

Turns the cleaned clauses from normalize.py into a validated Plan.
Fast, offline, and good for clear commands. Anything it does not
understand becomes intent "unknown" (later stages let an LLM handle those).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from .normalize import APP_ALIASES, MEDIA_APPS, SEARCH_APPS, resolve_app, split_clauses
from .schema import Action, Intent, Plan, Step

DEFAULT_MEDIA_APP = os.getenv("BRAIN_DEFAULT_MEDIA_APP", "youtube")
DEFAULT_SEARCH_TARGET = "google"

KNOWN_APPS = set(APP_ALIASES.values()) | set(APP_ALIASES.keys())
GENERIC_MEDIA_WORDS = {"music", "song", "a song", "some music", "something", "some song", "video", "a video"}

_DOMAIN = re.compile(r"^(https?://)?[\w\-]+(\.[\w\-]+)+(/\S*)?$")


def _split_app(rest: str) -> Tuple[str, Optional[str]]:
    """Split 'kesariya on spotify' -> ('kesariya', 'spotify') only if the tail is a known app."""
    # the greedy (.*) makes this match the LAST preposition: "weather in vadodara on google"
    m = re.match(r"^(?P<q>.*)\s+(?:on|in|from|using|through|via|at)\s+(?P<app>.+)$", rest)
    if m:
        cand = resolve_app(m.group("app"))
        if cand in KNOWN_APPS:
            return m.group("q").strip(), cand
    return rest.strip(), None


def _url(x: str) -> str:
    return x if x.startswith("http") else f"https://{x}"


class RulesPlanner:
    def plan(self, text: str, ctx: Optional[Dict] = None) -> Plan:
        ctx = ctx or {}
        clauses = split_clauses(text)
        steps: List[Step] = []
        ctx_app: Optional[str] = ctx.get("last_app")  # app from earlier turns (for "pause", "next")
        utt_app: Optional[str] = None                 # app opened in THIS sentence (for "search")
        unparsed = 0
        clarification: Optional[str] = None
        seen = set()  # which intents we saw, for primary-intent choice

        for clause in clauses:
            c = clause.strip()
            if not c:
                continue

            # --- open / go to ---
            m = re.match(r"^(?:open|launch|start|run)\s+(?P<x>.+)$", c)
            if m:
                x = m.group("x").strip()
                if _DOMAIN.match(x):
                    steps.append(Step(action=Action.OPEN_URL, value=_url(x)))
                    seen.add(Intent.OPEN_WEBSITE)
                else:
                    app = resolve_app(x)
                    steps.append(Step(action=Action.OPEN_APP, target=app))
                    ctx_app = utt_app = app
                    seen.add(Intent.OPEN_APP)
                continue
            m = re.match(r"^(?:go to|goto|visit|browse)\s+(?P<x>\S+)$", c)
            if m:
                x = m.group("x")
                if _DOMAIN.match(x):
                    steps.append(Step(action=Action.OPEN_URL, value=_url(x)))
                    seen.add(Intent.OPEN_WEBSITE)
                else:
                    unparsed += 1
                continue

            # --- close ---
            m = re.match(r"^(?:close|quit|exit|kill)\s+(?P<x>.+)$", c)
            if m:
                steps.append(Step(action=Action.CLOSE_APP, target=resolve_app(m.group("x"))))
                seen.add(Intent.CLOSE_APP)
                continue

            # --- play ---
            m = re.match(r"^(?:play|put on|listen to)(?:\s+(?P<rest>.+))?$", c)
            if m:
                rest = (m.group("rest") or "").strip()
                if not rest and steps and steps[-1].action == Action.OPEN_APP and steps[-1].target in MEDIA_APPS:
                    # "open yt and play" -> nothing to play yet, ask
                    clarification = f"What would you like me to play on {steps[-1].target}?"
                    continue
                if rest in ("", "it", "this", "that"):
                    steps.append(Step(action=Action.RESUME, target=ctx_app))
                    seen.add(Intent.CONTROL_PLAYBACK)
                    continue
                query, app = _split_app(rest)
                app = app or (ctx_app if ctx_app in MEDIA_APPS else None) or DEFAULT_MEDIA_APP
                if query in GENERIC_MEDIA_WORDS or not query:
                    clarification = f"What would you like me to play on {app}?"
                    continue
                if not (steps and steps[-1].action == Action.OPEN_APP and steps[-1].target == app):
                    steps.append(Step(action=Action.OPEN_APP, target=app))
                steps.append(Step(action=Action.SEARCH, target=app, query=query))
                steps.append(Step(action=Action.PLAY, target=app, value="first_result"))
                ctx_app = utt_app = app
                seen.add(Intent.PLAY_MEDIA)
                continue

            # --- search ---
            m = re.match(r"^(?:search(?:\s+for)?|look\s*up|google|find)\s+(?P<rest>.+)$", c)
            if m:
                query, app = _split_app(m.group("rest"))
                if not app:
                    app = utt_app if utt_app in (SEARCH_APPS - {"chrome"}) else DEFAULT_SEARCH_TARGET
                steps.append(Step(action=Action.SEARCH, target=app, query=query))
                seen.add(Intent.SEARCH_WEB)
                continue

            # --- playback controls ---
            if re.match(r"^(?:pause|stop|hold)(?:\s+(?:it|this|the\s+(?:song|music|video)|music|song|video|playback))?$", c):
                steps.append(Step(action=Action.PAUSE, target=ctx_app))
                seen.add(Intent.CONTROL_PLAYBACK)
                continue
            if re.match(r"^(?:resume|continue|unpause)(?:\s+(?:it|this|the\s+(?:song|music|video)|music|song|video|playback))?$", c):
                steps.append(Step(action=Action.RESUME, target=ctx_app))
                seen.add(Intent.CONTROL_PLAYBACK)
                continue
            if re.match(r"^(?:next|skip)(?:\s+(?:song|track|video|one|this))?$", c):
                steps.append(Step(action=Action.NEXT, target=ctx_app))
                seen.add(Intent.CONTROL_PLAYBACK)
                continue
            if re.match(r"^(?:previous|prev|go back)(?:\s+(?:song|track|video|one))?$", c):
                steps.append(Step(action=Action.PREVIOUS, target=ctx_app))
                seen.add(Intent.CONTROL_PLAYBACK)
                continue

            # --- volume ---
            m = re.match(r"^(?:volume\s+(?P<d1>up|down)|(?:increase|raise|turn up)\s+(?:the\s+)?volume|(?:decrease|lower|reduce|turn down)\s+(?:the\s+)?volume)$", c)
            if m:
                up = bool(re.match(r"^(?:volume up|increase|raise|turn up)", c))
                steps.append(Step(action=Action.VOLUME, value="up" if up else "down"))
                seen.add(Intent.SYSTEM_CONTROL)
                continue
            if re.match(r"^mute(?:\s+(?:it|the\s+(?:volume|sound)|sound))?$", c):
                steps.append(Step(action=Action.VOLUME, value="mute"))
                seen.add(Intent.SYSTEM_CONTROL)
                continue
            if re.match(r"^unmute(?:\s+(?:it|the\s+(?:volume|sound)|sound))?$", c):
                steps.append(Step(action=Action.VOLUME, value="unmute"))
                seen.add(Intent.SYSTEM_CONTROL)
                continue

            # --- system (risky: always confirmed by enforce_safety) ---
            m = re.match(r"^(?:(?P<a>shut\s*down|shutdown|restart|reboot|sleep|lock)|turn off)(?:\s+(?:the\s+|my\s+)?(?:laptop|computer|pc|system|screen))?$", c)
            if m:
                word = (m.group("a") or "shutdown").replace(" ", "")
                word = {"reboot": "restart"}.get(word, word)
                steps.append(Step(action=Action.SYSTEM, value=word))
                seen.add(Intent.SYSTEM_CONTROL)
                continue

            # --- typing ---
            m = re.match(r"^type\s+(?P<x>.+)$", c)
            if m:
                steps.append(Step(action=Action.TYPE_TEXT, query=m.group("x")))
                seen.add(Intent.SYSTEM_CONTROL)
                continue

            unparsed += 1

        # ---- build the plan ----
        if clarification:
            return Plan(intent=Intent.CLARIFY, steps=[], clarification_question=clarification, confidence=0.9)
        if not steps:
            return Plan(
                intent=Intent.UNKNOWN,
                clarification_question="Sorry, I didn't understand that. Could you say it again?",
                confidence=0.0,
            )

        for intent in (
            Intent.PLAY_MEDIA, Intent.SEARCH_WEB, Intent.OPEN_WEBSITE, Intent.SYSTEM_CONTROL,
            Intent.CONTROL_PLAYBACK, Intent.CLOSE_APP, Intent.OPEN_APP,
        ):
            if intent in seen:
                primary = intent
                break
        else:
            primary = Intent.UNKNOWN

        return Plan(intent=primary, steps=steps, confidence=0.5 if unparsed else 0.95)