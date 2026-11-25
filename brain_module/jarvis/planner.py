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
        clean_raw = re.sub(r"[^\w\s]", " ", text.strip().lower())
        clean_raw = re.sub(r"\s+", " ", clean_raw).strip()

        # --- wake words & empty invocation ---
        if clean_raw in ("jarvis", "hey jarvis", "ok jarvis", "okay jarvis", "hello jarvis", "hi jarvis", "yo jarvis"):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="Yes, I am here and listening. What can I do for you?",
                confidence=1.0,
            )

        # --- greetings & conversational talk (Intent.CHAT) ---
        if re.match(r"^(?:hello|hi|hey|greetings|good\s*(?:morning|afternoon|evening)|yo)\b", clean_raw):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="Hello! I am Jarvis, your autonomous desktop assistant. What can I do for you?",
                confidence=1.0,
            )
        if re.search(r"\b(?:how\s+are\s+you|how\s+are\s+you\s+doing|kya\s+haal\s+hai|kaise\s+ho)\b", clean_raw):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="I am running at optimal performance, ready for your commands!",
                confidence=1.0,
            )
        if re.search(r"\b(?:who\s+(?:has\s+)?(?:made|created|built|developed|coded)\s+(?:you|it|this|jarvis)|who\s+are\s+(?:your|the)\s+(?:creators|developers|makers)|kisne\s+banaya)\b", clean_raw):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="I was developed by Prashobh Nair, Sahaj Patel, Zeel Vasoya, and Mayank for our AI Autonomous Desktop Agent project.",
                confidence=1.0,
            )
        if re.search(r"\b(?:who\s+are\s+you|what\s+is\s+your\s+name|tum\s+kaun\s+ho|introduce\s+yourself)\b", clean_raw):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="I am Jarvis, the AI-Powered Autonomous Desktop Agent, developed by Prashobh Nair, Sahaj Patel, Zeel Vasoya, and Mayank.",
                confidence=1.0,
            )
        if re.search(r"\b(?:what\s+can\s+you\s+do|capabilities|features|commands|what\s+are\s+your\s+features|tum\s+kya\s+kar\s+sakte\s+ho)\b", clean_raw) or clean_raw in ("help", "features", "commands"):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="I can open applications, search Google, play music on YouTube, control media playback and volume, report the time and date, and manage system operations.",
                confidence=1.0,
            )
        if re.search(r"\b(?:thank\s+you|thanks|thank\s+you\s+so\s+much|shukriya|dhanyawad|good\s+job|great\s+job|awesome|nice|well\s+done)\b", clean_raw):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="You are very welcome! Let me know if you need anything else.",
                confidence=1.0,
            )
        if re.search(r"\b(?:are\s+you\s+(?:there|ready|listening|online)|system\s+status|status)\b", clean_raw) or clean_raw in ("status", "ready"):
            return Plan(
                intent=Intent.CHAT,
                steps=[],
                reply="All systems are nominal and ready for commands.",
                confidence=1.0,
            )

        # --- screen understanding (Intent.INFO -> Action.SEE_SCREEN) ---
        if re.search(
            r"\b(?:what(?:'s|\s+is)\s+on\s+(?:my|the)\s+screen|"
            r"(?:look\s+at|read|describe|check|see|analyze|explain)\s+(?:my|the)\s+screen|"
            r"what\s+am\s+i\s+looking\s+at|what\s+(?:app|window)\s+is\s+open)\b",
            clean_raw,
        ):
            return Plan(
                intent=Intent.INFO,
                steps=[Step(action=Action.SEE_SCREEN, query=text.strip())],
                reply="Let me take a look at your screen.",
                confidence=1.0,
            )

        # --- time & date (Intent.INFO) ---
        if re.search(r"\b(?:what(?:\s*is|'?s)?\s+(?:the\s+)?time|tell\s+(?:me\s+)?the\s+time|current\s+time|what\s+time\s+is\s+it|time\s+kya\s+hua|samay\s+kya\s+hua)\b", clean_raw) or clean_raw in ("time", "samay"):
            return Plan(
                intent=Intent.INFO,
                steps=[Step(action=Action.GET_TIME)],
                confidence=1.0,
            )
        if re.search(r"\b(?:what(?:\s*is|'?s)?\s+(?:the\s+|today'?s\s+)?date|tell\s+(?:me\s+)?the\s+date|today'?s\s+date|what\s+day\s+is\s+it|aaj\s+ka\s+din|aaj\s+ki\s+date)\b", clean_raw) or clean_raw in ("date", "today"):
            return Plan(
                intent=Intent.INFO,
                steps=[Step(action=Action.GET_DATE)],
                confidence=1.0,
            )

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

            # --- browser controls (checked before open/close so "open new tab" & "close tab" match here) ---
            BROWSER_PATTERNS = [
                (r"^(?:open\s+)?(?:a\s+)?new\s+tab$", "new_tab"),
                (r"^close\s+(?:this\s+|the\s+|current\s+)?tab$", "close_tab"),
                (r"^(?:reopen|restore)\s+(?:the\s+)?(?:last\s+|closed\s+)?tab$", "reopen_tab"),
                (r"^(?:next|switch)\s+tab$", "next_tab"),
                (r"^(?:previous|prev)\s+tab$", "prev_tab"),
                (r"^(?:refresh|reload)(?:\s+(?:the\s+|this\s+)?page)?$", "refresh"),
                (r"^(?:go\s+back|navigate\s+back|browser\s+back)$", "back"),
                (r"^(?:go\s+forward|navigate\s+forward|forward)$", "forward"),
                (r"^scroll\s+down(?:\s+(?:a\s+bit|more|page))?$", "scroll_down"),
                (r"^scroll\s+up(?:\s+(?:a\s+bit|more|page))?$", "scroll_up"),
                (r"^scroll\s+(?:to\s+)?(?:the\s+)?top$", "scroll_top"),
                (r"^scroll\s+(?:to\s+)?(?:the\s+)?bottom$", "scroll_bottom"),
                (r"^zoom\s+in$", "zoom_in"),
                (r"^zoom\s+out$", "zoom_out"),
                (r"^(?:reset\s+zoom|normal\s+zoom)$", "zoom_reset"),
                (r"^(?:open\s+)?(?:new\s+)?incognito(?:\s+(?:window|tab|mode))?$", "incognito"),
                (r"^(?:open|show)\s+(?:browser\s+)?history$", "history"),
                (r"^(?:open|show)\s+downloads$", "downloads"),
                (r"^bookmark(?:\s+(?:this\s+)?page)?$", "bookmark"),
                (r"^(?:toggle\s+)?full\s*screen$", "fullscreen"),
            ]
            matched_browser = next((val for pat, val in BROWSER_PATTERNS if re.match(pat, c)), None)
            if matched_browser:
                steps.append(Step(action=Action.BROWSER, target="browser", value=matched_browser))
                seen.add(Intent.BROWSER_CONTROL)
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
            m = re.match(r"^(?:search(?:\s+(?:for|about))?|look\s*up|google|find)\s+(?P<rest>.+)$", c)
            if m:
                rest = m.group("rest").strip()
                # Handle "in google about X" / "on youtube for X"
                m_pre = re.match(r"^(?:in|on|using|via)\s+(?P<app>\S+)\s+(?:for|about)\s+(?P<q>.+)$", rest)
                if m_pre and resolve_app(m_pre.group("app")) in KNOWN_APPS:
                    app = resolve_app(m_pre.group("app"))
                    query = m_pre.group("q").strip()
                else:
                    query, app = _split_app(rest)
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
            tip = "I didn't recognize that command. You can ask me to open apps (e.g. 'open chrome'), play music (e.g. 'play Believer'), search Google, or check the time."
            return Plan(
                intent=Intent.UNKNOWN,
                clarification_question=tip,
                reply=tip,
                confidence=0.0,
            )

        for intent in (
            Intent.PLAY_MEDIA, Intent.SEARCH_WEB, Intent.OPEN_WEBSITE, Intent.BROWSER_CONTROL,
            Intent.SYSTEM_CONTROL, Intent.CONTROL_PLAYBACK, Intent.CLOSE_APP, Intent.OPEN_APP,
        ):
            if intent in seen:
                primary = intent
                break
        else:
            primary = Intent.UNKNOWN

        return Plan(intent=primary, steps=steps, confidence=0.5 if unparsed else 0.95)