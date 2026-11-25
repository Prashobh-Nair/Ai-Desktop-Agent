"""Stage 1: the command schema.

This is the contract between the AI brain (my part) and the automation
module (Mayank's part). The brain must ALWAYS produce a Plan, and every
Plan is checked by pydantic before it is sent anywhere.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What the user wants, in one word."""
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    PLAY_MEDIA = "play_media"
    SEARCH_WEB = "search_web"
    OPEN_WEBSITE = "open_website"
    CONTROL_PLAYBACK = "control_playback"
    SYSTEM_CONTROL = "system_control"
    BROWSER_CONTROL = "browser_control"
    CHAT = "chat"        # just talk, no computer action
    INFO = "info"        # needs live info from the computer (time, date)
    CLARIFY = "clarify"  # brain needs more info from the user
    UNKNOWN = "unknown"


class Action(str, Enum):
    """One small thing the computer can do."""
    OPEN_APP = "open_app"      # target = app name
    CLOSE_APP = "close_app"    # target = app name
    OPEN_URL = "open_url"      # value = full URL
    SEARCH = "search"          # target = app/site, query = text
    PLAY = "play"              # target = app, value = "first_result"
    PAUSE = "pause"
    RESUME = "resume"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME = "volume"          # value = up | down | mute | unmute
    SYSTEM = "system"          # value = shutdown | restart | sleep | lock
    BROWSER = "browser"        # value = new_tab | close_tab | refresh | back | forward | scroll_down | ...
    TYPE_TEXT = "type_text"    # query = text to type
    SPEAK = "speak"            # value = text for TTS
    GET_TIME = "get_time"      # code looks up the current time
    GET_DATE = "get_date"      # code looks up today's date


class Step(BaseModel):
    """One action with its details. Unused fields stay None."""
    action: Action
    target: Optional[str] = None
    query: Optional[str] = None
    value: Optional[str] = None


class Plan(BaseModel):
    """The full answer the brain gives for one user command."""
    intent: Intent
    steps: List[Step] = Field(default_factory=list)
    needs_confirmation: bool = False
    clarification_question: Optional[str] = None
    reply: Optional[str] = None  # short sentence for the TTS to speak
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# These system actions are dangerous, so the user must always confirm them.
RISKY_SYSTEM_VALUES = {"shutdown", "restart", "sleep", "lock"}


def enforce_safety(plan: Plan) -> Plan:
    """Force needs_confirmation=True for risky steps, whatever the model said."""
    for step in plan.steps:
        if step.action == Action.SYSTEM and step.value in RISKY_SYSTEM_VALUES:
            plan.needs_confirmation = True
    return plan