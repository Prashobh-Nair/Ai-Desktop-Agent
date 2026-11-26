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
    VOLUME = "volume"          # value = up | down | mute | unmute | max | min | 0..100
    BRIGHTNESS = "brightness"  # value = up | down | max | min | 0..100
    WALLPAPER = "wallpaper"    # query = theme / description / filename / 'random'
    SYSTEM = "system"          # value = shutdown | restart | sleep | lock
    BROWSER = "browser"        # value = new_tab | close_tab | refresh | back | forward | scroll_down | ...
    TYPE_TEXT = "type_text"    # query = text to type, value = optional "replace" | "enter"
    HOTKEY = "hotkey"          # value = shortcut e.g. "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+s", "enter"
    FOCUS_WINDOW = "focus_window"  # target = app or window title
    WAIT = "wait"              # target = app/window name or value = seconds
    CLICK_ELEMENT = "click_element"  # query = UI element description, value = click|double_click|right_click
    VERIFY_SCREEN = "verify_screen"  # query = expected screen state to verify
    SPEAK = "speak"            # value = text for TTS
    GET_TIME = "get_time"      # code looks up the current time
    GET_DATE = "get_date"      # code looks up today's date
    SEE_SCREEN = "see_screen"  # local vision model inspects the current screen
    MOUSE = "mouse"            # value = click | double_click | right_click | move | drag | scroll_up | scroll_down
    RUN_TERMINAL = "run_terminal"  # query = shell command (requires permission)
    FILE_OP = "file_op"        # value = list | read | mkdir | write | search
    MEMORY_OP = "memory_op"    # value = remember | forget | list
    GIT_OP = "git_op"          # value = status | log | open_github
    CALENDAR_OP = "calendar_op"  # value = add | list | open_gmail | open_calendar
    WEB_FETCH = "web_fetch"    # target = weather | wikipedia, query = search
    PLUGIN_OP = "plugin_op"    # value = create | list | enable | disable | remove | reconnect | capabilities | health | use


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


# These system/shell actions are dangerous, so the user must always confirm them.
RISKY_SYSTEM_VALUES = {"shutdown", "restart", "sleep", "lock"}


def enforce_safety(plan: Plan) -> Plan:
    """Force needs_confirmation=True for risky steps, whatever the model said."""
    for step in plan.steps:
        if step.action == Action.SYSTEM and step.value in RISKY_SYSTEM_VALUES:
            plan.needs_confirmation = True
        elif step.action == Action.RUN_TERMINAL:
            plan.needs_confirmation = True
        elif step.action == Action.HOTKEY and (step.value or "").lower().replace(" ", "") in {"alt+f4", "ctrl+w"}:
            # Closing windows/tabs via hotkey is safe only if user asked; keep alt+f4 confirmed
            if (step.value or "").lower().replace(" ", "") == "alt+f4":
                plan.needs_confirmation = True
    return plan