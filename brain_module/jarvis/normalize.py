"""Text clean-up for noisy speech-to-text output, including simple Hinglish."""
from __future__ import annotations

import re

# Spoken/short forms -> canonical app or site name.
APP_ALIASES = {
    "yt": "youtube",
    "you tube": "youtube",
    "youtube": "youtube",
    "insta": "instagram",
    "instagram": "instagram",
    "wa": "whatsapp",
    "whatsapp": "whatsapp",
    "whats app": "whatsapp",
    "what's app": "whatsapp",
    "spotify": "spotify",
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "google": "google",
    "gmail": "gmail",
    "notepad": "notepad",
    "calculator": "calculator",
    "calc": "calculator",
    "vscode": "vscode",
    "vs code": "vscode",
    "code": "vscode",
    "explorer": "file explorer",
    "file explorer": "file explorer",
    "files": "file explorer",
    "word": "word",
    "excel": "excel",
    "powerpoint": "powerpoint",
    "ppt": "powerpoint",
    "telegram": "telegram",
    "netflix": "netflix",
    "prime video": "prime video",
    "amazon": "amazon",
    "wynk": "wynk music",
    "gaana": "gaana",
    "jiosaavn": "jiosaavn",
    "saavn": "jiosaavn",
}

# Apps/sites where a search or play makes sense (used for context decisions).
MEDIA_APPS = {"youtube", "spotify", "wynk music", "gaana", "jiosaavn", "netflix", "prime video"}
SEARCH_APPS = MEDIA_APPS | {"google", "amazon", "chrome"}

FILLERS = [
    "hey jarvis", "ok jarvis", "okay jarvis", "jarvis", "please", "pls", "plz",
    "can you", "could you", "would you", "will you", "uh", "um", "umm", "hmm",
    "yaar", "bhai", "zara", "na", "kindly",
]

# Hinglish verb phrases (object-verb order) -> canonical English verb.
# Each entry: (regex for verb phrase, english verb)
HINGLISH_VERBS = [
    (r"khol\s*do|khol\s*de|kholo|kholna|khol\s*dena", "open"),
    (r"chala\s*do|chala\s*de|chalao|chalana|baja\s*do|baja\s*de|bajao|laga\s*do|laga\s*de|lagao", "play"),
    (r"band\s*karo|band\s*kar\s*do|band\s*kar\s*de|band\s*kardo", "close"),
    (r"dhundo|dhoondo|dhundh\s*do|search\s*karo|search\s*kar\s*do", "search"),
]

CLAUSE_SPLIT = re.compile(r"\s*(?:,|\band then\b|\bthen\b|\band\b|\baur\b|\bphir\b|\bfir\b)\s*")


def _strip_fillers(t: str) -> str:
    for f in sorted(FILLERS, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resolve_app(name: str | None) -> str | None:
    if not name:
        return None
    n = re.sub(r"\s+", " ", name.strip().lower())
    if n in APP_ALIASES:                      # exact alias first, e.g. "whats app"
        return APP_ALIASES[n]
    n = re.sub(r"^(the|my)\s+", "", n)
    n = re.sub(r"\s+(app|application|website|site)$", "", n)
    return APP_ALIASES.get(n, n)


def normalize_text(text: str) -> str:
    """Lowercase, drop fillers/punctuation, keep apostrophes and dots in URLs."""
    t = text.lower().strip()
    t = re.sub(r"[\"“”!?;]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = _strip_fillers(t)
    t = re.sub(r"\s*,\s*", ", ", t)          # tidy spaces around commas
    t = re.sub(r"^[\s,.]+|[\s,.]+$", "", t)  # no leading/trailing commas or dots
    return t


def hinglish_to_english_clause(clause: str) -> str:
    """Turn 'X kholo' -> 'open X', 'Y chala do' -> 'play Y', 'X par Y chala do' -> 'play Y on X'."""
    c = clause.strip()
    # remove common Hindi glue that doesn't matter for planning
    c = re.sub(r"\b(ka|ke|ki)\s+(song|gaana|gaane|gana|video)\b", "", c)
    c = re.sub(r"\b(mujhe|mere liye|meri)\b", " ", c)
    c = re.sub(r"\s+", " ", c).strip()

    # media controls / volume first, so "agla gaana chalao" is not read as "play ..."
    if re.search(r"\b(agla|agle|next)\b.*\b(song|gaana|gaane|gana|track)\b", c):
        return "next"
    if re.search(r"\b(pichla|pichle|previous|prev)\b.*\b(song|gaana|gaane|gana|track)\b", c):
        return "previous"
    if re.search(r"\b(ruk\s*jao|ruko|rok\s*do|rok\s*de)\b", c):
        return "pause"
    if re.search(r"\b(volume|awaaz|awaz|aawaz)\b", c):
        if re.search(r"\b(badhao|badha\s*do|badha\s*de|zyada|jyada|up|increase)\b", c):
            return "volume up"
        if re.search(r"\b(kam|ghatao|ghata\s*do|down|decrease|lower)\b", c):
            return "volume down"

    for verb_re, eng in HINGLISH_VERBS:
        m = re.search(rf"^(?P<obj>.*?)\s*\b(?:{verb_re})\b\s*$", c)
        if not m:
            continue
        obj = m.group("obj").strip()
        if eng == "play":
            # "spotify par tum hi ho" / "spotify pe tum hi ho" / "spotify mein ..."
            pm = re.match(r"^(?P<app>.+?)\s+(?:par|pe|me|mein|se|pr)\s+(?P<q>.+)$", obj)
            if pm:
                return f"play {pm.group('q').strip()} on {pm.group('app').strip()}"
            return f"play {obj}".strip()
        if eng == "close":
            if re.search(r"\b(song|gaana|gaane|gana|music|video|playback)\b", obj):
                return "pause"
            return f"close {obj}".strip()
        return f"{eng} {obj}".strip()

    return c


# Splits on "and", "then", "aur", commas... and KEEPS the separator text.
_SPLIT_WITH_SEP = re.compile(r"(\s*(?:,|\band then\b|\bthen\b|\band\b|\baur\b|\bphir\b|\bfir\b)\s*)")

# A piece that starts like this is a new command.
_COMMAND_START = re.compile(
    r"^(?:open|launch|start|run|go to|goto|visit|browse|close|quit|exit|kill|play|put on|listen to"
    r"|search|look\s*up|google|find|pause|stop|hold|resume|continue|unpause|next|skip|previous|prev"
    r"|go back|volume|increase|raise|turn|decrease|lower|reduce|mute|unmute|shut\s*down|shutdown"
    r"|restart|reboot|sleep|lock|type)\b"
)
# Commands that carry free text (a song title or search words).
_FREE_TEXT_START = re.compile(r"^(?:play|put on|listen to|search|look\s*up|google|find)\b")


def split_clauses(text: str) -> list[str]:
    """Normalize, split into ordered clauses, convert Hinglish verbs to English.

    A piece that does not start like a command is glued back onto the previous
    play/search clause, so "play rock and roll" is not split into "rock" + "roll".
    """
    t = normalize_text(text)
    # alias multi-word spoken forms before splitting, so "you tube" survives
    t = re.sub(r"\byou tube\b", "youtube", t)
    t = re.sub(r"\bwhat'?s app\b", "whatsapp", t)

    raws: list[str] = []
    pending_sep = ""
    for i, piece in enumerate(_SPLIT_WITH_SEP.split(t)):
        if i % 2 == 1:                       # odd positions are the separators
            pending_sep = piece
            continue
        if not piece.strip():
            continue
        english = hinglish_to_english_clause(piece)
        if (
            raws
            and not _COMMAND_START.match(english)
            and _FREE_TEXT_START.match(hinglish_to_english_clause(raws[-1]))
        ):
            raws[-1] = raws[-1] + pending_sep + piece   # part of the same song/search text
        else:
            raws.append(piece)
        pending_sep = ""
    return [hinglish_to_english_clause(r) for r in raws]