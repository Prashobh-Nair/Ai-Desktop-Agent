"""Stage 5: build the training data for the brain model.

    python make_dataset.py

Creates data/train.jsonl, data/val.jsonl and data/heldout.jsonl.
Each line is one training example in chat format:
    system  : short instruction
    user    : what the user said (noisy, like speech-to-text output)
    assistant: the JSON plan the model must learn to produce

Why generate data? A model learns from examples. We combine many sentence
templates (English, Hinglish, casual speech) with lists of apps, songs and
search topics, so the model sees thousands of different ways to say the same
thing and learns the PATTERN, not a fixed list of commands.

`heldout.jsonl` uses sentence templates that are NEVER in train/val. It tells
us honestly whether the model generalises to phrasing it has not seen.
"""
import json
import random
from collections import Counter
from pathlib import Path

from jarvis.schema import Plan, enforce_safety

random.seed(7)
OUT = Path("data")

SYSTEM_PROMPT = (
    "You are the brain of a desktop voice assistant. The user's command comes from speech "
    "recognition and may be noisy or Hinglish. Reply with ONLY a JSON plan."
)

# ------------------------------------------------------------------ vocabulary
# canonical app -> ways people SAY it
APP_SAY = {
    "youtube": ["youtube", "youtube", "yt", "you tube"],
    "spotify": ["spotify"],
    "jiosaavn": ["jiosaavn", "saavn"],
    "gaana": ["gaana"],
    "wynk music": ["wynk", "wynk music"],
    "chrome": ["chrome", "google chrome", "browser"],
    "instagram": ["instagram", "insta"],
    "whatsapp": ["whatsapp", "whats app"],
    "telegram": ["telegram"],
    "netflix": ["netflix"],
    "gmail": ["gmail"],
    "notepad": ["notepad"],
    "calculator": ["calculator", "calc"],
    "vscode": ["vs code", "vscode"],
    "safari": ["safari"],
    "terminal": ["terminal"],
    "notes": ["notes"],
    "calendar": ["calendar"],
    "finder": ["finder"],
    "word": ["word", "microsoft word"],
    "excel": ["excel"],
    "powerpoint": ["powerpoint", "ppt"],
}
MEDIA_APPS = ["youtube", "youtube", "youtube", "spotify", "spotify", "jiosaavn", "gaana", "wynk music"]
OPEN_APPS = list(APP_SAY)

SONGS = [
    "kesariya", "tum hi ho", "believer", "shape of you", "blinding lights", "channa mereya",
    "apna bana le", "kal ho naa ho", "senorita", "perfect", "see you again", "closer",
    "rock and roll", "tom and jerry theme", "baby", "lose yourself", "levitating",
    "kabira", "ghungroo", "raataan lambiyan", "tera ban jaunga", "agar tum saath ho",
    "stay", "starboy", "love me like you do", "someone you loved", "thunder", "faded",
    "despacito", "bad guy", "hawayein", "pehle bhi main", "maan meri jaan", "o bedardeya",
    "lofi beats", "arijit singh songs", "ap dhillon songs", "sidhu moosewala", "badshah songs",
    "relaxing music", "workout music", "bollywood hits", "punjabi songs", "old hindi songs",
    "gym motivation songs", "study music", "believer by imagine dragons", "sunflower", "dandelions",
    "heat waves", "unstoppable", "counting stars", "viva la vida", "shallow", "kar har maidan fateh",
    "naatu naatu", "chaleya", "jhoome jo pathaan", "zaalima", "deewani mastani", "tujhe kitna chahne lage",
]
TOPICS = [
    "weather in vadodara", "python tutorial", "best biryani places in vadodara", "machine learning courses",
    "nearest petrol pump", "how to make tea", "latest iphone price", "cricket score today",
    "flights to mumbai", "java vs python", "healthy breakfast ideas", "gym workout plan",
    "railway timetable", "best laptops under 60000", "how to tie a tie", "ipl schedule",
    "news today", "movie showtimes near me", "what is machine learning", "diwali date",
    "cheap hotels in goa", "learn guitar online", "bitcoin price", "football live score",
    "python list comprehension", "cooking pasta", "msu baroda admission", "online coding courses",
]
DOMAINS = ["youtube.com", "github.com", "gmail.com", "google.com", "netflix.com", "amazon.in",
           "flipkart.com", "linkedin.com", "wikipedia.org", "stackoverflow.com", "twitter.com",
           "instagram.com", "chat.openai.com", "hotstar.com", "zomato.com"]
SEARCH_APPS = ["google", "google", "google", "youtube"]

# ------------------------------------------------------------------ plan builders
def S(action, **kw):
    return {"action": action, **kw}

def plan_play(app, q):
    return {"intent": "play_media", "steps": [S("open_app", target=app), S("search", target=app, query=q),
                                               S("play", target=app, value="first_result")]}

def plan_open(app):   return {"intent": "open_app", "steps": [S("open_app", target=app)]}
def plan_close(app):  return {"intent": "close_app", "steps": [S("close_app", target=app)]}
def plan_search(app, q):
    return {"intent": "search_web", "steps": [S("search", target=app, query=q)]}
def plan_url(d):      return {"intent": "open_website", "steps": [S("open_url", value="https://" + d)]}
def plan_ctl(a):      return {"intent": "control_playback", "steps": [S(a)]}
def plan_vol(v):      return {"intent": "system_control", "steps": [S("volume", value=v)]}
def plan_sys(v):      return {"intent": "system_control", "steps": [S("system", value=v)], "needs_confirmation": True}
def plan_clarify(q):  return {"intent": "clarify", "steps": [], "clarification_question": q}
def plan_chat(r):     return {"intent": "chat", "steps": [], "reply": r}
def plan_info(a):     return {"intent": "info", "steps": [S(a)]}

def say(app):
    return random.choice(APP_SAY[app])

# ------------------------------------------------------------------ templates
# Each template: function(app, thing) -> utterance. TRAIN ones are seen in training;
# HELD ones are only used for heldout.jsonl (to test generalisation).
PLAY_APP_TRAIN = [
    "play {s} on {a}", "play {s} from {a}", "open {a} and play {s}", "{a} pe {s} chala do",
    "{a} par {s} baja do", "{a} kholo aur {s} chalao", "yaar {a} khol ke {s} laga do",
    "i want to listen to {s} on {a}", "can you play {s} on {a}", "start playing {s} on {a}",
    "put on {s} on {a}", "{a} mein {s} chalao", "launch {a} and play {s}", "play {s} using {a}",
    "{a} open karo aur {s} play karo", "mujhe {a} pe {s} sunna hai",
]
PLAY_APP_HELD = [
    "i feel like hearing {s} on {a}", "queue up {s} on {a}", "{a} mein {s} laga do",
    "get {a} running and play {s}", "throw on {s} from {a}",
]
PLAY_DEFAULT_TRAIN = [
    "play {s}", "play the song {s}", "can you play {s}", "i want to listen to {s}", "put on {s}",
    "{s} chala do", "{s} baja do", "{s} laga do", "mujhe {s} sunna hai", "{s} sunao",
    "play {s} for me", "i wanna hear {s}", "play some {s}", "start {s}", "{s} play karo",
    "mujhe {s} gaana sunna hai", "yaar {s} chalao",
]
PLAY_DEFAULT_HELD = ["i feel like hearing {s}", "throw on {s}", "{s} bajao yaar", "let's listen to {s}"]

OPEN_TRAIN = [
    "open {a}", "launch {a}", "start {a}", "{a} kholo", "{a} khol do", "{a} chalu karo",
    "can you open {a}", "please open {a}", "take me to {a}", "i want to use {a}", "open the {a} app",
    "{a} open karo", "{a} khol de bhai", "run {a}",
]
OPEN_HELD = ["fire up {a}", "get {a} running", "bring up {a}", "{a} ko kholna"]
CLOSE_TRAIN = [
    "close {a}", "quit {a}", "exit {a}", "{a} band karo", "{a} band kar do", "band karo {a}",
    "please close {a}", "can you close {a}", "shut {a}", "{a} ko band karna", "kill {a}",
]
CLOSE_HELD = ["get rid of {a}", "i am done with {a}", "turn off {a}", "{a} ko hata do"]
SEARCH_TRAIN = [
    "search {q}", "search for {q}", "google {q}", "look up {q}", "find {q}", "{q} search karo",
    "i want to search {q}", "can you search {q}", "please search {q}", "{q} dhundo", "search {q} on {a}",
    "search for {q} on {a}", "{a} pe {q} search karo", "{a} par {q} dhundo", "look up {q} on {a}",
    "tell me about {q} by searching", "do a search for {q}",
]
SEARCH_HELD = ["i need info on {q}", "can you check {q} online", "{q} ke baare mein search karo", "web search {q}"]
URL_TRAIN = ["go to {d}", "open {d}", "take me to {d}", "visit {d}", "{d} kholo", "browse {d}", "{d} open karo"]
URL_HELD = ["navigate to {d}", "pull up {d}", "{d} par jao"]

CONTROL = {
    "pause":    ["pause", "pause it", "pause the music", "pause the song", "pause the video", "stop the music",
                 "hold on pause it", "ruko", "gaana ruk jao", "rok do", "stop playing"],
    "resume":   ["resume", "resume it", "continue", "continue playing", "unpause", "play it again",
                 "resume the music", "phir se chalao", "chalu karo gaana"],
    "next":     ["next", "next song", "next track", "skip", "skip this song", "skip it", "play the next one",
                 "agla gaana", "agla gaana chalao", "next gaana lagao", "agla song"],
    "previous": ["previous", "previous song", "go back", "play the previous track", "previous track",
                 "pichla gaana", "pichla gaana lagao", "last song again", "back"],
}
CONTROL_HELD = {
    "pause": ["freeze it", "hold the music"], "resume": ["carry on", "keep going"],
    "next": ["change the song", "next one please"], "previous": ["go to the last track"],
}
VOLUME = {
    "up":     ["volume up", "increase the volume", "turn up the volume", "make it louder", "louder",
               "volume badhao", "awaaz badhao", "awaaz zyada karo", "raise the volume", "volume increase karo", "i can't hear it"],
    "down":   ["volume down", "decrease the volume", "turn down the volume", "make it quieter", "lower the volume",
               "volume kam karo", "awaaz kam karo", "awaaz ghatao", "reduce the volume", "too loud", "volume low karo"],
    "mute":   ["mute", "mute the volume", "mute it", "silence", "mute karo", "awaaz band karo", "be quiet"],
    "unmute": ["unmute", "unmute it", "unmute the volume", "sound on", "unmute karo", "awaaz chalu karo"],
}
VOLUME_HELD = {"up": ["crank it up", "a bit louder please"], "down": ["bring the volume down", "softer please"],
               "mute": ["kill the sound"], "unmute": ["bring the sound back"]}
SYSTEM = {
    "shutdown": ["shutdown", "shutdown the laptop", "shut down the computer", "turn off the laptop",
                 "turn off the computer", "power off", "laptop band karo", "computer band kar do", "shut down my pc"],
    "restart":  ["restart", "restart the laptop", "reboot the computer", "restart my pc", "laptop restart karo", "reboot"],
    "sleep":    ["sleep", "put the laptop to sleep", "go to sleep mode", "laptop ko sleep mode me daalo", "sleep the computer"],
    "lock":     ["lock the laptop", "lock my computer", "lock the screen", "lock", "laptop lock karo", "screen lock karo"],
}
SYSTEM_HELD = {"shutdown": ["switch the laptop off", "power down"], "restart": ["start the computer again"],
               "sleep": ["let the laptop rest"], "lock": ["secure my screen"]}

CLARIFY = [
    (["play something", "play some music", "play a song", "play music", "kuch chala do", "koi gaana chalao", "music chalao",
      "play anything", "gaana sunao"], "What would you like me to play?"),
    (["search", "search something", "google something", "kuch search karo", "i want to search"], "What should I search for?"),
    (["open", "open an app", "kuch kholo", "launch something", "open something"], "Which app should I open?"),
    (["close", "close it", "close the app", "band karo", "close something"], "Which app should I close?"),
    (["type", "type something", "kuch type karo"], "What should I type?"),
]
CLARIFY_APP_PLAY = ["open {a} and play", "{a} kholo aur gaana chalao", "{a} open karke kuch chalao",
                    "launch {a} and play something", "{a} pe kuch chala do"]

CHAT = [
    (["hi", "hello", "hey", "hey jarvis", "hello jarvis", "hi there", "hii", "namaste", "hey buddy", "yo"],
     ["Hello! How can I help you?", "Hi there! What can I do for you?", "Hey! I'm listening.", "Namaste! How can I help?"]),
    (["good morning", "good morning jarvis", "suprabhat"], ["Good morning! What can I do for you today?", "Good morning! Ready when you are."]),
    (["good afternoon", "good afternoon jarvis"], ["Good afternoon! How can I help?"]),
    (["good evening", "good evening jarvis"], ["Good evening! What can I do for you?"]),
    (["good night", "good night jarvis", "shubh ratri"], ["Good night! Sleep well.", "Good night! See you tomorrow."]),
    (["how are you", "how are you doing", "how r you", "kaise ho", "kya haal hai", "how's it going", "how do you do", "what's up", "sab theek"],
     ["I'm doing great, thank you! How can I help?", "I'm good! What would you like to do?", "All good here! What can I do for you?"]),
    (["who are you", "what are you", "tell me about yourself", "introduce yourself", "tum kaun ho", "aap kaun ho"],
     ["I'm Jarvis, a voice assistant built by a student team as a college project. I run fully on your laptop.",
      "I'm Jarvis, your personal voice assistant. I run on your laptop and help you open apps, play music and search."]),
    (["what is your name", "what's your name", "your name", "tumhara naam kya hai", "aapka naam kya hai"],
     ["My name is Jarvis.", "I'm Jarvis, nice to meet you!"]),
    (["what can you do", "help", "what do you do", "how can you help me", "tum kya kar sakte ho", "what are your features"],
     ["I can open and close apps, play songs on YouTube or Spotify, search the web, control volume and playback, and chat with you.",
      "I can open apps, play music, search the web, tell the time, control the volume, and answer simple questions."]),
    (["who made you", "who created you", "who built you", "tumhe kisne banaya", "who is your creator"],
     ["I was built by a team of four students as a college minor project.", "A team of students built me for their college project."]),
    (["thank you", "thanks", "thank you jarvis", "thanks a lot", "shukriya", "dhanyavad", "thanks buddy"],
     ["You're welcome!", "Happy to help!", "Anytime!"]),
    (["bye", "goodbye", "see you later", "bye jarvis", "alvida", "chalo bye", "see you"], ["Goodbye! Talk to you soon.", "Bye! Have a great day."]),
    (["i am bored", "i'm bored", "main bore ho raha hoon", "bore ho raha hai"],
     ["How about some music? Tell me a song and I'll play it.", "Let's fix that! I can play a song or look something up for you."]),
    (["i am sad", "i feel sad", "mujhe bura lag raha hai", "i'm feeling low"],
     ["I'm sorry you feel that way. Would you like me to play some relaxing music?", "That's tough. Want me to play something calming?"]),
    (["i am happy", "i'm so happy today", "i feel great", "mujhe bahut khushi hai"], ["That's wonderful to hear!", "Great! Glad you're happy."]),
    (["tell me a joke", "say a joke", "make me laugh", "koi joke sunao", "joke sunao", "can you tell me a joke"],
     ["Why do programmers prefer dark mode? Because light attracts bugs!", "Why did the computer go to the doctor? It caught a virus.",
      "I would tell you a UDP joke, but you might not get it."]),
    (["are you real", "are you human", "kya tum insaan ho"], ["I'm not human. I'm an AI assistant running on your laptop."]),
    (["do you sleep", "are you tired"], ["I never get tired. I'm always ready to help."]),
    (["i love you", "love you jarvis"], ["That's kind of you! I'm here whenever you need me."]),
    (["are you there", "can you hear me", "jarvis are you there", "hello are you there"], ["Yes, I'm here! How can I help?"]),
    (["what's the meaning of life", "meaning of life"], ["That's a big question! Some say 42, but I think it's about the people and things you care about."]),
    (["sorry", "my bad", "maaf karna"], ["No problem at all!"]),
    (["ok", "okay", "cool", "nice", "great", "accha", "theek hai"], ["Alright!", "Okay!", "Got it."]),
    (["never mind", "forget it", "cancel", "rehne do", "chhodo"], ["Okay, no problem.", "Alright, cancelled."]),
]
CHAT_HELD = [
    (["howdy", "what's good"], ["Hello! How can I help?"]),
    (["you are awesome", "you're the best"], ["Thank you! That's very kind."]),
    (["can you help me", "i need some help"], ["Of course! Tell me what you need."]),
]
KNOWLEDGE = [
    ("what is the capital of france", "The capital of France is Paris."),
    ("what is the capital of india", "The capital of India is New Delhi."),
    ("what is the capital of japan", "The capital of Japan is Tokyo."),
    ("who is the prime minister of india", None),   # changes over time -> skipped on purpose
    ("how many days are there in a week", "There are seven days in a week."),
    ("how many days are in a year", "A year has 365 days, or 366 in a leap year."),
    ("what is two plus two", "Two plus two is four."),
    ("what is 10 times 10", "Ten times ten is one hundred."),
    ("what is the largest planet", "Jupiter is the largest planet in our solar system."),
    ("which planet is closest to the sun", "Mercury is the closest planet to the sun."),
    ("what is the largest ocean", "The Pacific Ocean is the largest ocean."),
    ("how many continents are there", "There are seven continents."),
    ("what is the boiling point of water", "Water boils at 100 degrees Celsius at sea level."),
    ("who wrote ramayana", "The Ramayana is traditionally attributed to the sage Valmiki."),
    ("what is python", "Python is a popular programming language known for being simple and readable."),
    ("what is machine learning", "Machine learning is a way of teaching computers to learn patterns from data instead of programming every rule by hand."),
    ("what is artificial intelligence", "Artificial intelligence is the ability of computers to do tasks that normally need human intelligence, like understanding language."),
    ("what is the speed of light", "Light travels at about 300,000 kilometres per second."),
    ("who discovered gravity", "Isaac Newton is famous for describing gravity."),
    ("what is the national animal of india", "The national animal of India is the Bengal tiger."),
    ("how many states are there in india", "India has 28 states and 8 union territories."),
    ("what is the full form of cpu", "CPU stands for Central Processing Unit."),
    ("what is the full form of ram", "RAM stands for Random Access Memory."),
    ("what is the full form of ai", "AI stands for Artificial Intelligence."),
    ("how many hours are in a day", "There are 24 hours in a day."),
]
INFO = {
    "get_time": ["what time is it", "what's the time", "tell me the time", "time kya hua hai", "abhi kitne baje hain",
                 "current time", "what is the time now", "time batao", "kitne baje hain", "can you tell me the time"],
    "get_date": ["what is the date today", "what's today's date", "today's date", "aaj kya date hai", "what day is it",
                 "which day is today", "aaj kaun sa din hai", "tell me the date", "date batao", "what's the date"],
}
INFO_HELD = {"get_time": ["got the time?", "time check"], "get_date": ["what's today", "date check"]}

# ------------------------------------------------------------------ speech noise
PREFIX = ["", "", "", "", "hey jarvis ", "jarvis ", "please ", "uh ", "ok jarvis ", "yaar ", "hey jarvis please ", "um "]
SUFFIX = ["", "", "", "", "", " please", " yaar", " na", " thanks"]

def noisy(u: str) -> str:
    u = random.choice(PREFIX) + u + random.choice(SUFFIX)
    r = random.random()
    if r < 0.20:
        u = u.capitalize()
    elif r < 0.28:
        u = u.capitalize() + random.choice([".", "?", "!"])
    return u.strip()

def core(u: str) -> str:
    """The command without noise (no prefix/suffix words, case or punctuation) - used to compare sentences."""
    u = u.lower().strip(" .?!")
    for w in sorted({x.strip() for x in PREFIX if x}, key=len, reverse=True):
        if u.startswith(w + " "):
            u = u[len(w) + 1:]
    for w in sorted({x.strip() for x in SUFFIX if x}, key=len, reverse=True):
        if u.endswith(" " + w):
            u = u[: -len(w) - 1]
    return u.strip(" .?!")


def reserved_test_sentences():
    """Sentences from our hand-written test set. They must never be used for training."""
    path = Path("tests/test_set.json")
    return {core(c["text"]) for c in json.loads(path.read_text())} if path.exists() else set()


# ------------------------------------------------------------------ generation
def gen(held: bool):
    """Return a list of (utterance, plan_dict)."""
    out = []
    pick = lambda train, h: h if held else train
    n = (lambda a, b: a if not held else b)          # counts: train vs heldout

    # play with an app
    for _ in range(n(700, 60)):
        app, s = random.choice(MEDIA_APPS), random.choice(SONGS)
        out.append((random.choice(pick(PLAY_APP_TRAIN, PLAY_APP_HELD)).format(a=say(app), s=s), plan_play(app, s)))
    # play with no app -> youtube
    for _ in range(n(450, 40)):
        s = random.choice(SONGS)
        out.append((random.choice(pick(PLAY_DEFAULT_TRAIN, PLAY_DEFAULT_HELD)).format(s=s), plan_play("youtube", s)))
    # open / close
    for _ in range(n(450, 40)):
        app = random.choice(OPEN_APPS)
        out.append((random.choice(pick(OPEN_TRAIN, OPEN_HELD)).format(a=say(app)), plan_open(app)))
    for _ in range(n(220, 25)):
        app = random.choice(OPEN_APPS)
        out.append((random.choice(pick(CLOSE_TRAIN, CLOSE_HELD)).format(a=say(app)), plan_close(app)))
    # search
    for _ in range(n(480, 45)):
        q, app = random.choice(TOPICS), random.choice(SEARCH_APPS)
        t = random.choice(pick(SEARCH_TRAIN, SEARCH_HELD))
        if "{a}" in t:
            out.append((t.format(q=q, a=say(app) if app == "youtube" else "google"), plan_search(app, q)))
        else:
            out.append((t.format(q=q), plan_search("google", q)))
    # open website
    for _ in range(n(200, 20)):
        d = random.choice(DOMAINS)
        out.append((random.choice(pick(URL_TRAIN, URL_HELD)).format(d=d), plan_url(d)))
    # playback control, volume, system
    for _ in range(n(300, 30)):
        a = random.choice(list(CONTROL))
        pool = CONTROL_HELD[a] if held else CONTROL[a]
        out.append((random.choice(pool), plan_ctl(a)))
    for _ in range(n(260, 30)):
        v = random.choice(list(VOLUME))
        pool = VOLUME_HELD[v] if held else VOLUME[v]
        out.append((random.choice(pool), plan_vol(v)))
    for _ in range(n(220, 25)):
        v = random.choice(list(SYSTEM))
        pool = SYSTEM_HELD[v] if held else SYSTEM[v]
        out.append((random.choice(pool), plan_sys(v)))
    # info (time/date)
    for _ in range(n(120, 15)):
        a = random.choice(list(INFO))
        pool = INFO_HELD[a] if held else INFO[a]
        out.append((random.choice(pool), plan_info(a)))
    # chat
    chat_bank = CHAT_HELD if held else CHAT
    for _ in range(n(520, 40)):
        prompts, replies = random.choice(chat_bank)
        out.append((random.choice(prompts), plan_chat(random.choice(replies))))
    # knowledge (only in train: heldout facts would just test memorisation of facts)
    if not held:
        for q, a in KNOWLEDGE:
            if a:
                for _ in range(4):
                    out.append((q, plan_chat(a)))
        # clarify
        for prompts, question in CLARIFY:
            for _ in range(40):
                out.append((random.choice(prompts), plan_clarify(question)))
        for _ in range(120):
            app = random.choice(MEDIA_APPS)
            out.append((random.choice(CLARIFY_APP_PLAY).format(a=say(app)),
                        {"intent": "clarify", "steps": [],
                         "clarification_question": f"What would you like me to play on {app}?"}))
    else:
        for _ in range(20):
            out.append((random.choice(["play karo", "kuch sunao", "i want music"]), plan_clarify("What would you like me to play?")))
    return out


def to_target(plan: dict) -> str:
    """Compact JSON target: defaults left out so the model has less to write."""
    p = Plan.model_validate(plan)            # raises if the plan breaks the schema
    p = enforce_safety(p)
    d = p.model_dump(mode="json", exclude_none=True)
    d.pop("confidence", None)
    for s in d["steps"]:
        for k in [k for k, v in s.items() if v is None]:
            s.pop(k)
    if not d["needs_confirmation"]:
        d.pop("needs_confirmation")
    if not d["steps"]:
        d["steps"] = []
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def build(pairs, seen=None, reserved=frozenset()):
    rows, keys = [], set()
    for utt, plan in pairs:
        if core(utt) in reserved:            # keep our test set clean
            continue
        utt = noisy(utt)
        key = utt.lower()
        if key in keys or (seen and key in seen):
            continue
        keys.add(key)
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utt},
            {"role": "assistant", "content": to_target(plan)},
        ]})
    random.shuffle(rows)
    return rows, keys


def write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    reserved = reserved_test_sentences()
    train_all, train_keys = build(gen(held=False), reserved=reserved)
    heldout, _ = build(gen(held=True), seen=train_keys)      # never overlaps with training
    cut = int(len(train_all) * 0.95)
    train, val = train_all[:cut], train_all[cut:]
    write(OUT / "train.jsonl", train)
    write(OUT / "val.jsonl", val)
    write(OUT / "heldout.jsonl", heldout)

    intents = Counter(json.loads(r["messages"][2]["content"])["intent"] for r in train_all)
    print(f"train: {len(train)}   val: {len(val)}   heldout (unseen phrasing): {len(heldout)}")
    print("intent mix in training data:")
    for k, v in intents.most_common():
        print(f"  {k:18s} {v}")
    print("\nSample examples:")
    for r in random.sample(train, 6):
        print("  USER :", r["messages"][1]["content"])
        print("  MODEL:", r["messages"][2]["content"])


if __name__ == "__main__":
    main()
