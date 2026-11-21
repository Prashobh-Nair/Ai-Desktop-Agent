from jarvis.brain import Brain

brain = Brain(mode="rules")


def show(text, session="demo"):
    out = brain.plan(text, session)
    p = out["plan"]
    print(f"\nYou said : {text!r}")
    print(f"Intent   : {p['intent']}   confirm={p['needs_confirmation']}")
    for s in p["steps"]:
        details = {k: v for k, v in s.items() if v is not None and k != "action"}
        print(f"  step   : {s['action']:10} {details}")
    print(f"Reply    : {p['reply']}")
    return p


show("open yt and play kesariya")
show("yaar youtube kholo aur arijit ka song chala do")
show("spotify par tum hi ho chala do")
show("search weather in vadodara")
show("shutdown the laptop")
show("open yt and play")
show("tell me a joke")

print("\n--- Follow-ups remember the app ---")
show("open spotify and play believer", session="s1")
p = show("pause", session="s1")
assert p["steps"][0]["target"] == "spotify"

print("\n--- Search ignores the app from an earlier turn ---")
show("open spotify and play believer", session="s2")
p = show("search weather in vadodara", session="s2")
assert p["steps"][0]["target"] == "google"
p = show("open youtube and search lofi", session="s2")
assert p["steps"][1]["target"] == "youtube"

print("\n--- Self-check ---")
assert show("open yt and play kesariya")["intent"] == "play_media"
assert show("shutdown the laptop")["needs_confirmation"] is True
assert show("open yt and play")["intent"] == "clarify"
assert show("tell me a joke")["intent"] == "unknown"
assert brain.plan("open youtube and search lofi", "s3")["plan"]["reply"] == "Searching youtube for lofi."
print("\nAll Stage 3 checks passed")


p = brain.plan("play rock and roll from yt", "s4")["plan"]
assert p["steps"][1]["query"] == "rock and roll" and p["confidence"] >= 0.9
print("Rock and roll check passed")