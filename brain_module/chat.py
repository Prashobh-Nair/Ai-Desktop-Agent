"""Type commands like the STT would give them. 'quit' to stop."""
import json

from jarvis.brain import Brain

brain = Brain()          # uses YOUR trained model; falls back to the rules if it is not available
print(f"JARVIS brain, mode: {brain.mode}  (type a command, 'quit' to exit)")
while True:
    text = input("\nYou > ").strip()
    if text.lower() in ("quit", "exit", "q"):
        break
    if not text:
        continue
    out = brain.plan(text, session_id="chat")
    print(json.dumps(out["plan"], indent=2, ensure_ascii=False))
    print(f"Jarvis says: {out['plan']['reply']}")
    print(f"[answered by: {out['backend']}   {out['latency_ms']} ms]")
    if out["note"]:
        print("[note]", out["note"])
