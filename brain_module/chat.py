"""Type commands like the STT would give them. 'quit' to stop."""
import json
import sys

from jarvis.brain import Brain

brain = Brain()
if not brain.ollama_available:
    print(f"JARVIS Brain mode: {brain.mode} (fast offline rule-based engine)")
    print("[INFO] Ollama is not active on localhost:11434. Operating in 100% offline rule-based mode.")
    print("       (To enable Sahaj's fine-tuned LLM, run 'ollama serve' in another terminal)")
else:
    print(f"JARVIS Brain mode: {brain.mode} (powered by local LLM)")

print("Type a command (e.g. 'open chrome', 'play song', 'what is the time', 'help'), or 'quit' to exit.\n")

while True:
    try:
        text = input("You > ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting JARVIS Brain...")
        break

    if text.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not text:
        continue

    out = brain.plan(text, session_id="chat")
    print(json.dumps(out["plan"], indent=2, ensure_ascii=False))
    print(f"\nJarvis says: {out['plan']['reply']}")
    print(f"[answered by: {out['backend']} | {out['latency_ms']} ms]\n")
    if out.get("note"):
        print(f"[note] {out['note']}\n")
