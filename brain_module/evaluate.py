"""Stage 4/7: score the brain on a test file.

    python evaluate.py                                   # rules baseline on tests/test_set.json
    python evaluate.py --mode model                      # YOUR trained model (Ollama must be running)
    python evaluate.py --mode model --tests data/heldout.jsonl --show-fails

--tests accepts tests/test_set.json (hand-written) or a .jsonl file such as data/heldout.jsonl.

Per command we check:
    intent  : predicted intent == expected intent
    steps   : predicted steps == expected steps (action, target, query, value)
    confirm : needs_confirmation matches (only checked when the test sets it)
    full    : all of the above are right
For chat, only the intent and "no steps" are compared (reply wording may differ).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from jarvis.brain import Brain

TEST_PATH = Path(__file__).parent / "tests" / "test_set.json"


def load_tests(path: str):
    p = Path(path)
    if p.suffix == ".jsonl":                      # chat-format file made by make_dataset.py
        tests = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)["messages"]
            gold = json.loads(m[2]["content"])
            t = {"text": m[1]["content"], "intent": gold["intent"], "steps": gold.get("steps", [])}
            if gold.get("needs_confirmation"):
                t["needs_confirmation"] = True
            tests.append(t)
        return tests
    return json.loads(p.read_text())


def norm_step(s: dict) -> tuple:
    def g(key):
        v = s.get(key)
        return str(v).strip().lower() if v is not None else None
    return (g("action"), g("target"), g("query"), g("value"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rules", choices=["rules", "model", "hybrid"])
    ap.add_argument("--show-fails", action="store_true")
    ap.add_argument("--tests", default=str(TEST_PATH))
    args = ap.parse_args()

    tests = load_tests(args.tests)
    brain = Brain(mode=args.mode)

    n = intent_ok = steps_ok = confirm_ok = full_ok = fallback = 0
    by_intent = defaultdict(lambda: [0, 0])
    fails, latencies = [], []

    for i, t in enumerate(tests):
        out = brain.plan(t["text"], session_id=f"test-{i}")   # fresh memory for every test
        plan = out["plan"]
        latencies.append(out["latency_ms"])
        fallback += bool(out.get("note"))

        a = plan["intent"] == t["intent"]
        if t["intent"] == "chat":
            b = plan["steps"] == []
        else:
            b = [norm_step(s) for s in plan["steps"]] == [norm_step(s) for s in t["steps"]]
        c = plan["needs_confirmation"] == t["needs_confirmation"] if "needs_confirmation" in t else True

        n += 1
        intent_ok += a
        steps_ok += b
        confirm_ok += c
        full_ok += a and b and c
        by_intent[t["intent"]][0] += a and b and c
        by_intent[t["intent"]][1] += 1
        if not (a and b and c):
            fails.append((t, plan, out.get("note")))

    pct = lambda x: f"{100 * x / n:5.1f}%  ({x}/{n})"
    print(f"\nMode            : {args.mode}")
    print(f"Test file       : {args.tests}")
    print(f"Commands tested : {n}")
    print(f"Intent accuracy : {pct(intent_ok)}")
    print(f"Steps accuracy  : {pct(steps_ok)}")
    print(f"Confirmation    : {pct(confirm_ok)}")
    print(f"FULL match      : {pct(full_ok)}")
    print(f"Avg latency     : {sum(latencies) / len(latencies):.1f} ms   (max {max(latencies):.0f} ms)")
    if args.mode != "rules":
        print(f"Fell back to rules (model offline/invalid): {fallback}")
    print("\nFull match by intent:")
    for k, (ok, total) in sorted(by_intent.items()):
        print(f"  {k:18s} {ok}/{total}")

    if args.show_fails:
        print(f"\nWrong answers ({len(fails)}):")
        for t, plan, note in fails:
            print(f"- {t['text']!r}")
            print(f"    expected: {t['intent']}  {[norm_step(s) for s in t['steps']]}")
            print(f"    got     : {plan['intent']}  {[norm_step(s) for s in plan['steps']]}")
            if note:
                print(f"    note    : {note}")


if __name__ == "__main__":
    main()
