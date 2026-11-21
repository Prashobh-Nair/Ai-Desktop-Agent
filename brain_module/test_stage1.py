from pydantic import ValidationError

from jarvis.schema import Action, Intent, Plan, Step, enforce_safety

# 1. Build a plan by hand: "open yt and play kesariya"
plan = Plan(
    intent=Intent.PLAY_MEDIA,
    steps=[
        Step(action=Action.OPEN_APP, target="youtube"),
        Step(action=Action.SEARCH, target="youtube", query="kesariya"),
        Step(action=Action.PLAY, target="youtube", value="first_result"),
    ],
    reply="Playing kesariya on youtube.",
)
print("1. Plan as JSON:")
print(plan.model_dump_json(indent=2))

# 2. Valid JSON text coming from a model is accepted
raw = '{"intent": "close_app", "steps": [{"action": "close_app", "target": "chrome"}]}'
print("\n2. Parsed from text:", Plan.model_validate_json(raw).intent.value)

# 3. Invalid output is rejected (made-up action)
bad = '{"intent": "open_app", "steps": [{"action": "hack_pentagon"}]}'
try:
    Plan.model_validate_json(bad)
    print("\n3. ERROR: bad plan was accepted")
except ValidationError:
    print("\n3. Bad plan rejected correctly")

# 4. Safety rule: shutdown must always need confirmation
risky = Plan(intent=Intent.SYSTEM_CONTROL, steps=[Step(action=Action.SYSTEM, value="shutdown")])
print("4. needs_confirmation before:", risky.needs_confirmation)
print("   needs_confirmation after :", enforce_safety(risky).needs_confirmation)