from jarvis.normalize import normalize_text, resolve_app, split_clauses

print("--- 1. Cleaning noisy speech text ---")
for t in ["Hey Jarvis, please open YT!", "uh open you tube", "Yaar youtube kholo"]:
    print(f"{t!r:35} -> {normalize_text(t)!r}")

print("\n--- 2. App short forms ---")
for a in ["yt", "insta", "whats app", "vs code", "the spotify app", "chrome"]:
    print(f"{a!r:20} -> {resolve_app(a)!r}")

print("\n--- 3. Splitting into clauses (English verbs) ---")
tests = [
    "open yt and play kesariya",
    "yaar youtube kholo aur arijit ka song chala do",
    "spotify par tum hi ho chala do",
    "chrome band karo",
    "agla gaana chalao",
    "volume badhao",
    "open youtube and play kesariya then pause it",
]
for t in tests:
    print(f"{t!r}\n    -> {split_clauses(t)}")

print("\n--- 4. Self-check ---")
assert normalize_text("Hey Jarvis, please open YT!") == "open yt"
assert resolve_app("whats app") == "whatsapp"
assert resolve_app("the spotify app") == "spotify"
assert split_clauses("open yt and play kesariya") == ["open yt", "play kesariya"]
assert split_clauses("yaar youtube kholo aur arijit ka song chala do") == ["open youtube", "play arijit"]
assert split_clauses("spotify par tum hi ho chala do") == ["play tum hi ho on spotify"]
assert split_clauses("agla gaana chalao") == ["next"]
print("All Stage 2 checks passed")


assert split_clauses("play rock and roll from yt") == ["play rock and roll from yt"]
assert split_clauses("open yt and play rock and roll") == ["open yt", "play rock and roll"]
assert split_clauses("play tom and jerry theme then pause") == ["play tom and jerry theme", "pause"]
print("Titles with 'and' stay together")