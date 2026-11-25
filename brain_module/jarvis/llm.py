"""Stage 7: talk to YOUR fine-tuned model running locally in Ollama.

No internet and no API key: Ollama runs on this laptop, the model is the one
we trained in Colab. The system prompt below MUST be identical to the one used
during training (make_dataset.py), otherwise accuracy drops.
"""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from pydantic import ValidationError

from .schema import Plan

SYSTEM_PROMPT = (
    "You are the brain of a desktop voice assistant. The user's command comes from speech "
    "recognition and may be noisy or Hinglish. Reply with ONLY a JSON plan."
)

CHAT_SYSTEM_PROMPT = (
    "You are Jarvis, a smart and polite AI desktop assistant developed by Prashobh Nair, "
    "Sahaj Patel, Zeel Vasoya, and Mayank. Answer the user's question directly and clearly "
    "in 1 to 3 concise sentences suitable for speaking aloud."
)

UNIFIED_GEMINI_PROMPT = (
    "You are Jarvis, an intelligent, helpful AI Desktop Assistant developed by Prashobh Nair, "
    "Sahaj Patel, Zeel Vasoya, and Mayank.\n"
    "You must ALWAYS reply with valid JSON matching this exact schema:\n"
    "{\n"
    '  "intent": "chat" | "open_app" | "close_app" | "play_media" | "search_web" | "open_website" | "browser_control" | "control_playback" | "system_control" | "info" | "clarify",\n'
    '  "steps": [{"action": "...", "target": null or str, "query": null or str, "value": null or str}],\n'
    '  "needs_confirmation": false,\n'
    '  "clarification_question": null,\n'
    '  "reply": "natural, concise spoken response",\n'
    '  "confidence": 0.0 to 1.0\n'
    "}\n\n"
    "Rules:\n"
    "1. For ANY general question, conversation, advice, coding help, joke, follow-up ('what about...', 'why?'), "
    "or 'tell me about X': set intent='chat', steps=[], and put a smart, clear 1-3 sentence answer in 'reply'.\n"
    "2. NEVER use action='search' unless the user explicitly asks to search/google/look up on the web.\n"
    "3. Screen Awareness: Whenever the user asks you to look at, read, describe, check, or explain their screen "
    "(e.g. 'can you see my screen', 'what is on my screen', 'what error is this', 'read this'), ALWAYS emit a "
    "step with action='see_screen' and query=their question. NEVER reply that you cannot see the screen.\n"
    "4. Multi-Tasking: If the user asks for 2 or more tasks in one request (using 'and', 'then', or commas, e.g. "
    "'open chrome and open notepad' or 'open notepad and type hello' or 'check time and play music'), ALWAYS "
    "include ALL requested steps in 'steps' in exact execution order.\n"
    "5. For desktop/browser control, use ONLY these actions:\n"
    "   - open_app (target: app name like chrome, notepad, calculator, vscode, spotify, youtube, gmail, whatsapp)\n"
    "   - close_app (target: app name)\n"
    "   - open_url (value: full https URL)\n"
    "   - search (target: google|youtube|wikipedia|amazon|flipkart|github, query: search terms)\n"
    "   - play (target: youtube|spotify, value: 'first_result', preceded by search step)\n"
    "   - browser (target: 'browser', value: new_tab|close_tab|reopen_tab|next_tab|prev_tab|refresh|back|forward|scroll_down|scroll_up|scroll_top|scroll_bottom|zoom_in|zoom_out|zoom_reset|incognito|history|downloads|bookmark|fullscreen)\n"
    "   - pause | resume | next | previous\n"
    "   - volume (value: up|down|mute|unmute)\n"
    "   - type_text (query: text to type)\n"
    "   - mouse (value: click|double_click|right_click|move, target: 'x,y' for move)\n"
    "   - run_terminal (query: shell command -> set needs_confirmation=true)\n"
    "   - see_screen (query: question about current screen)\n"
    "   - file_op (value: list|read|mkdir|write|search, target: relative path, query: content)\n"
    "   - memory_op (value: remember|forget|list, query: fact or search)\n"
    "   - git_op (value: status|log|open_github)\n"
    "   - calendar_op (value: add|list|open_gmail|open_calendar, query: title, target: when)\n"
    "   - web_fetch (target: weather|wikipedia, query: city or topic)\n"
    "   - system (value: shutdown|restart|sleep|lock -> set needs_confirmation=true)\n"
    "   - get_time | get_date\n"
)


def _load_env_file() -> None:
    """Loads key=value pairs from project .env into os.environ if not already set."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


class GroqBrainModel:
    """Ultra-fast cloud Brain backed by Groq (GPT-OSS-20B / 120B)."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model_name: str = "openai/gpt-oss-20b", timeout: float = 8.0):
        _load_env_file()
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model_name = model_name
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_groq_api_key_here")

    def _generate(
        self,
        system_prompt: str,
        user_text: str,
        json_mode: bool = False,
        history: list[dict] | None = None,
        memory_context: list[str] | None = None,
    ) -> str:
        if not self.is_configured:
            raise ModelUnavailable("GROQ_API_KEY is not configured.")

        full_sys = system_prompt
        if memory_context:
            full_sys += "\n\nRelevant Memory Context:\n" + "\n".join(memory_context)

        messages = [{"role": "system", "content": full_sys}]
        if history:
            for msg in history[-10:]:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_text})

        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2 if json_mode else 0.4,
            "max_tokens": 400,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        models_to_try: list[str] = []
        for m in (self.model_name, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_err: Exception | None = None
        for idx, model_id in enumerate(models_to_try):
            payload["model"] = model_id
            req = urllib.request.Request(
                self.ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Jarvis-Desktop-Agent/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                    return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                last_err = ModelUnavailable(f"Groq error {e.code} ({model_id}): {e.read().decode(errors='ignore')[:200]}")
                if e.code in (400, 404, 429, 500, 503, 504) and idx + 1 < len(models_to_try):
                    continue
                raise last_err from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, IndexError) as e:
                raise ModelUnavailable(str(e)) from e
        raise last_err or ModelUnavailable("Groq request failed.")

    def chat_answer(self, text: str, history: list[dict] | None = None, memory_context: list[str] | None = None) -> str:
        return self._generate(CHAT_SYSTEM_PROMPT, text, json_mode=False, history=history, memory_context=memory_context)

    def plan(self, text: str, history: list[dict] | None = None, memory_context: list[str] | None = None) -> Plan:
        raw = self._generate(UNIFIED_GEMINI_PROMPT, text, json_mode=True, history=history, memory_context=memory_context)
        try:
            return Plan.model_validate_json(raw)
        except ValidationError as e:
            raise BadModelOutput(f"Groq output did not match Plan schema: {e}") from e


class GeminiModel:
    """Fast cloud Brain backed by Google Gemini (3.1 Flash Lite)."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None, timeout: float = 15.0):
        _load_env_file()
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")).strip()
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def _generate(
        self,
        system_prompt: str,
        user_text: str,
        json_mode: bool = False,
        history: list[dict] | None = None,
        memory_context: list[str] | None = None,
    ) -> str:
        if not self.is_configured:
            raise ModelUnavailable("GEMINI_API_KEY is not configured.")

        gen_cfg: dict = {"temperature": 0.2 if json_mode else 0.4, "maxOutputTokens": 400}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        full_sys = system_prompt
        if memory_context:
            full_sys += "\n\nRelevant Memory Context:\n" + "\n".join(memory_context)

        contents = []
        if history:
            for msg in history[-10:]:
                role = "model" if msg.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        body = json.dumps({
            "systemInstruction": {"parts": [{"text": full_sys}]},
            "contents": contents,
            "generationConfig": gen_cfg,
        }).encode("utf-8")

        models_to_try: list[str] = []
        for m in (
            self.model_name,
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
        ):
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_err: Exception | None = None
        for idx, model_id in enumerate(models_to_try):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode(errors="ignore")[:200]
                last_err = ModelUnavailable(f"Gemini error {e.code} ({model_id}): {err_msg}")
                if e.code in (404, 429, 500, 503, 504) and idx + 1 < len(models_to_try):
                    continue
                raise last_err from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, IndexError) as e:
                raise ModelUnavailable(str(e)) from e

        raise last_err or ModelUnavailable("Gemini request failed.")

    def chat_answer(self, text: str, history: list[dict] | None = None, memory_context: list[str] | None = None) -> str:
        return self._generate(CHAT_SYSTEM_PROMPT, text, json_mode=False, history=history, memory_context=memory_context)

    def plan(self, text: str, history: list[dict] | None = None, memory_context: list[str] | None = None) -> Plan:
        raw = self._generate(UNIFIED_GEMINI_PROMPT, text, json_mode=True, history=history, memory_context=memory_context)
        try:
            return Plan.model_validate_json(raw)
        except ValidationError as e:
            raise BadModelOutput(f"Gemini output did not match Plan schema: {e}") from e


class ModelUnavailable(Exception):
    """Ollama is not running, or the model is not installed."""


class BadModelOutput(Exception):
    """The model answered, but not with a valid Plan."""


class JarvisModel:
    def __init__(self, name=None, url=None, timeout=60.0):
        self.name = name or os.getenv("BRAIN_MODEL", "jarvis")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def is_reachable(self) -> bool:
        """Fast non-blocking check whether local Ollama daemon is actively listening."""
        import socket
        from urllib.parse import urlparse
        try:
            parsed = urlparse(self.url)
            port = parsed.port or 11434
            for host in ("127.0.0.1", parsed.hostname or "127.0.0.1"):
                try:
                    with socket.create_connection((host, port), timeout=0.3):
                        return True
                except (OSError, TimeoutError):
                    continue
            return False
        except Exception:
            return False

    def warmup(self) -> bool:
        """Preloads the model into Ollama memory with a minimal 1-token call."""
        if not self.is_reachable():
            return False
        body = json.dumps({
            "model": self.name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 1, "num_ctx": 512, "num_thread": 8},
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=20.0):
                return True
        except Exception:
            return False

    def _chat(self, text: str) -> str:
        if not self.is_reachable():
            raise ModelUnavailable("Ollama daemon is not running on localhost:11434. Using deterministic rule-based planner.")

        body = json.dumps({
            "model": self.name,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": text}],
            "stream": False,
            "format": "json",                 # Ollama only lets the model write valid JSON
            "keep_alive": "10m",              # keep the model in memory between commands
            "options": {"temperature": 0, "num_predict": 140, "num_ctx": 512, "num_thread": 8},
        }).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except urllib.error.HTTPError as e:                  # e.g. 404 = model not created yet
            raise ModelUnavailable(f"Ollama error {e.code}: {e.read().decode(errors='ignore')[:200]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ModelUnavailable(str(e)) from e

    def chat_answer(self, text: str) -> str:
        """Ask the local model for a direct conversational answer (no JSON action plan)."""
        if not self.is_reachable():
            raise ModelUnavailable("Ollama daemon is not running on localhost:11434.")

        body = json.dumps({
            "model": self.name,
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 512, "num_thread": 8},
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            raise ModelUnavailable(f"Ollama error {e.code}: {e.read().decode(errors='ignore')[:200]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ModelUnavailable(str(e)) from e

    def plan(self, text: str) -> Plan:
        raw = self._chat(text)
        try:
            return Plan.model_validate_json(raw.strip())      # rejects invented intents/actions
        except (ValidationError, ValueError) as e:
            raise BadModelOutput(f"{raw.strip()[:120]!r} -> {str(e).splitlines()[0]}") from e
