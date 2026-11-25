"""Stage 7: talk to YOUR fine-tuned model running locally in Ollama.

No internet and no API key: Ollama runs on this laptop, the model is the one
we trained in Colab. The system prompt below MUST be identical to the one used
during training (make_dataset.py), otherwise accuracy drops.
"""
import json
import os
import urllib.error
import urllib.request

from pydantic import ValidationError

from .schema import Plan

SYSTEM_PROMPT = (
    "You are the brain of a desktop voice assistant. The user's command comes from speech "
    "recognition and may be noisy or Hinglish. Reply with ONLY a JSON plan."
)

CHAT_SYSTEM_PROMPT = (
    "You are Jarvis, a smart and polite AI desktop assistant. "
    "Answer the user's question directly and clearly in 1 to 3 concise sentences suitable for speaking aloud."
)


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
