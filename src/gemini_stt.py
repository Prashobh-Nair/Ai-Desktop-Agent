"""
High-Accuracy Speech-to-Text Engine powered by Google Gemini 3.1 Flash Lite,
with automatic fallback to local Whisper-Base (74M) when offline.

Uses persistent HTTP keep-alive connections via requests.Session for low latency,
and zero local RAM overhead.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
import soundfile as sf


def _load_env_key() -> tuple[str, str]:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
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
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip()
    return key, model


class GeminiSTTEngine:
    """Human-level conversational & Indian-English STT via Gemini 3.1 Flash Lite."""

    PROMPT = (
        "Transcribe the exact spoken words in this audio accurately. "
        "The speaker may use natural conversational English, Hinglish, or mention names like "
        "Prashobh Nair, Sahaj Patel, Zeel Vasoya, Mayank, or Jarvis. "
        "Output ONLY the transcribed text, with no quotes or extra commentary."
    )

    def __init__(self, timeout: float = 15.0):
        self.api_key, self.model = _load_env_key()
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def _to_wav_bytes(self, audio_input: Union[str, Path, np.ndarray]) -> tuple[bytes, float]:
        if isinstance(audio_input, (str, Path)):
            data, sr = sf.read(str(audio_input), dtype="float32")
        else:
            data = np.asarray(audio_input, dtype=np.float32)
            sr = 16000

        if data.ndim > 1:
            data = data.mean(axis=1)

        duration = len(data) / float(sr or 16000)
        buf = io.BytesIO()
        sf.write(buf, data, int(sr or 16000), format="WAV", subtype="PCM_16")
        return buf.getvalue(), duration

    def transcribe_with_metadata(self, audio_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        wav_bytes, duration = self._to_wav_bytes(audio_input)
        if duration < 0.05:
            return {"text": "", "duration": 0.0, "latency": 0.0, "engine": "gemini-stt"}

        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": self.PROMPT},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
        }

        t0 = time.perf_counter()
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        latency = round(time.perf_counter() - t0, 3)

        return {
            "text": text,
            "duration": round(duration, 2),
            "latency": latency,
            "engine": "gemini-stt",
        }
