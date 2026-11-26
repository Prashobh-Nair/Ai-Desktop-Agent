"""
Local Offline Neural Speech-to-Text Engine
High-Capacity Transformer Architecture (Whisper-Base 74M / Small 244M)
Optimized with INT8 Quantization via CTranslate2 for Local CPU/GPU Inference.
"""
import io
import json
import os
import sys
import time
import math
import urllib.error
import urllib.request
import uuid
import warnings
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# MKL & OpenMP heap stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

try:
    import ctranslate2
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


# Default local models directory path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "stt-engine" / "models" / "acoustic-model"

# Model specifications
WHISPER_SPECS = {
    "tiny": {
        "parameters": "39 Million",
        "param_count": 39_000_000,
        "ram_mb": 150,
        "layers": 4,
        "hidden_dim": 384,
        "heads": 6
    },
    "base": {
        "parameters": "74 Million",
        "param_count": 74_000_000,
        "ram_mb": 220,
        "layers": 6,
        "hidden_dim": 512,
        "heads": 8
    },
    "small": {
        "parameters": "244 Million",
        "param_count": 244_000_000,
        "ram_mb": 580,
        "layers": 12,
        "hidden_dim": 768,
        "heads": 12
    },
    "small.en": {
        "parameters": "244 Million (English-Specialized)",
        "param_count": 244_000_000,
        "ram_mb": 580,
        "layers": 12,
        "hidden_dim": 768,
        "heads": 12
    },
    "medium": {
        "parameters": "769 Million",
        "param_count": 769_000_000,
        "ram_mb": 1500,
        "layers": 24,
        "hidden_dim": 1024,
        "heads": 16
    }
}


DEFAULT_PROMPT_BIAS = (
    "JARVIS desktop assistant commands and natural conversation. "
    "Names and places: Prashobh Nair, Sahaj Patel, Zeel Vasoya, Mayank, Navrachana University, Vadodara. "
    "Apps and tools: YouTube, Spotify, Chrome, VS Code, GitHub, Gmail, WhatsApp, Instagram, Gemini, Ollama, ChatGPT, Python, PowerShell."
)

DEFAULT_HOTWORDS = (
    "JARVIS, Prashobh Nair, Sahaj Patel, Zeel Vasoya, Mayank, "
    "Navrachana University, Vadodara, YouTube, Spotify, Chrome, "
    "VS Code, GitHub, Gmail, WhatsApp, Instagram, Gemini, Ollama, "
    "ChatGPT, Python, PowerShell, Believer, Kesariya"
)

HOTWORDS_FILE = PROJECT_ROOT / "hotwords.txt"


def build_hotwords(extra_terms: Optional[list[str]] = None, max_terms: int = 40) -> str:
    """Combines default proper nouns, optional hotwords.txt, and dynamic terms."""
    seen: set[str] = set()
    terms: list[str] = []

    def _add(raw: str) -> None:
        t = raw.strip()
        if t and t.lower() not in seen and len(terms) < max_terms:
            seen.add(t.lower())
            terms.append(t)

    if extra_terms:
        for item in extra_terms:
            _add(item)
    if HOTWORDS_FILE.exists():
        try:
            for line in HOTWORDS_FILE.read_text(encoding="utf-8").splitlines():
                for part in line.split(","):
                    _add(part)
        except OSError:
            pass
    for part in DEFAULT_HOTWORDS.split(","):
        _add(part)
    return ", ".join(terms)


class LocalWhisperEngine:
    """
    Production-grade, 100% Offline Local Speech-to-Text Engine.
    High-capacity neural acoustic model replacing toy 1.6M parameter scratch model.
    Utilizes process-level singleton caching to prevent MKL heap duplication.
    """
    _cached_models: Dict[str, Any] = {}

    def __init__(
        self,
        model_size: str = "base",
        model_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 2,
    ):
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper package not installed. Install via: pip install faster-whisper"
            )

        self.model_size = model_size.lower()
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads

        # Determine model path (use local acoustic-model folder only for default 'base')
        if model_path and os.path.exists(model_path):
            self.model_source = str(Path(model_path).resolve())
        elif self.model_size == "base" and DEFAULT_LOCAL_MODEL_PATH.exists() and (DEFAULT_LOCAL_MODEL_PATH / "model.bin").exists():
            self.model_source = str(DEFAULT_LOCAL_MODEL_PATH.resolve())
        else:
            self.model_source = self.model_size

        self.specs = WHISPER_SPECS.get(self.model_size, WHISPER_SPECS["base"])
        self.cache_key = f"{self.model_source}:{self.device}:{self.compute_type}:{self.cpu_threads}"

    def initialize(self):
        """Loads model into memory if not already initialized (singleton cache)."""
        if self.cache_key not in LocalWhisperEngine._cached_models:
            t0 = time.time()
            LocalWhisperEngine._cached_models[self.cache_key] = WhisperModel(
                self.model_source,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=1
            )
            self.load_time = time.time() - t0
        return LocalWhisperEngine._cached_models[self.cache_key]

    @property
    def model(self):
        return self.initialize()

    @staticmethod
    def load_audio(audio_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Loads and standardizes audio to 16,000 Hz Mono Float32 NumPy array.
        Bypasses pyav audio decoder to prevent container errors.
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            data, sr = sf.read(audio_path, dtype="float32")
        elif isinstance(audio_input, np.ndarray):
            data = audio_input.astype(np.float32)
            sr = 16000
        elif hasattr(audio_input, "detach"): # Torch Tensor
            data = audio_input.detach().cpu().numpy().astype(np.float32)
            sr = 16000
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Convert stereo or multi-channel to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16,000 Hz if necessary
        target_sr = 16000
        if sr != target_sr:
            gcd = math.gcd(int(sr), target_sr)
            up = target_sr // gcd
            down = int(sr) // gcd
            data = resample_poly(data, up, down).astype(np.float32)

        # Normalize audio peak
        max_val = np.abs(data).max() if len(data) > 0 else 0.0
        if max_val > 1.0:
            data = data / max_val

        return data

    def transcribe(self, audio_input: Union[str, Path, np.ndarray], language: str = "en") -> str:
        """
        Transcribes audio into text.
        Returns the decoded text string with capitalization & punctuation.
        """
        result = self.transcribe_with_metadata(audio_input, language=language)
        return result["text"]

    def transcribe_with_metadata(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = "en",
        initial_prompt: Optional[str] = DEFAULT_PROMPT_BIAS,
        beam_size: int = 5,
        temperature: float = 0.0,
        hotwords: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes audio and returns comprehensive transcription & performance metrics.
        """
        m = self.initialize()
        audio_data = self.load_audio(audio_input)
        duration = len(audio_data) / 16000.0

        if len(audio_data) == 0:
            return {
                "text": "",
                "language": language or "en",
                "confidence": 0.0,
                "duration": 0.0,
                "latency": 0.0,
                "rtf": 0.0,
                "model_parameters": self.specs["parameters"]
            }

        effective_hotwords = hotwords if hotwords is not None else build_hotwords()
        t0 = time.time()
        segments, info = m.transcribe(
            audio_data,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=effective_hotwords or None,
            beam_size=beam_size,
            patience=1.0,
            temperature=temperature,
            condition_on_previous_text=False,
        )

        collected_text = []
        for s in segments:
            text = s.text.strip()
            if text:
                collected_text.append(text)

        latency = time.time() - t0
        rtf = latency / max(0.01, duration)

        return {
            "text": " ".join(collected_text).strip(),
            "language": info.language.upper() if info.language else (language.upper() if language else "EN"),
            "confidence": round(float(info.language_probability) * 100, 1) if info.language_probability else 99.0,
            "duration": round(duration, 2),
            "latency": round(latency, 3),
            "rtf": round(rtf, 3),
            "model_name": f"Whisper-{self.model_size.capitalize()} (Local INT8)",
            "model_parameters": self.specs["parameters"],
            "parameter_count": self.specs["param_count"],
            "quantization": self.compute_type.upper(),
            "offline": True
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Returns detailed architectural specifications of this local engine."""
        return {
            "name": f"Local Neural Whisper ({self.model_size.capitalize()})",
            "model_type": "Sequence-to-Sequence Deep Transformer",
            "parameters": self.specs["parameters"],
            "parameter_count": self.specs["param_count"],
            "quantization": f"{self.compute_type.upper()} CPU Optimized (CTranslate2)",
            "ram_footprint": f"~{self.specs['ram_mb']} MB",
            "offline": True,
            "source_path": self.model_source,
        }


def _load_env_key(name: str) -> str:
    val = os.getenv(name, "").strip()
    if val:
        return val
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


class GroqWhisperClient:
    """Fast cloud STT via Groq Whisper large-v3-turbo (in-memory WAV, zero disk writes)."""

    ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3-turbo", timeout: float = 10.0):
        self.api_key = (api_key or _load_env_key("GROQ_API_KEY")).strip()
        self.model = model
        self.timeout = timeout
        try:
            import httpx
            self._http = httpx.Client(timeout=self.timeout)
        except Exception:
            self._http = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_groq_api_key_here")

    def transcribe_with_metadata(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = "en",
        initial_prompt: Optional[str] = DEFAULT_PROMPT_BIAS,
        hotwords: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        audio_data = LocalWhisperEngine.load_audio(audio_input)
        duration = len(audio_data) / 16000.0
        if len(audio_data) == 0:
            return {
                "text": "",
                "language": (language or "en").upper(),
                "confidence": 0.0,
                "duration": 0.0,
                "latency": 0.0,
                "rtf": 0.0,
                "model_name": f"Groq {self.model} (Cloud)",
                "offline": False,
            }

        buf = io.BytesIO()
        sf.write(buf, audio_data, 16000, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        prompt_parts = [p for p in (initial_prompt, f"Key terms: {hotwords}" if hotwords else None) if p]
        prompt_str = " ".join(prompt_parts)[:800]

        boundary = f"----JarvisBoundary{uuid.uuid4().hex}"
        parts: list[bytes] = []
        for field, value in (("model", self.model), ("language", language or "en"), ("prompt", prompt_str)):
            if not value:
                continue
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
            + wav_bytes
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
        body = b"".join(parts)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Jarvis-Desktop-Agent/1.0",
        }
        t0 = time.time()
        if self._http is not None:
            resp = self._http.post(self.ENDPOINT, content=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        else:
            req = urllib.request.Request(self.ENDPOINT, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        latency = time.time() - t0
        return {
            "text": (payload.get("text") or "").strip(),
            "language": (language or "en").upper(),
            "confidence": 99.0,
            "duration": round(duration, 2),
            "latency": round(latency, 3),
            "rtf": round(latency / max(0.01, duration), 3),
            "model_name": f"Groq {self.model} (Cloud)",
            "offline": False,
        }
