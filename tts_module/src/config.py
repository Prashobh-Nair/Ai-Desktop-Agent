"""
Configuration module for the Local Kokoro-82M TTS Engine.
Contains model paths, audio settings, voice registries, and hardware optimizations.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Root directory of the TTS module
TTS_ROOT_DIR = Path(__file__).resolve().parent.parent

# Subdirectories
MODELS_DIR = TTS_ROOT_DIR / "models" / "kokoro"
AUDIO_OUTPUT_DIR = TTS_ROOT_DIR / "audio" / "output"

# Model filenames
ONNX_MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"


# Voice registry with human-readable metadata
# Voices available in Kokoro-82M v1.0
AVAILABLE_VOICES: Dict[str, Dict[str, str]] = {
    # American English - Female
    "af_heart": {
        "name": "Heart",
        "gender": "Female",
        "accent": "American",
        "description": "Warm, natural, highly expressive (Flagship assistant voice)",
        "language_code": "en-us"
    },
    "af_bella": {
        "name": "Bella",
        "gender": "Female",
        "accent": "American",
        "description": "Cheerful, energetic, friendly tone",
        "language_code": "en-us"
    },
    "af_sarah": {
        "name": "Sarah",
        "gender": "Female",
        "accent": "American",
        "description": "Clear, professional, articulate narrator",
        "language_code": "en-us"
    },
    "af_nicole": {
        "name": "Nicole",
        "gender": "Female",
        "accent": "American",
        "description": "Calm, gentle, soothing presence",
        "language_code": "en-us"
    },
    "af_sky": {
        "name": "Sky",
        "gender": "Female",
        "accent": "American",
        "description": "Bright, modern, engaging",
        "language_code": "en-us"
    },
    "af_alloy": {
        "name": "Alloy",
        "gender": "Female",
        "accent": "American",
        "description": "Neutral, versatile, balanced",
        "language_code": "en-us"
    },
    "af_jessica": {
        "name": "Jessica",
        "gender": "Female",
        "accent": "American",
        "description": "Direct, conversational, crisp",
        "language_code": "en-us"
    },
    # American English - Male
    "am_adam": {
        "name": "Adam",
        "gender": "Male",
        "accent": "American",
        "description": "Deep, confident, natural male assistant",
        "language_code": "en-us"
    },
    "am_michael": {
        "name": "Michael",
        "gender": "Male",
        "accent": "American",
        "description": "Authoritative, clear, corporate/assistant style",
        "language_code": "en-us"
    },
    "am_echo": {
        "name": "Echo",
        "gender": "Male",
        "accent": "American",
        "description": "Calm, steady, analytical tone",
        "language_code": "en-us"
    },
    "am_eric": {
        "name": "Eric",
        "gender": "Male",
        "accent": "American",
        "description": "Warm, approachable, friendly",
        "language_code": "en-us"
    },
    "am_liam": {
        "name": "Liam",
        "gender": "Male",
        "accent": "American",
        "description": "Young, energetic, casual",
        "language_code": "en-us"
    },
    "am_onyx": {
        "name": "Onyx",
        "gender": "Male",
        "accent": "American",
        "description": "Rich, resonant, deep bass",
        "language_code": "en-us"
    },
    # British English - Female
    "bf_emma": {
        "name": "Emma",
        "gender": "Female",
        "accent": "British",
        "description": "Sophisticated, articulate, classic British RP",
        "language_code": "en-gb"
    },
    "bf_isabella": {
        "name": "Isabella",
        "gender": "Female",
        "accent": "British",
        "description": "Formal, refined, polite",
        "language_code": "en-gb"
    },
    "bf_alice": {
        "name": "Alice",
        "gender": "Female",
        "accent": "British",
        "description": "Gentle, pleasant, clear diction",
        "language_code": "en-gb"
    },
    "bf_lily": {
        "name": "Lily",
        "gender": "Female",
        "accent": "British",
        "description": "Warm British conversationalist",
        "language_code": "en-gb"
    },
    # British English - Male
    "bm_george": {
        "name": "George",
        "gender": "Male",
        "accent": "British",
        "description": "Classic English gentleman, polished and warm",
        "language_code": "en-gb"
    },
    "bm_daniel": {
        "name": "Daniel",
        "gender": "Male",
        "accent": "British",
        "description": "Authoritative British male narrator",
        "language_code": "en-gb"
    },
    "bm_lewis": {
        "name": "Lewis",
        "gender": "Male",
        "accent": "British",
        "description": "Contemporary British, dynamic and engaging",
        "language_code": "en-gb"
    },
    "bm_fable": {
        "name": "Fable",
        "gender": "Male",
        "accent": "British",
        "description": "Storyteller, expressive British accent",
        "language_code": "en-gb"
    }
}


@dataclass
class TTSConfig:
    """
    Configuration parameters for the Kokoro TTS engine and audio pipeline.
    """
    # Model configuration
    model_path: Path = field(default_factory=lambda: MODELS_DIR / ONNX_MODEL_FILENAME)
    voices_path: Path = field(default_factory=lambda: MODELS_DIR / VOICES_FILENAME)
    output_dir: Path = field(default_factory=lambda: AUDIO_OUTPUT_DIR)
    
    # Audio parameters
    sample_rate: int = 24000  # Native Kokoro sample rate (24 kHz)
    default_voice: str = "af_heart"  # Default flagship assistant voice
    default_speed: float = 1.0  # Normal speed (range: 0.5 - 2.0)
    default_language: str = "en-us"
    
    # Inter-sentence pause (silence inserted between sentences in seconds)
    inter_sentence_pause_sec: float = 0.20  # 200 ms natural conversational pause
    
    # Text segmentation limits for long LLM responses
    max_chunk_chars: int = 250  # Maximum characters per chunk before splitting
    max_chunk_words: int = 40   # Maximum words per chunk
    
    # Hardware & Performance Optimizations (tuned for 8GB RAM Windows Laptop)
    onnx_intra_threads: int = 4  # Intra-op thread pool for multi-core CPU
    onnx_inter_threads: int = 1  # Inter-op thread pool
    enable_memory_pattern: bool = True
    enable_cpu_memory_arena: bool = True
    
    # Audio playback defaults
    playback_backend: str = "sounddevice"  # "sounddevice" with "winsound" fallback
    default_volume: float = 1.0  # Range 0.0 to 1.0
    
    # Audio cleanup / retention
    cleanup_older_than_days: int = 7
    
    def validate(self) -> None:
        """Validates that configuration parameters are within acceptable bounds."""
        if self.default_speed < 0.2 or self.default_speed > 3.0:
            raise ValueError(f"Speed must be between 0.2 and 3.0, got {self.default_speed}")
        if self.sample_rate not in (16000, 22050, 24000, 44100, 48000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.default_voice not in AVAILABLE_VOICES:
            print(f"[!] Warning: Voice '{self.default_voice}' not in recognized registry. Available: {list(AVAILABLE_VOICES.keys())[:5]}...")


# Global default configuration instance
DEFAULT_CONFIG = TTSConfig()

