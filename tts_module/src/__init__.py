"""
TTS Module: Custom Local Text-to-Speech Engine powered by Kokoro-82M.
Part of the AI-Powered Autonomous Desktop Agent.
"""

from .tts_engine import (
    TTSEngine,
    get_tts_engine,
    generate_speech,
    speak,
    stop_speech,
    get_available_voices
)
from .audio_player import (
    AudioPlayer,
    get_audio_player,
    play_audio
)
from .text_processor import TextProcessor
from .config import TTSConfig, DEFAULT_CONFIG, AVAILABLE_VOICES

__version__ = "1.0.0"

__all__ = [
    # Primary interface functions (Section 8)
    "generate_speech",
    "play_audio",
    "speak",
    "stop_speech",
    "get_available_voices",
    # Core Classes
    "TTSEngine",
    "AudioPlayer",
    "TextProcessor",
    "TTSConfig",
    "DEFAULT_CONFIG",
    "AVAILABLE_VOICES",
    "get_tts_engine",
    "get_audio_player"
]

