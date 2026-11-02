"""
Local Kokoro-82M Text-to-Speech Engine package root.
Allows importing directly as:
    from tts_module import generate_speech, play_audio, speak
"""

from .src import (
    generate_speech,
    play_audio,
    speak,
    stop_speech,
    get_available_voices,
    TTSEngine,
    AudioPlayer,
    TextProcessor,
    TTSConfig,
    DEFAULT_CONFIG,
    AVAILABLE_VOICES,
    get_tts_engine,
    get_audio_player
)

__all__ = [
    "generate_speech",
    "play_audio",
    "speak",
    "stop_speech",
    "get_available_voices",
    "TTSEngine",
    "AudioPlayer",
    "TextProcessor",
    "TTSConfig",
    "DEFAULT_CONFIG",
    "AVAILABLE_VOICES",
    "get_tts_engine",
    "get_audio_player"
]

