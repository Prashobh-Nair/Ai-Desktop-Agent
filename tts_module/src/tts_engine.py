"""
Kokoro-82M Local Neural Text-to-Speech Engine.
Provides high-performance, natural, offline speech synthesis for the AI Desktop Agent.

Key Features:
- 100% Local offline inference via optimized ONNX Runtime.
- Support for 50+ expressive voices (American, British, etc.).
- Configurable speaking speed (0.5x to 2.0x).
- Intelligent text preprocessing (numbers, dates, currency, markdown, acronyms).
- Long LLM response chunking with natural pause insertion.
- Audio caching for instant response on repeated commands.
- Standalone integration functions (generate_speech, play_audio, speak).
- Optimized for 8GB RAM laptops (lightweight ~250MB footprint).
"""

import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator, Union

import numpy as np
import soundfile as sf
import kokoro_onnx

from .config import TTSConfig, DEFAULT_CONFIG, AVAILABLE_VOICES, AUDIO_OUTPUT_DIR
from .text_processor import TextProcessor
from .audio_player import AudioPlayer, get_audio_player, play_audio

# Configure module logger
logger = logging.getLogger("TTS_Engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [KokoroTTS] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TTSEngine:
    """
    Core Text-to-Speech synthesis engine using the Kokoro-82M neural model.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        
        self.text_processor = TextProcessor(
            max_chunk_chars=self.config.max_chunk_chars,
            max_chunk_words=self.config.max_chunk_words
        )
        self.audio_player = get_audio_player()
        
        self._kokoro: Optional[kokoro_onnx.Kokoro] = None
        self._is_initialized = False
        self._cached_audio: Dict[str, str] = {}  # Cache key -> file path
        
        # Telemetry / metrics from the most recent generation
        self.last_metrics: Dict[str, Union[float, int, str]] = {}

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """
        Loads the Kokoro-82M ONNX model weights and voice configuration.
        Caches the model in memory so subsequent generations are immediate.
        """
        if self._is_initialized and self._kokoro is not None:
            return

        model_path = str(self.config.model_path.resolve())
        voices_path = str(self.config.voices_path.resolve())

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Kokoro-82M model file not found at: {model_path}\n"
                f"Please run 'python tts_module/download_model.py' to download the model."
            )

        if not os.path.exists(voices_path):
            raise FileNotFoundError(
                f"Kokoro-82M voices file not found at: {voices_path}\n"
                f"Please run 'python tts_module/download_model.py' to download the voices."
            )

        logger.info(f"Loading Kokoro-82M model from: {model_path} ...")
        t0 = time.time()
        
        try:
            self._kokoro = kokoro_onnx.Kokoro(
                model_path=model_path,
                voices_path=voices_path
            )
            self._is_initialized = True
            load_time = time.time() - t0
            logger.info(f"Kokoro-82M loaded successfully in {load_time:.2f}s! Ready for synthesis.")
        except Exception as e:
            logger.error(f"Failed to load Kokoro-82M ONNX model: {e}")
            raise RuntimeError(f"Kokoro-82M initialization error: {e}") from e

    def get_available_voices(self) -> List[str]:
        """Returns the list of all available voice IDs."""
        self.initialize()
        return self._kokoro.get_voices()

    def get_voice_info(self, voice_name: str) -> Dict[str, str]:
        """Returns metadata for a given voice ID."""
        if voice_name in AVAILABLE_VOICES:
            return AVAILABLE_VOICES[voice_name]
        return {
            "name": voice_name,
            "gender": "Unknown",
            "accent": "English",
            "description": f"Kokoro-82M voice style: {voice_name}",
            "language_code": "en-us"
        }

    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generates a unique deterministic cache key for text + voice + speed."""
        content = f"{text}|{voice}|{speed:.2f}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        output_filename: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Core synthesis function.
        Converts input text (from LLM) into speech audio, saves as WAV, and returns the path.

        Args:
            text: Text to synthesize (supports raw LLM outputs with markdown, numbers, etc.).
            voice: Optional voice ID (e.g. 'af_heart', 'am_adam', 'bf_emma').
                   Defaults to config.default_voice.
            speed: Optional speaking speed multiplier (0.5 to 2.0). Defaults to 1.0.
            output_filename: Optional custom output WAV filename.
            use_cache: If True, uses cached audio for identical requests.

        Returns:
            Absolute file path to the generated WAV audio file.
        """
        # Step 1: Input validation
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text passed to generate_speech.")
            return self._generate_silence_wav(duration_sec=0.2, output_filename=output_filename)

        self.initialize()

        # Step 2: Parameter resolution
        target_voice = voice or self.config.default_voice
        target_speed = float(speed if speed is not None else self.config.default_speed)
        target_speed = max(0.5, min(2.0, target_speed))
        
        # Determine language code based on voice prefix or registry
        voice_info = self.get_voice_info(target_voice)
        lang_code = voice_info.get("language_code", self.config.default_language)

        # Step 3: Check cache for fast retrieval
        normalized_full_text = self.text_processor.normalize(text)
        if not normalized_full_text:
            return self._generate_silence_wav(duration_sec=0.2, output_filename=output_filename)

        cache_key = self._get_cache_key(normalized_full_text, target_voice, target_speed)
        if use_cache and cache_key in self._cached_audio:
            cached_path = self._cached_audio[cache_key]
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                logger.info(f"Audio cache hit for key [{cache_key[:8]}] - 0ms synthesis time!")
                self.last_metrics = {
                    "generation_time": 0.001,
                    "audio_duration": self._get_audio_duration(cached_path),
                    "rtf": 0.0,
                    "voice": target_voice,
                    "speed": target_speed,
                    "cached": True
                }
                return cached_path

        # Step 4: Text chunking for long LLM responses
        chunks = self.text_processor.chunk_text(text)
        if not chunks:
            chunks = [normalized_full_text]

        logger.info(f"Synthesizing {len(chunks)} chunk(s) using voice '{target_voice}' (speed={target_speed:.2f})...")
        t_start = time.time()

        audio_segments: List[np.ndarray] = []
        sample_rate = self.config.sample_rate

        # Generate silence buffer to insert between chunks
        pause_samples = int(sample_rate * self.config.inter_sentence_pause_sec)
        silence_buffer = np.zeros(pause_samples, dtype=np.float32)

        for i, chunk in enumerate(chunks):
            try:
                # Synthesize chunk with Kokoro-82M
                samples, sr = self._kokoro.create(
                    text=chunk,
                    voice=target_voice,
                    speed=target_speed,
                    lang=lang_code,
                    sentence_pause=self.config.inter_sentence_pause_sec
                )
                sample_rate = sr
                audio_segments.append(samples)
                
                # Add natural pause between chunks if more chunks follow
                if i < len(chunks) - 1:
                    audio_segments.append(silence_buffer)
            except Exception as chunk_err:
                logger.error(f"Error synthesizing chunk {i+1}/{len(chunks)} ('{chunk[:30]}...'): {chunk_err}")
                continue

        if not audio_segments:
            logger.error("Failed to generate any audio segments from text.")
            return self._generate_silence_wav(duration_sec=0.2, output_filename=output_filename)

        # Concatenate all audio segments into a single waveform
        full_audio = np.concatenate(audio_segments)
        gen_time = time.time() - t_start
        total_duration = len(full_audio) / sample_rate
        rtf = gen_time / max(total_duration, 0.001)

        # Step 5: Save audio to WAV file
        if output_filename:
            if not output_filename.endswith(".wav"):
                output_filename += ".wav"
            out_path = self.config.output_dir / output_filename
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            short_hash = cache_key[:8]
            out_path = self.config.output_dir / f"tts_{timestamp}_{short_hash}.wav"

        sf.write(str(out_path), full_audio, sample_rate)
        saved_path_str = str(out_path.resolve())

        # Update cache
        if use_cache:
            self._cached_audio[cache_key] = saved_path_str

        # Record telemetry metrics
        self.last_metrics = {
            "generation_time": round(gen_time, 3),
            "audio_duration": round(total_duration, 2),
            "rtf": round(rtf, 3),
            "sample_rate": sample_rate,
            "characters": len(text),
            "words": len(text.split()),
            "voice": target_voice,
            "speed": target_speed,
            "cached": False,
            "output_path": saved_path_str
        }

        logger.info(
            f"Speech generated successfully! Duration: {total_duration:.2f}s | "
            f"Time: {gen_time:.2f}s | RTF: {rtf:.2f} | File: {out_path.name}"
        )
        return saved_path_str

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        blocking: bool = True
    ) -> str:
        """
        High-level all-in-one assistant method:
        Generates speech and immediately plays it through the speakers.

        Args:
            text: Text to speak.
            voice: Voice ID.
            speed: Speed multiplier.
            blocking: If True, waits until speech playback finishes before returning.

        Returns:
            Path to the generated WAV audio file.
        """
        wav_path = self.generate_speech(text=text, voice=voice, speed=speed)
        self.audio_player.play(wav_path, blocking=blocking)
        return wav_path

    def stop(self) -> None:
        """Stops current audio playback immediately."""
        self.audio_player.stop()

    def _generate_silence_wav(self, duration_sec: float = 0.2, output_filename: Optional[str] = None) -> str:
        """Generates a small silent WAV file for empty or invalid inputs."""
        sr = self.config.sample_rate
        silence = np.zeros(int(sr * duration_sec), dtype=np.float32)
        if output_filename:
            if not output_filename.endswith(".wav"):
                output_filename += ".wav"
            out_path = self.config.output_dir / output_filename
        else:
            out_path = self.config.output_dir / "silence.wav"
            
        sf.write(str(out_path), silence, sr)
        return str(out_path.resolve())

    def _get_audio_duration(self, audio_path: str) -> float:
        """Calculates audio duration in seconds from a WAV file."""
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Global Singleton Engine & Convenience Functions for Section 8 Integration
# ---------------------------------------------------------------------------

_global_tts_engine: Optional[TTSEngine] = None

def get_tts_engine() -> TTSEngine:
    """Returns the singleton instance of TTSEngine."""
    global _global_tts_engine
    if _global_tts_engine is None:
        _global_tts_engine = TTSEngine()
        _global_tts_engine.initialize()
    return _global_tts_engine


def generate_speech(text: str) -> str:
    """
    REQUIRED PROJECT INTERFACE FUNCTION (Section 8):
    Accepts text generated by the LLM.
    Converts the text into speech using Kokoro-82M locally.
    Saves the generated audio locally.
    Returns the audio file path.
    """
    engine = get_tts_engine()
    return engine.generate_speech(text)


def speak(text: str, blocking: bool = True) -> str:
    """Convenience function: Generates speech and plays it through system speakers."""
    engine = get_tts_engine()
    return engine.speak(text, blocking=blocking)


def stop_speech() -> None:
    """Stops any active speech playback."""
    engine = get_tts_engine()
    engine.stop()


def get_available_voices() -> List[str]:
    """Returns list of available Kokoro voice styles."""
    engine = get_tts_engine()
    return engine.get_available_voices()

