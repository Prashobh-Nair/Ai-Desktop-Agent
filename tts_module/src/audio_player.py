"""
Audio Player Module for Windows Desktop Agent.
Plays generated speech audio through system speakers with:
- Blocking and non-blocking (background) playback.
- Instant interruption / stop capability.
- Volume control.
- Primary backend: sounddevice (PortAudio).
- Native fallback: Windows winsound (zero external dependency).
- Thread-safe state tracking.
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Union

import soundfile as sf
import numpy as np

# Configure logger
logger = logging.getLogger("TTS_AudioPlayer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [AudioPlayer] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AudioPlayer:
    """
    Thread-safe audio player for WAV speech output on Windows.
    Supports asynchronous playback, volume adjustment, pause/stop controls,
    and automatic fallback to winsound if sounddevice fails.
    """

    def __init__(self, default_volume: float = 1.0):
        self.default_volume = max(0.0, min(1.0, default_volume))
        self._current_volume = self.default_volume
        self._is_playing = False
        self._stop_requested = False
        self._playback_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Check sounddevice availability
        self.has_sounddevice = False
        try:
            import sounddevice as sd
            self._sd = sd
            self.has_sounddevice = True
        except Exception as e:
            logger.warning(f"sounddevice not available ({e}). Falling back to winsound.")
            self._sd = None

        # Check winsound availability (built-in on Windows)
        self.has_winsound = False
        try:
            import winsound
            self._winsound = winsound
            self.has_winsound = True
        except ImportError:
            self._winsound = None

    @property
    def is_playing(self) -> bool:
        """Returns True if audio is currently playing."""
        with self._lock:
            return self._is_playing

    def set_volume(self, volume: float) -> None:
        """Sets playback volume level between 0.0 (mute) and 1.0 (max)."""
        self._current_volume = max(0.0, min(1.0, float(volume)))
        logger.debug(f"Volume set to {self._current_volume:.2f}")

    def stop(self) -> None:
        """
        Immediately interrupts and stops current audio playback.
        Useful when the user starts speaking or gives a new command.
        """
        with self._lock:
            self._stop_requested = True
            
        if self.has_sounddevice and self._sd:
            try:
                self._sd.stop()
            except Exception as e:
                logger.debug(f"Error stopping sounddevice: {e}")

        if self.has_winsound and self._winsound:
            try:
                # SND_PURGE stops all instances of playing sounds
                self._winsound.PlaySound(None, self._winsound.SND_PURGE)
            except Exception:
                pass

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5)

        with self._lock:
            self._is_playing = False
            self._stop_requested = False
            
        logger.debug("Playback stopped.")

    def play_numpy(self, audio_data: np.ndarray, sample_rate: int = 24000, blocking: bool = False, volume: Optional[float] = None) -> bool:
        """
        Plays raw NumPy audio samples directly through speakers without saving to disk first.
        Useful for streaming sentence-by-sentence synthesis.
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio buffer passed to play_numpy.")
            return False

        vol = self._current_volume if volume is None else max(0.0, min(1.0, float(volume)))
        scaled_audio = audio_data * vol

        if not self.has_sounddevice or not self._sd:
            logger.warning("sounddevice is required for in-memory NumPy playback.")
            return False

        self.stop()

        def _play_worker():
            with self._lock:
                self._is_playing = True
                self._stop_requested = False
            try:
                self._sd.play(scaled_audio, samplerate=sample_rate)
                self._sd.wait()
            except Exception as e:
                logger.error(f"Error during sounddevice playback: {e}")
            finally:
                with self._lock:
                    self._is_playing = False

        if blocking:
            _play_worker()
            return True
        else:
            self._playback_thread = threading.Thread(target=_play_worker, daemon=True)
            self._playback_thread.start()
            return True

    def play(self, audio_path: Union[str, Path], blocking: bool = False, volume: Optional[float] = None) -> bool:
        """
        Plays a WAV audio file through Windows system speakers.
        
        Args:
            audio_path: Absolute or relative path to the WAV file.
            blocking: If True, waits for playback to finish before returning.
                      If False, returns immediately and plays in background.
            volume: Optional volume override (0.0 to 1.0).
            
        Returns:
            True if playback started/finished successfully, False otherwise.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return False

        if path.stat().st_size == 0:
            logger.error(f"Audio file is empty (0 bytes): {path}")
            return False

        # Stop any ongoing playback before starting new sound
        self.stop()

        vol = self._current_volume if volume is None else max(0.0, min(1.0, float(volume)))

        def _playback_task():
            with self._lock:
                self._is_playing = True
                self._stop_requested = False
                
            success = False
            # Method 1: sounddevice (supports volume scaling and clean interruption)
            if self.has_sounddevice and self._sd:
                try:
                    data, samplerate = sf.read(str(path), dtype="float32")
                    if vol < 1.0:
                        data = data * vol
                    self._sd.play(data, samplerate=samplerate)
                    self._sd.wait()
                    success = True
                except Exception as sd_err:
                    logger.warning(f"sounddevice playback failed ({sd_err}), attempting winsound fallback...")
                    success = False

            # Method 2: Fallback to native Windows winsound
            if not success and self.has_winsound and self._winsound:
                try:
                    flags = self._winsound.SND_FILENAME | self._winsound.SND_NODEFAULT
                    self._winsound.PlaySound(str(path), flags)
                    success = True
                except Exception as win_err:
                    logger.error(f"winsound fallback also failed: {win_err}")
                    success = False

            with self._lock:
                self._is_playing = False
            return success

        if blocking:
            return _playback_task()
        else:
            self._playback_thread = threading.Thread(target=_playback_task, daemon=True)
            self._playback_thread.start()
            return True

    def wait_until_done(self, poll_interval: float = 0.05) -> None:
        """Blocks execution until the current audio finishes playing."""
        while self.is_playing:
            time.sleep(poll_interval)


# Global default audio player instance
_default_player: Optional[AudioPlayer] = None

def get_audio_player() -> AudioPlayer:
    """Returns the singleton AudioPlayer instance."""
    global _default_player
    if _default_player is None:
        _default_player = AudioPlayer()
    return _default_player


def play_audio(audio_path: str, blocking: bool = True) -> None:
    """
    Standard interface function required by Project Architecture (Section 8).
    Accepts an audio file path and plays it through system speakers.
    """
    player = get_audio_player()
    player.play(audio_path, blocking=blocking)

