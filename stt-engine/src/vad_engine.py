"""
Voice Activity Detection (VAD) Engine.
Integrates Silero VAD parameters and energy-based speech segmentation.
"""
from typing import List, Tuple, Dict
import numpy as np
from config import settings
from src.logger import log_info


class VADEngine:
    """
    Voice Activity Detection and Speech Segmentation Engine.
    Filters ambient silence and segments continuous speech for Whisper.
    """
    def __init__(
        self,
        sample_rate: int = settings.SAMPLE_RATE,
        vad_threshold: float = settings.VAD_THRESHOLD,
        min_silence_duration_ms: int = settings.MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = settings.SPEECH_PAD_MS,
    ):
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

    def get_vad_parameters(self) -> Dict:
        """
        Returns parameters dictionary configured for Faster-Whisper's integrated Silero VAD.
        """
        return {
            "threshold": self.vad_threshold,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }

    def compute_energy(self, audio: np.ndarray, frame_duration_ms: int = 30) -> np.ndarray:
        """Computes Root-Mean-Square (RMS) energy across sliding frames."""
        frame_size = int(self.sample_rate * (frame_duration_ms / 1000.0))
        num_frames = max(1, len(audio) // frame_size)
        energies = np.zeros(num_frames, dtype=np.float32)
        
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            energies[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
            
        return energies

    def has_speech(self, audio: np.ndarray, threshold: float = 0.015) -> bool:
        """
        Quick check to determine if audio buffer contains human speech or pure silence.
        """
        if len(audio) == 0:
            return False
            
        energies = self.compute_energy(audio)
        speech_frames = np.sum(energies > threshold)
        speech_ratio = speech_frames / len(energies)
        return speech_ratio > 0.08

    def segment_speech(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Energy-based speech segmentation for fast offline boundary analysis.
        Returns list of (start_seconds, end_seconds).
        """
        energies = self.compute_energy(audio, frame_duration_ms=30)
        threshold = np.mean(energies) * 0.75 + 0.005
        frame_sec = 0.03

        is_speech = energies > threshold
        segments = []
        in_segment = False
        start_idx = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_segment:
                in_segment = True
                start_idx = i
            elif not speech and in_segment:
                in_segment = False
                end_idx = i
                if (end_idx - start_idx) * frame_sec >= 0.2:  # at least 200ms
                    segments.append((round(start_idx * frame_sec, 2), round(end_idx * frame_sec, 2)))

        if in_segment:
            segments.append((round(start_idx * frame_sec, 2), round(len(energies) * frame_sec, 2)))

        return segments

