"""
Audio Preprocessing and Format Normalization Module.
Standardizes audio files and arrays into 16kHz mono float32 for Whisper Small.
"""
import os
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from config import settings
from src.logger import log_info, log_error


class AudioProcessor:
    """
    Standardizes diverse audio formats and waveforms into 16kHz mono float32.
    """
    TARGET_SR = settings.SAMPLE_RATE

    @classmethod
    def load_audio(cls, audio_source) -> np.ndarray:
        """
        Loads audio from a file path or array and normalizes to 16kHz mono float32.
        Supports WAV, MP3, FLAC, OGG, etc.
        """
        if isinstance(audio_source, (str, os.PathLike)):
            audio_path = str(audio_source)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            try:
                # soundfile supports WAV, FLAC, OGG, and modern formats
                data, sr = sf.read(audio_path, dtype="float32")
            except Exception:
                # Fallback to PyAV if soundfile fails on complex containers (e.g. m4a/mp3)
                import av
                container = av.open(audio_path)
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=cls.TARGET_SR)
                frames = []
                for frame in container.decode(audio=0):
                    resampled = resampler.resample(frame)
                    frames.append(resampled.to_ndarray())
                if not frames:
                    raise ValueError(f"No audio frames decoded from: {audio_path}")
                data = np.concatenate(frames, axis=1).squeeze()
                sr = cls.TARGET_SR

        elif isinstance(audio_source, np.ndarray):
            data = audio_source.astype(np.float32)
            sr = cls.TARGET_SR
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_source)}")

        # Convert stereo or multichannel to mono
        if data.ndim > 1:
            data = data.mean(axis=1 if data.shape[0] > data.shape[1] else 0)

        # Resample to 16,000 Hz if sample rate differs
        if sr != cls.TARGET_SR:
            gcd = math.gcd(int(sr), cls.TARGET_SR)
            up = cls.TARGET_SR // gcd
            down = int(sr) // gcd
            data = resample_poly(data, up, down).astype(np.float32)

        # Normalize amplitude to prevent clipping and enhance faint speech
        data = cls.normalize_amplitude(data)
        return data

    @classmethod
    def normalize_amplitude(cls, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Scales audio peak to target_peak if not silent."""
        max_val = np.abs(audio).max()
        if max_val > 1e-4:
            return (audio / max_val * target_peak).astype(np.float32)
        return audio.astype(np.float32)

    @classmethod
    def get_duration_seconds(cls, audio: np.ndarray) -> float:
        """Returns audio duration in seconds."""
        return float(len(audio) / cls.TARGET_SR)

