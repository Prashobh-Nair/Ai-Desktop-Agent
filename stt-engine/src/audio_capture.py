"""
Microphone Audio Capture Module using SoundDevice.
Provides interactive recording, timed recording, and device enumeration.
"""
import sys
import time
import threading
from typing import List, Dict, Optional
import numpy as np
import sounddevice as sd
from config import settings
from src.logger import log_info, log_error, log_warning


class AudioCapture:
    """
    Microphone audio capture manager.
    """
    def __init__(self, sample_rate: int = settings.SAMPLE_RATE, channels: int = settings.CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels

    @staticmethod
    def get_input_devices() -> List[Dict]:
        """Lists all audio devices that support input recording."""
        devices = sd.query_devices()
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append({
                    "index": idx,
                    "name": dev.get("name", f"Device {idx}"),
                    "samplerate": dev.get("default_samplerate", 44100),
                    "channels": dev.get("max_input_channels", 1),
                })
        return input_devices

    def print_devices(self):
        """Prints formatted table of available microphone devices."""
        devices = self.get_input_devices()
        print("\n" + "=" * 60)
        print(f"{'Index':<6} | {'Name':<35} | {'Max In':<8} | {'Default SR'}")
        print("=" * 60)
        for d in devices:
            print(f"{d['index']:<6} | {d['name'][:35]:<35} | {d['channels']:<8} | {int(d['samplerate'])} Hz")
        print("=" * 60 + "\n")

    def record_duration(self, duration: float = 5.0, device: Optional[int] = None) -> np.ndarray:
        """
        Records microphone audio for a fixed duration.
        """
        log_info(f"Recording for {duration:.1f} seconds on device {device if device is not None else 'Default'}...")
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device
            )
            sd.wait()
            log_info("Recording finished.")
            return recording.squeeze()
        except Exception as e:
            log_error(f"Microphone recording error: {e}")
            raise

    def record_interactive(self, device: Optional[int] = None) -> np.ndarray:
        """
        Records interactively: press Enter to start, press Enter to stop.
        """
        print("\n[*] Interactive Microphone Recording:")
        print("    --> Press ENTER to START recording...")
        try:
            input()
        except EOFError:
            pass

        print("[*] RECORDING STARTED! Speak now. (Press ENTER to STOP)...")
        frames = []
        is_recording = True

        def callback(indata, frame_count, time_info, status):
            if status:
                log_warning(f"Audio buffer status: {status}")
            if is_recording:
                frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                callback=callback
            ):
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    pass
        except Exception as e:
            log_error(f"Failed to open microphone stream: {e}")
            raise
        finally:
            is_recording = False

        print("[*] RECORDING STOPPED.")
        if not frames:
            log_warning("No audio frames recorded.")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).squeeze()
        log_info(f"Captured {len(audio)/self.sample_rate:.2f}s of audio ({len(audio)} samples).")
        return audio

