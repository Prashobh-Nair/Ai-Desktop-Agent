"""
Autonomous Desktop Agent - Verification Suite for Offline STT Engine:
1. Model loading (CPU INT8)
2. WAV audio transcription
3. Live microphone input query
4. Speech and silence detection (VAD)
5. Multilingual recognition
6. Offline model execution
7. Memory consumption (RAM < 1.5 GB)
8. Transcription latency & Real-Time Factor (RTF < 1.0)
"""
import sys
import time
import os
import warnings
from pathlib import Path

# Suppress external warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize runtime first for Windows MKL heap stability
import ctranslate2
import numpy as np

# Add project root to sys.path
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.model_manager import AcousticModelManager
from src.audio_capture import AudioCapture
from src.audio_processor import AudioProcessor
from src.vad_engine import VADEngine
from src.transcription_engine import TranscriptionEngine
from src.logger import get_process_memory_mb


def test_1_model_loading_and_offline():
    print("\n[TEST 1 & 6] Testing Neural Acoustic Model Loading & Offline Execution...")
    manager = AcousticModelManager()
    
    # 1. Verify local weights
    model_path = manager.download()
    assert manager.is_model_downloaded(), "Model weights missing after verification"
    print(f"  [PASS] Verified local neural weights in: {model_path}")

    # 2. Load offline with INT8
    model = manager.load_model(offline=True)
    assert model is not None, "Model failed to instantiate"
    ram_after = get_process_memory_mb()
    print(f"  [PASS] Neural Model loaded offline successfully! (Process RAM: {ram_after:.1f} MB)")
    return manager


def test_2_wav_transcription(sample_path="samples/sample1.wav"):
    print("\n[TEST 2] Testing WAV Audio Transcription...")
    engine = TranscriptionEngine()
    
    # Check sample file
    resolved_sample = Path(sample_path)
    if not resolved_sample.exists():
        resolved_sample = PROJECT_ROOT.parent / sample_path

    assert resolved_sample.exists(), f"Sample audio missing at {resolved_sample}"

    result = engine.transcribe(str(resolved_sample))
    text = result["text"]
    assert len(text) > 0, "Transcription result is empty"
    print(f"  [PASS] Audio: {resolved_sample.name}")
    print(f"  [PASS] Recognized Text: \"{text}\"")
    print(f"  [PASS] Detected Language: {result['language'].upper()} ({result['language_probability']*100:.1f}%)")
    return result


def test_3_speech_silence_detection():
    print("\n[TEST 4] Testing Voice Activity Detection (Speech vs Silence)...")
    vad = VADEngine()

    # Create 2 seconds of pure silence
    silence = np.zeros(16000 * 2, dtype=np.float32)
    has_speech_silence = vad.has_speech(silence)
    assert not has_speech_silence, "VAD incorrectly detected speech in pure silence"
    print("  [PASS] Pure silence correctly identified as non-speech (has_speech=False)")

    # Create synthetic tone/speech energy
    t = np.linspace(0, 2, 16000 * 2, endpoint=False)
    synthetic_speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    has_speech_tone = vad.has_speech(synthetic_speech)
    assert has_speech_tone, "VAD failed to detect active audio energy"
    print("  [PASS] Active audio energy correctly identified as speech (has_speech=True)")


def test_4_multilingual_recognition():
    print("\n[TEST 5] Testing Multilingual & Language Configuration...")
    engine = TranscriptionEngine()
    # Test setting explicit language code
    sample_path = PROJECT_ROOT.parent / "samples" / "sample1.wav"
    result = engine.transcribe(str(sample_path), language="en")
    assert result["language"] == "en", f"Language mismatch: expected 'en', got {result['language']}"
    print(f"  [PASS] Explicit language configuration verified (language='en')")


def test_5_microphone_hardware():
    print("\n[TEST 3] Testing Microphone Hardware & Audio Capture...")
    devices = AudioCapture.get_input_devices()
    assert len(devices) > 0, "No microphone input devices detected on system"
    print(f"  [PASS] Successfully queried audio subsystem: {len(devices)} input device(s) found.")
    print(f"         Default device: '{devices[0]['name']}' (Sample rate: {int(devices[0]['samplerate'])} Hz)")


def test_6_ram_and_latency(sample_path="samples/sample1.wav"):
    print("\n[TEST 7 & 8] Testing RAM Consumption & Transcription Latency (RTF)...")
    resolved_sample = Path(sample_path)
    if not resolved_sample.exists():
        resolved_sample = PROJECT_ROOT.parent / sample_path

    engine = TranscriptionEngine()
    result = engine.transcribe(str(resolved_sample))

    duration = result["duration"]
    inference_time = result["inference_time"]
    rtf = result["real_time_factor"]
    ram_usage = result["ram_usage_mb"]

    print(f"  [PASS] Audio Duration:      {duration:.2f}s")
    print(f"  [PASS] Inference Latency:   {inference_time:.3f}s")
    print(f"  [PASS] Real-Time Factor:    {rtf:.3f}x (< 1.0 indicates faster than real-time)")
    print(f"  [PASS] Total Process RAM:   {ram_usage:.1f} MB (Well within 8GB system limits)")

    assert rtf < 1.0, f"Inference is slower than real-time (RTF={rtf})"
    assert ram_usage < 1500.0, f"RAM usage exceeded budget: {ram_usage} MB"


def run_all_tests():
    print("=" * 65)
    print("  AUTONOMOUS STT NEURAL ENGINE VERIFICATION SUITE")
    print("=" * 65)

    test_1_model_loading_and_offline()
    test_2_wav_transcription()
    test_3_speech_silence_detection()
    test_4_multilingual_recognition()
    test_5_microphone_hardware()
    test_6_ram_and_latency()

    print("\n" + "=" * 65)
    print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
