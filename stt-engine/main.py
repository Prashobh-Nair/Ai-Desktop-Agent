"""
Autonomous Desktop Agent - Unified Speech-to-Text CLI Entry Point
End-to-end command line interface for offline local speech recognition.
"""
import sys
import os
import warnings
import argparse
from pathlib import Path

# Suppress external warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings
from src.model_manager import AcousticModelManager
from src.audio_capture import AudioCapture
from src.transcription_engine import TranscriptionEngine
from src.logger import log_info, log_success, log_error, get_process_memory_mb


def cmd_download():
    """Model weights initialization."""
    print("=" * 60)
    print("  ACOUSTIC NEURAL MODEL INITIALIZATION")
    print("=" * 60)
    manager = AcousticModelManager()
    path = manager.download()
    print(f"\n[OK] Model successfully verified at: {path}")


def cmd_transcribe_file(file_path: str, language: str = None, vad: bool = True):
    """Transcribes an audio file."""
    print("=" * 60)
    print(f"  PROCESSING AUDIO FILE: {file_path}")
    print("=" * 60)
    engine = TranscriptionEngine()
    result = engine.transcribe(file_path, language=language, use_vad=vad)

    print("\n--- TRANSCRIPTION RESULT ---")
    print(f"Text:             \"{result['text']}\"")
    print(f"Language:         {result['language'].upper()} (Confidence: {result['language_probability']*100:.1f}%)")
    print(f"Audio Duration:   {result['duration']}s")
    print(f"Inference Time:   {result['inference_time']}s")
    print(f"Real-Time Factor: {result['real_time_factor']}x (Lower than 1.0 is faster than real-time)")
    print(f"RAM Usage:        {result['ram_usage_mb']} MB")

    if result.get("segments"):
        print("\n--- TIMESTAMPS ---")
        for seg in result["segments"]:
            print(f"  [{seg['start']:>5.2f}s -> {seg['end']:>5.2f}s] {seg['text']}")
    print("=" * 60)


def cmd_live_mic(duration: float = None, device: int = None, language: str = None, continuous: bool = True):
    """Captures microphone and runs speech-to-text."""
    if continuous and duration is None:
        from engine import DesktopAgentSTTEngine, run_continuous_listening
        eng = DesktopAgentSTTEngine()
        eng.initialize()
        run_continuous_listening(eng)
        return

    print("=" * 60)
    print("  LIVE MICROPHONE SPEECH-TO-TEXT (OFFLINE INT8 ENGINE)")
    print("=" * 60)
    capture = AudioCapture()
    engine = TranscriptionEngine()
    engine.initialize()

    if duration is not None:
        audio = capture.record_duration(duration=duration, device=device)
    else:
        audio = capture.record_interactive(device=device)

    if len(audio) == 0:
        print("[!] No audio recorded.")
        return

    print("\n[*] Processing speech through Neural Acoustic Decoder...")
    result = engine.transcribe(audio, language=language)

    print("\n" + "=" * 60)
    print(f"Recognized Speech: \"{result['text']}\"")
    print(f"Language:         {result['language'].upper()} ({result['language_probability']*100:.1f}%)")
    print(f"Latency:          {result['inference_time']}s (RTF: {result['real_time_factor']}x)")
    print("=" * 60 + "\n")


def cmd_benchmark(file_path: str):
    """Benchmarks processing performance."""
    print("=" * 60)
    print(f"  PERFORMANCE BENCHMARK: {file_path}")
    print("=" * 60)
    engine = TranscriptionEngine()
    engine.initialize()

    ram_start = get_process_memory_mb()
    print(f"[*] Base RAM usage: {ram_start:.1f} MB")

    # Run 3 consecutive passes
    runs = []
    res = None
    for i in range(3):
        res = engine.transcribe(file_path)
        runs.append(res["inference_time"])
        print(f"  Pass {i+1}: Duration: {res['duration']}s | Inference: {res['inference_time']}s | RTF: {res['real_time_factor']}x")

    avg_time = sum(runs) / len(runs)
    avg_rtf = avg_time / max(0.01, res["duration"] if res else 1.0)
    ram_end = get_process_memory_mb()

    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Average Inference Time: {avg_time:.3f}s")
    print(f"Average RTF:            {avg_rtf:.3f}x real-time")
    print(f"Peak RAM Usage:         {ram_end:.1f} MB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Desktop Agent - Standalone Speech-to-Text Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # mic
    mic_p = subparsers.add_parser("mic", help="Record from microphone and transcribe")
    mic_p.add_argument("--duration", type=float, default=None, help="Fixed duration in seconds")
    mic_p.add_argument("--device", type=int, default=None, help="Input device index")
    mic_p.add_argument("--language", type=str, default=None, help="Target language code (e.g. 'en')")

    # transcribe
    file_p = subparsers.add_parser("transcribe", help="Transcribe audio file")
    file_p.add_argument("--file", type=str, required=True, help="Path to audio file (WAV, MP3, FLAC)")
    file_p.add_argument("--language", type=str, default=None, help="Target language code")
    file_p.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark latency and memory")
    bench_p.add_argument("--file", type=str, required=True, help="Path to audio file for benchmark")

    # download / init
    subparsers.add_parser("init", help="Verify neural model weights")

    # list-devices
    subparsers.add_parser("list-devices", help="List available microphone audio devices")

    args = parser.parse_args()

    if args.command == "init" or args.command == "download":
        cmd_download()
    elif args.command == "transcribe":
        cmd_transcribe_file(args.file, language=args.language, vad=not args.no_vad)
    elif args.command == "mic":
        cmd_live_mic(duration=args.duration, device=args.device, language=args.language)
    elif args.command == "benchmark":
        cmd_benchmark(args.file)
    elif args.command == "list-devices":
        AudioCapture().print_devices()
    else:
        # Default behavior: run live mic interactive mode
        cmd_live_mic()


if __name__ == "__main__":
    main()
