"""
Main CLI Interface for Kokoro-82M TTS Module.
Provides command-line synthesis, voice selection, speed adjustment,
audio playback, interactive loop, and benchmark execution.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tts_module.src.tts_engine import TTSEngine, generate_speech, speak, get_available_voices
from tts_module.src.audio_player import play_audio
from tts_module.src.config import AVAILABLE_VOICES, DEFAULT_CONFIG


def list_voices():
    """Displays all available voice options with metadata."""
    print("=" * 70)
    print("AVAILABLE KOKORO-82M VOICES")
    print("=" * 70)
    print(f"{'Voice ID':<15} | {'Name':<10} | {'Gender':<8} | {'Accent':<10} | {'Description'}")
    print("-" * 70)
    for vid, meta in AVAILABLE_VOICES.items():
        print(f"{vid:<15} | {meta['name']:<10} | {meta['gender']:<8} | {meta['accent']:<10} | {meta['description']}")
    print("=" * 70)
    print(f"Total curated voices: {len(AVAILABLE_VOICES)}")
    print(f"Default voice: '{DEFAULT_CONFIG.default_voice}'")


def interactive_mode(voice: str = "af_heart", speed: float = 1.0):
    """Interactive shell for live speech synthesis."""
    print("=" * 70)
    print("KOKORO-82M INTERACTIVE SPEECH SYNTHESIS CONSOLE")
    print("=" * 70)
    print(f"Active Voice: {voice} | Speed: {speed:.2f}x")
    print("Type any text and press Enter to hear it spoken through your speakers.")
    print("Commands:")
    print("  :voice <name>  - Change active voice (e.g., :voice am_adam)")
    print("  :speed <float> - Change speaking speed (e.g., :speed 1.2)")
    print("  :voices        - List popular voices")
    print("  :exit or :quit - Exit console")
    print("=" * 70)

    engine = TTSEngine()
    engine.initialize()

    current_voice = voice
    current_speed = speed

    while True:
        try:
            user_input = input("\n[Agent Input] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in (":exit", ":quit", "exit", "quit"):
                print("Exiting interactive console. Goodbye!")
                break

            if user_input.startswith(":voice "):
                new_voice = user_input.split(" ", 1)[1].strip()
                current_voice = new_voice
                print(f"[+] Voice changed to: {current_voice}")
                continue

            if user_input.startswith(":speed "):
                try:
                    new_speed = float(user_input.split(" ", 1)[1].strip())
                    current_speed = new_speed
                    print(f"[+] Speed set to: {current_speed:.2f}x")
                except ValueError:
                    print("[-] Invalid speed. Please enter a float (e.g. 1.1).")
                continue

            if user_input == ":voices":
                list_voices()
                continue

            print("[*] Generating speech...")
            wav_path = engine.generate_speech(
                text=user_input,
                voice=current_voice,
                speed=current_speed
            )
            metrics = engine.last_metrics
            print(f"[+] Generated in {metrics.get('generation_time', 0):.2f}s "
                  f"(Audio: {metrics.get('audio_duration', 0):.2f}s | RTF: {metrics.get('rtf', 0):.2f}x)")
            print("[*] Playing audio...")
            engine.audio_player.play(wav_path, blocking=True)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[-] Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro-82M Neural Text-to-Speech Engine for AI Desktop Agent"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize into speech"
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="af_heart",
        help="Voice ID to use (e.g., af_heart, am_adam, bf_emma). Default: af_heart"
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speaking speed multiplier (0.5 to 2.0). Default: 1.0"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Optional custom output WAV filename"
    )
    parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play audio through speakers after generating"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Display list of available voice styles and exit"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive speech synthesis shell"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance and resource benchmarks"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run end-to-end AI Desktop Agent demonstration (STT -> LLM -> TTS)"
    )

    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if args.benchmark:
        from tts_module.tests.benchmark import run_benchmark
        run_benchmark()
        return

    if args.demo:
        from tts_module.demo_pipeline import run_pipeline_demo
        run_pipeline_demo()
        return

    if args.interactive:
        interactive_mode(voice=args.voice, speed=args.speed)
        return

    if args.text:
        engine = TTSEngine()
        engine.initialize()
        wav_path = engine.generate_speech(
            text=args.text,
            voice=args.voice,
            speed=args.speed,
            output_filename=args.output
        )
        print(f"[SUCCESS] Audio saved to: {wav_path}")
        metrics = engine.last_metrics
        print(f"Metrics: Duration: {metrics.get('audio_duration', 0):.2f}s | "
              f"Gen Time: {metrics.get('generation_time', 0):.2f}s | "
              f"RTF: {metrics.get('rtf', 0):.2f}x")
        
        if args.play:
            print("[*] Playing audio through Windows speakers...")
            engine.audio_player.play(wav_path, blocking=True)
            print("[+] Playback complete.")
        return

    # If no arguments provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()

