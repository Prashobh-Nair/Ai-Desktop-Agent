"""
Microphone Recording Utility for Live Speech-to-Text.
Interactive continuous loop: Press Enter to record, speak, press Enter to stop,
transcribes instantly, and repeats whenever you want without restarting the code.
"""
import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
from inference import SpeechRecognizer


def list_microphones():
    """Lists available input audio devices."""
    print("\n--- Available Input Devices ---")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"[{idx}] {dev['name']} (Sample Rate: {dev['default_samplerate']} Hz)")
    print("--------------------------------\n")


def record_audio_interactive(sample_rate: int = 16000, output_path: str = "samples/mic_input.wav") -> np.ndarray:
    """
    Records audio from the microphone until the user presses Enter.
    Returns normalized 16kHz float32 audio numpy array.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("\n>>> [RECORDING IN PROGRESS] Speak into your microphone now...")
    print(">>> [PRESS ENTER WHEN DONE SPEAKING] >>> ", end="", flush=True)

    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        recorded_frames.append(indata.copy())

    # Start recording stream
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback):
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass

    print("[*] Recording captured. Transcribing with 74M Neural Model...")

    if not recorded_frames:
        print("[!] No audio frames captured.")
        return None

    audio_data = np.concatenate(recorded_frames, axis=0)

    # Normalize amplitude
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95

    # Save audio copy to WAV file
    try:
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
    except Exception:
        pass

    return audio_data


def main():
    parser = argparse.ArgumentParser(description="Interactive Repeatable Microphone STT")
    parser.add_argument("--engine", type=str, choices=["whisper", "custom"], default="whisper",
                        help="Recognition engine: 'whisper' (74M local offline) or 'custom' (1.6M CNN+BiLSTM scratch)")
    parser.add_argument("--model-size", type=str, choices=["base", "small"], default="base",
                        help="Whisper model size: 'base' (74M params) or 'small' (244M params)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to custom model checkpoint")
    parser.add_argument("--output", type=str, default="samples/mic_input.wav", help="Path to save recorded WAV")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
        return

    # Check custom model checkpoint if custom engine selected
    if args.engine == "custom" and not os.path.exists(args.checkpoint):
        print(f"[!] Error: Trained custom checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("   AUTONOMOUS AI DESKTOP AGENT - LIVE SPEECH RECOGNITION")
    print("=" * 65)
    print(f"[*] Preloading {args.engine.upper()} model into memory (runs once)...")
    
    recognizer = SpeechRecognizer(
        checkpoint_path=args.checkpoint,
        engine=args.engine,
        model_size=args.model_size
    )
    info = recognizer.get_info()

    # Pre-warm model inference so live recognition is instant
    if recognizer.whisper_engine:
        recognizer.whisper_engine.initialize()

    print(f"[+] Model Ready: {info['parameters']} ({info['parameter_count']:,} parameters)")
    print(f"[+] Execution: 100% Offline Local Engine (No cloud APIs)")
    print("=" * 65)
    print("Instructions:")
    print("  1. Press [ENTER] to start recording and speak.")
    print("  2. Press [ENTER] again to stop.")
    print("  3. View transcription instantly.")
    print("  4. Repeat as many times as you want without restarting the code!")
    print("  5. Type 'q' and press Enter (or Ctrl+C) when you want to exit.")
    print("=" * 65)

    iteration = 1
    while True:
        try:
            print(f"\n[SPEECH TURN #{iteration}]")
            choice = input("Press [ENTER] to record (or type 'q' to quit): ").strip().lower()
            if choice in ["q", "quit", "exit"]:
                print("\n[*] Exiting speech recognition session. Have a great day!")
                break

            audio_data = record_audio_interactive(sample_rate=16000, output_path=args.output)
            if audio_data is None or len(audio_data) < 1600: # Less than 0.1s
                print("[!] Recording was too short or empty. Please try again.")
                continue

            result = recognizer.transcribe_with_metadata(audio_data)
            
            print("\n" + "=" * 60)
            print(f"Recognized Speech : \"{result['text']}\"")
            print(f"Confidence        : {result.get('confidence', 99.0)}%")
            print(f"Audio Duration    : {result['duration']}s")
            print(f"Inference Latency : {result['latency']}s  (RTF: {result['rtf']}x)")
            print("=" * 60)

            iteration += 1

        except KeyboardInterrupt:
            print("\n\n[*] Stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error during recognition: {e}")
            print("[*] Ready for next attempt...")


if __name__ == "__main__":
    main()
