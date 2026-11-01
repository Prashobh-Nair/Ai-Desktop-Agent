"""
Microphone Recording Utility for Live Speech-to-Text.
Records audio from the default or chosen microphone, saves to WAV,
and feeds it to our custom trained STT model.
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


def record_audio(duration: float = None, sample_rate: int = 16000, output_path: str = "samples/mic_input.wav"):
    """
    Records audio from the microphone.
    If duration is specified, records for that many seconds.
    Otherwise, records interactively until the user presses Enter.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if duration is not None:
        print(f"[*] Recording for {duration} seconds at {sample_rate} Hz... Speak now!")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("[*] Recording complete.")
        # Peak normalization
        max_val = np.abs(audio_data).max()
        if max_val > 1e-4:
            audio_data = audio_data / max_val * 0.95
    else:
        print("\n[*] Interactive Recording Mode:")
        print("    Press ENTER to START recording...")
        try:
            input()
        except EOFError:
            pass
            
        print("[*] RECORDING STARTED! Speak into your microphone. (Press ENTER to STOP)...")
        recorded_frames = []
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[!] Status warning: {status}", file=sys.stderr)
            recorded_frames.append(indata.copy())

        # Start input stream
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback):
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
                
        print("[*] RECORDING STOPPED.")
        if not recorded_frames:
            print("[!] No audio frames captured.")
            return None
            
        audio_data = np.concatenate(recorded_frames, axis=0)

    # Peak amplitude normalization for clear acoustic feature extraction
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95

    # Save to standard 16kHz WAV file
    sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
    print(f"[+] Saved recording to: {output_path} (Duration: {len(audio_data)/sample_rate:.2f}s)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Microphone STT Recording Utility")
    parser.add_argument("--engine", type=str, choices=["wav2vec2", "custom"], default="wav2vec2",
                        help="Recognition engine: 'wav2vec2' (high-accuracy production) or 'custom' (CNN+BiLSTM from scratch)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to custom model checkpoint")
    parser.add_argument("--duration", type=float, default=None, help="Fixed recording duration in seconds (optional)")
    parser.add_argument("--output", type=str, default="samples/mic_input.wav", help="Path to save recorded WAV")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
        return

    # Check custom model checkpoint if custom engine selected
    if args.engine == "custom" and not os.path.exists(args.checkpoint):
        print(f"[!] Error: Trained custom checkpoint '{args.checkpoint}' not found.")
        print("    Please train the model first using: python train.py")
        sys.exit(1)

    print(f"[*] Initializing SpeechRecognizer (Engine: {args.engine.upper()})...")
    recognizer = SpeechRecognizer(checkpoint_path=args.checkpoint, engine=args.engine)

    # Record audio
    recorded_wav = record_audio(duration=args.duration, output_path=args.output)
    if recorded_wav and os.path.exists(recorded_wav):
        print(f"\n[*] Running {args.engine.upper()} Neural Network CTC Inference...")
        result = recognizer.transcribe(recorded_wav)
        print("\n" + "="*50)
        print(f"Recognized Speech: \"{result}\"")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()

