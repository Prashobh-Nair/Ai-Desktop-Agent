"""
Autonomous AI Desktop Agent - Complete Live Jarvis Voice Assistant
Pipeline:
  [USER SPEAKS] -> [MODULE 1: WHISPER STT (74M)] -> [MODULE 2: JARVIS BRAIN] -> [MODULE 3: KOKORO TTS (82M)] -> [SPEAKERS]

Authors:
- Module 1: STT (Speech-to-Text) - Prashobh Nair
- Module 2: Brain (LLM & Planning) - Sahaj Patel
- Module 3: TTS (Text-to-Speech) - Zeel Vasoya
"""

import os
import sys
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SpeechRecognizer
from brain_module.jarvis.brain import Brain
from tts_module.src.tts_engine import speak, get_tts_engine


def record_audio_interactive(sample_rate: int = 16000, output_path: str = "samples/mic_input.wav") -> np.ndarray:
    """Records audio from microphone until the user presses Enter."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("\n>>> [RECORDING IN PROGRESS] Speak into your microphone now...")
    print(">>> [PRESS ENTER WHEN DONE SPEAKING] >>> ", end="", flush=True)

    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        recorded_frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback):
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass

    if not recorded_frames:
        return None

    audio_data = np.concatenate(recorded_frames, axis=0)

    # Normalize volume
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95

    try:
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
    except Exception:
        pass

    return audio_data


def main():
    print("=" * 70)
    print("   AI-POWERED AUTONOMOUS DESKTOP AGENT: LIVE JARVIS ASSISTANT")
    print("=" * 70)
    print("Complete End-to-End Multimodal Pipeline:")
    print("  [VOICE] -> [1: Whisper 74M STT] -> [2: Jarvis Brain] -> [3: Kokoro 82M TTS] -> [AUDIO]")
    print("=" * 70)
    print("[*] Initializing local neural models across all 3 modules...\n")

    # 1. Initialize Module 1 (Whisper STT)
    print("[1/3] Loading Module 1: Whisper-Base (74M Parameters) STT Engine...")
    t0 = time.time()
    stt_recognizer = SpeechRecognizer(engine="whisper", model_size="base")
    print(f"      [OK] Whisper STT ready ({time.time() - t0:.2f}s)")

    # 2. Initialize Module 2 (Brain)
    print("[2/3] Loading Module 2: Jarvis Brain / Reasoning Engine...")
    t0 = time.time()
    brain = Brain(mode="auto")
    engine_desc = "Local Neural LLM (Jarvis via Ollama)" if brain.ollama_available else "Deterministic Rule-Based Fallback"
    print(f"      [OK] Brain ready in mode: '{brain.mode}' -> Engine: {engine_desc} ({time.time() - t0:.2f}s)")

    # 3. Initialize Module 3 (Kokoro TTS)
    print("[3/3] Loading Module 3: Kokoro-82M Neural Text-to-Speech Engine...")
    t0 = time.time()
    tts_engine = get_tts_engine()
    print(f"      [OK] Kokoro TTS ready ({time.time() - t0:.2f}s)\n")

    print("-" * 70)
    print("All modules are online and ready! Welcome to Jarvis.")
    print("You can speak general knowledge questions (e.g. 'what is quantum physics?')")
    print("or desktop control commands (e.g. 'open chrome', 'play song', 'time').")
    print("-" * 70)

    # Introductory greeting from Jarvis
    speak("Hello, I am Jarvis. All three modules are online and I am ready for your commands.")

    turn = 1
    while True:
        print(f"\n[SPEECH TURN #{turn}]")
        print("Press [ENTER] to record your voice (or type 'q' to quit): ", end="", flush=True)
        user_choice = sys.stdin.readline().strip()

        if user_choice.lower() in ("q", "quit", "exit"):
            print("\nShutting down Jarvis. Goodbye!")
            speak("Goodbye! Have a great day.")
            break

        # Record voice from microphone
        audio_data = record_audio_interactive()
        if audio_data is None or len(audio_data) < 1600:
            print("[!] Audio too short or empty. Please try again.")
            continue

        # Step 1: STT (Speech-to-Text)
        t_stt = time.time()
        print("\n[*] Transcribing speech with 74M Neural Model...")
        stt_result = stt_recognizer.transcribe_with_metadata(audio_data)
        recognized_text = stt_result["text"].strip()
        latency_stt = time.time() - t_stt

        print(f"\n>>> YOU SAID : \"{recognized_text}\" (STT latency: {latency_stt:.2f}s)")

        if not recognized_text:
            print("[!] No speech detected. Please speak louder into your microphone.")
            continue

        # Step 2: Brain Reasoning
        t_brain = time.time()
        print("[*] Reasoning with Jarvis Brain...")
        brain_out = brain.plan(recognized_text, session_id="live_jarvis")
        plan = brain_out.get("plan", {})
        intent = plan.get("intent", "chat")
        spoken_reply = plan.get("reply") or "I processed your request."
        backend = brain_out.get("backend", "model")
        latency_brain = time.time() - t_brain

        print(f">>> JARVIS   : \"{spoken_reply}\" (Engine: {backend} | Intent: {intent} | Latency: {latency_brain:.2f}s)")

        # Step 3: TTS (Text-to-Speech Audio Playback)
        print("[*] Speaking reply with Kokoro-82M Neural Engine...")
        t_tts = time.time()
        speak(spoken_reply, blocking=True)
        latency_tts = time.time() - t_tts

        total_latency = latency_stt + latency_brain + latency_tts
        print(f"[*] Turn #{turn} complete! Total pipeline latency: {total_latency:.2f}s")
        turn += 1


if __name__ == "__main__":
    main()
