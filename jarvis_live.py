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

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import subprocess
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
from src.executor import ActionExecutor
from src.vision_engine import LocalScreenVision
from brain_module.jarvis.brain import Brain


def speak_reply(text: str, tts_engine=None) -> None:
    """Speaks text in memory (RAM-only, zero disk files) using Kokoro-82M or Windows SAPI5."""
    clean = " ".join((text or "").split())
    if not clean:
        return
    if tts_engine is not None and getattr(tts_engine, "_kokoro", None) is not None:
        try:
            processed = tts_engine.text_processor.preprocess_text(clean)
            samples, sr = tts_engine._kokoro.create(
                processed,
                voice=tts_engine.config.voice,
                speed=tts_engine.config.speed,
                lang=tts_engine.config.lang,
            )
            sd.play(samples, sr)
            sd.wait()
            return
        except Exception:
            pass
    safe = clean.replace("'", "''")
    ps_cmd = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; $s.Speak('{safe}')"
    )
    subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd], capture_output=True)


def record_audio_interactive(sample_rate: int = 16000) -> np.ndarray:
    """Records audio from microphone in RAM only (no file saved) until Enter is pressed."""
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

    return audio_data


def main():
    print("=" * 70)
    print("   AI-POWERED AUTONOMOUS DESKTOP AGENT: LIVE JARVIS ASSISTANT")
    print("=" * 70)
    print("Complete End-to-End Multimodal Pipeline:")
    print("  [VOICE] -> [1: Whisper 74M STT] -> [2: Jarvis Brain] -> [3: Kokoro 82M TTS] -> [AUDIO]")
    print("=" * 70)
    print("[*] Initializing local neural models across all 3 modules...\n")

    # 1. Initialize Module 3 (Kokoro TTS first before STT/LLM reserve RAM, with SAPI5 fallback)
    print("[1/3] Loading Module 3: Kokoro-82M Neural Text-to-Speech Engine...")
    t0 = time.time()
    tts_engine = None
    try:
        from tts_module.src.tts_engine import get_tts_engine
        tts_engine = get_tts_engine()
        print(f"      [OK] Kokoro TTS ready ({time.time() - t0:.2f}s)")
    except Exception:
        print("      [OK] Using fast Windows SAPI5 TTS fallback")

    # 2. Initialize Module 1 (Local Whisper STT)
    print("[2/3] Loading Module 1: Local Whisper-Small.en (244M) STT Engine...")
    t0 = time.time()
    try:
        stt_recognizer = SpeechRecognizer(engine="whisper", model_size="small.en")
        stt_recognizer.whisper_engine.initialize()
        print(f"      [OK] Whisper-Small.en (244M) ready ({time.time() - t0:.2f}s)")
    except Exception:
        stt_recognizer = SpeechRecognizer(engine="whisper", model_size="base")
        stt_recognizer.whisper_engine.initialize()
        print(f"      [OK] Whisper-Base (74M) ready ({time.time() - t0:.2f}s)")

    # 3. Initialize Module 2 (Brain) & Module 4 (OS/Browser Executor + Screen Vision)
    print("[3/3] Loading Module 2: Jarvis Brain, OS/Browser Executor & Screen Vision...")
    t0 = time.time()
    brain = Brain(mode="auto")
    executor = ActionExecutor()
    vision = LocalScreenVision()
    if getattr(brain, "gemini_available", False):
        engine_desc = "Gemini 3.1 Flash Lite + Local Rules"
    elif brain.ollama_available:
        engine_desc = "Local Neural LLM (Ollama) + Rules"
    else:
        engine_desc = "Deterministic Rule-Based Engine"
    print(f"      [OK] Brain ready in mode: '{brain.mode}' -> {engine_desc} ({time.time() - t0:.2f}s)")
    print("      [OK] OS & Browser Action Executor ready.\n")

    print("-" * 70)
    print("All modules are online and ready! Welcome to Jarvis.")
    print("You can speak general knowledge questions (e.g. 'what is quantum physics?')")
    print("or desktop/browser commands (e.g. 'open chrome', 'new tab', 'time').")
    print("-" * 70)

    # Introductory greeting from Jarvis
    speak_reply("Hello, I am Jarvis. All modules are online and I am ready for your commands.", tts_engine)

    turn = 1
    while True:
        print(f"\n[SPEECH TURN #{turn}]")
        print("Press [ENTER] to record your voice (or type 'q' to quit): ", end="", flush=True)
        user_choice = sys.stdin.readline().strip()

        if user_choice.lower() in ("q", "quit", "exit"):
            print("\nShutting down Jarvis. Goodbye!")
            speak_reply("Goodbye! Have a great day.", tts_engine)
            break

        # Record voice from microphone
        audio_data = record_audio_interactive()
        if audio_data is None or len(audio_data) < 1600:
            print("[!] Audio too short or empty. Please try again.")
            continue

        # Step 1: STT (Speech-to-Text)
        t_stt = time.time()
        print("\n[*] Transcribing speech with local Whisper model...")
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

        # Handle local screen understanding (Moondream VLM)
        steps = plan.get("steps") or []
        if any(s.get("action") == "see_screen" for s in steps):
            print("[*] Capturing screen and analyzing with local SmolVLM-256M Vision model...")
            spoken_reply = vision.analyze_screen(recognized_text)
            steps = [s for s in steps if s.get("action") != "see_screen"]
            plan["steps"] = steps

        print(f">>> JARVIS   : \"{spoken_reply}\" (Engine: {backend} | Intent: {intent} | Latency: {latency_brain:.2f}s)")

        # Step 3: Speak reply first so feedback is immediate
        print("[*] Speaking reply...")
        t_tts = time.time()
        speak_reply(spoken_reply, tts_engine)
        latency_tts = time.time() - t_tts

        # Step 4: Execute OS & Browser Actions (if any)
        steps = plan.get("steps") or []
        if steps:
            if plan.get("needs_confirmation"):
                confirm = input("[!] Confirm risky system action? (yes/no): ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("[*] Action cancelled.")
                    speak_reply("Action cancelled.", tts_engine)
                    turn += 1
                    continue
            exec_logs = executor.execute_plan(plan)
            for log in exec_logs:
                print(f"[ACTION] {log}")

        total_latency = latency_stt + latency_brain + latency_tts
        print(f"[*] Turn #{turn} complete! Total pipeline latency: {total_latency:.2f}s")
        turn += 1


if __name__ == "__main__":
    main()
