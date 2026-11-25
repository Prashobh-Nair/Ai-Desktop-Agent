"""
Autonomous AI Desktop Agent - Fully Automatic Hands-Free JARVIS
Strict Microphone Lifecycle (No Wake Word):
  [1. MIC ON: Listen for command]
       -> User finishes speaking
  [2. MIC OFF: Transcribe (Whisper) -> Brain -> Execute Tasks -> Speak Reply (TTS)]
       -> JARVIS finishes speaking completely
  [3. MIC ON: Automatically listen for next command]

Important: Microphone is NEVER active during AI processing, task execution, or TTS playback.
"""

import argparse
import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import queue
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SpeechRecognizer
from src.autonomous import AutonomousGoalRunner, BackgroundTaskManager
from src.executor import ActionExecutor
from src.vision_engine import LocalScreenVision
from brain_module.jarvis.brain import Brain

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)  # 100ms chunks


def calibrate_noise_floor(seconds: float = 0.8) -> float:
    """Measures ambient room noise RMS and returns a speech detection threshold."""
    frames = []
    num_chunks = max(3, int(seconds / 0.1))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
        for _ in range(num_chunks):
            data, _ = stream.read(CHUNK_SAMPLES)
            frames.append(float(np.sqrt(np.mean(data ** 2))))
    ambient = float(np.median(frames)) if frames else 0.008
    return max(0.014, min(0.06, ambient * 2.6))


def speak_reply(text: str, tts_engine=None) -> None:
    """
    Speaks text completely in memory (RAM-only, zero disk files) with MIC strictly OFF.
    Blocks until audio playback finishes completely.
    """
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
    subprocess.run(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,
    )


def listen_utterance(threshold: float, max_seconds: float = 15.0) -> Optional[np.ndarray]:
    """
    Turns microphone ON, waits for a spoken command, and closes the microphone
    immediately once the user finishes speaking (~1.1s trailing silence).
    """
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time_info, _status):
        audio_q.put(indata.copy())

    speech_frames = []
    pre_buffer = []
    speaking = False
    speech_hits = 0
    silence_hits = 0
    max_chunks = int(max_seconds / 0.1)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES, callback=callback
    ):
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if not speaking:
                pre_buffer.append(chunk)
                if len(pre_buffer) > 3:
                    pre_buffer.pop(0)
                if rms > threshold:
                    speech_hits += 1
                    if speech_hits >= 2:  # ~200ms speech onset
                        speaking = True
                        speech_frames.extend(pre_buffer)
                        silence_hits = 0
                else:
                    speech_hits = 0
            else:
                speech_frames.append(chunk)
                if rms <= threshold * 0.85:
                    silence_hits += 1
                else:
                    silence_hits = 0

                if silence_hits >= 11 or len(speech_frames) >= max_chunks:  # ~1.1s silence
                    break

    # Microphone stream is now completely CLOSED (OFF)
    if not speech_frames:
        return None

    audio_data = np.concatenate(speech_frames, axis=0).flatten()
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95
    return audio_data


def record_audio_interactive(sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Optional push-to-talk fallback: records audio in RAM until Enter is pressed."""
    print("\n>>> [MIC ON] Speak now... [PRESS ENTER WHEN DONE] >>> ", end="", flush=True)
    recorded_frames = []

    def audio_callback(indata, _frames, _time_info, _status):
        recorded_frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback):
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass

    if not recorded_frames:
        return None

    audio_data = np.concatenate(recorded_frames, axis=0).flatten()
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95
    return audio_data


def main():
    parser = argparse.ArgumentParser(description="Live Hands-Free JARVIS (No Wake Word)")
    parser.add_argument("--push-to-talk", action="store_true", help="Use Enter-key push-to-talk instead of automatic VAD listening")
    args = parser.parse_args()

    print("=" * 70)
    print("   AI-POWERED AUTONOMOUS DESKTOP AGENT: HANDS-FREE JARVIS")
    print("=" * 70)
    print("[*] Initializing modules (RAM-only audio, zero disk recording)...\n")

    # 1. TTS (Kokoro-82M first, SAPI5 fallback)
    print("[1/3] Loading TTS Engine...")
    t0 = time.time()
    tts_engine = None
    try:
        from tts_module.src.tts_engine import get_tts_engine
        tts_engine = get_tts_engine()
        print(f"      [OK] Kokoro-82M ready ({time.time() - t0:.2f}s)")
    except Exception:
        print("      [OK] Fast Windows SAPI5 TTS ready")

    # 2. Local Faster-Whisper STT
    print("[2/3] Loading Faster-Whisper Local STT...")
    t0 = time.time()
    try:
        stt_recognizer = SpeechRecognizer(engine="whisper", model_size="small.en")
        stt_recognizer.whisper_engine.initialize()
        print(f"      [OK] Whisper-Small.en (244M) ready ({time.time() - t0:.2f}s)")
    except Exception:
        stt_recognizer = SpeechRecognizer(engine="whisper", model_size="base")
        stt_recognizer.whisper_engine.initialize()
        print(f"      [OK] Whisper-Base (74M) ready ({time.time() - t0:.2f}s)")

    # 3. Brain + Executor + Screen Vision + Autonomous Runner
    print("[3/3] Loading Brain, Tools, Memory & Autonomous Runner...")
    t0 = time.time()
    brain = Brain(mode="auto")
    executor = ActionExecutor()
    vision = LocalScreenVision()
    bg_manager = BackgroundTaskManager()
    goal_runner = AutonomousGoalRunner(brain=brain, executor=executor, vision=vision)
    if getattr(brain, "gemini_available", False):
        engine_desc = "Gemini 3.1 Flash Lite + Local Rules"
    elif brain.ollama_available:
        engine_desc = "Local Neural LLM (Ollama) + Rules"
    else:
        engine_desc = "Deterministic Rule-Based Engine"
    print(f"      [OK] Brain ready ({brain.mode} -> {engine_desc}, {time.time() - t0:.2f}s)")

    threshold = 0.02
    if not args.push_to_talk:
        print("\n[*] Calibrating microphone noise floor (stay quiet for 1s)...")
        threshold = calibrate_noise_floor()
        print(f"    [OK] Speech threshold: {threshold:.4f}")

    print("-" * 70)
    if args.push_to_talk:
        print("Mode: Push-to-Talk (Press Enter to speak, 'q' to quit)")
    else:
        print("Mode: Automatic Hands-Free (No Wake Word)")
        print("      [MIC ON] Listen -> [MIC OFF] Think, Execute & Speak -> [MIC ON]")
        print("      Press Ctrl+C to exit.")
    print("-" * 70)

    # Speak greeting with MIC strictly OFF before opening mic for turn 1
    speak_reply("Hello, I am Jarvis. I am ready for your commands.", tts_engine)
    time.sleep(0.25)

    turn = 1
    while True:
        try:
            # Speak any background notifications while MIC is still OFF
            for note in bg_manager.pop_notifications():
                print(f"\n[NOTIFICATION] {note}")
                speak_reply(note, tts_engine)
                time.sleep(0.2)

            # --- PHASE A: MIC ON (wait for voice command) ---
            if args.push_to_talk:
                print(f"\n[TURN #{turn}] Press [ENTER] to speak (or 'q' to quit): ", end="", flush=True)
                choice = sys.stdin.readline().strip()
                if choice.lower() in ("q", "quit", "exit"):
                    speak_reply("Goodbye! Have a great day.", tts_engine)
                    break
                audio_data = record_audio_interactive()
            else:
                print(f"\n[TURN #{turn} | MIC ON] Listening for your command...", flush=True)
                audio_data = listen_utterance(threshold)

            # --- PHASE B: MIC OFF (process, execute, and speak) ---
            print("[MIC OFF] Processing...", flush=True)

            if audio_data is None or len(audio_data) < 2400:
                continue

            t_stt = time.time()
            stt_result = stt_recognizer.transcribe_with_metadata(audio_data)
            recognized_text = stt_result["text"].strip()
            latency_stt = time.time() - t_stt
            if not recognized_text:
                continue

            print(f">>> YOU    : \"{recognized_text}\" (STT: {latency_stt:.2f}s)")

            lower_cmd = recognized_text.lower().strip(".,!? ")
            if lower_cmd in ("exit", "quit", "shut down jarvis", "bye jarvis", "goodbye"):
                speak_reply("Goodbye! Have a great day.", tts_engine)
                break

            if lower_cmd in ("task status", "background tasks", "what tasks are running"):
                summary = bg_manager.status_summary()
                print(f">>> JARVIS : \"{summary}\"")
                speak_reply(summary, tts_engine)
                time.sleep(0.25)
                continue

            if lower_cmd.startswith(("run in background ", "in background ")):
                bg_obj = re.sub(r"^(?:run\s+)?in\s+background\s+", "", recognized_text, flags=re.IGNORECASE).strip()
                tid = bg_manager.submit(bg_obj[:32], lambda o=bg_obj: goal_runner.run_objective(o, session_id="bg")[0])
                msg = f"Started {tid} in the background. I will notify you when it finishes."
                print(f">>> JARVIS : \"{msg}\"")
                speak_reply(msg, tts_engine)
                time.sleep(0.25)
                continue

            def _confirm_cli(prompt_txt: str) -> bool:
                speak_reply(f"{prompt_txt} Please type yes in the terminal to confirm.", tts_engine)
                return input("[!] Confirm risky action? (yes/no): ").strip().lower() in ("y", "yes")

            # Brain Reasoning + Tool/Action Execution (MIC is OFF)
            t_brain = time.time()
            spoken_reply, action_logs = goal_runner.run_objective(
                recognized_text, session_id="live_jarvis", confirm_fn=_confirm_cli
            )
            latency_brain = time.time() - t_brain

            print(f">>> JARVIS : \"{spoken_reply}\" ({latency_brain:.2f}s)")
            for log in action_logs:
                print(f"[ACTION] {log}")

            # Speak full reply with MIC strictly OFF
            speak_reply(spoken_reply, tts_engine)

            # Brief echo guard before turning MIC ON again
            time.sleep(0.25)
            turn += 1

        except KeyboardInterrupt:
            print("\nShutting down Jarvis. Goodbye!")
            break


if __name__ == "__main__":
    main()
