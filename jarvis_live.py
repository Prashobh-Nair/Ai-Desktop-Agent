"""
Autonomous AI Desktop Agent - Phase 1: Conversational JARVIS
Pipeline:
  [CONTINUOUS MIC / "Hey Jarvis"] -> [1: Faster-Whisper 244M] -> [2: Hybrid Brain + Memory] -> [3: Interruptible TTS] -> [OS/Browser/Vision]

Features:
- Wake word: "Hey Jarvis" (with 25s active follow-up conversation window)
- Continuous hands-free voice listening with auto-calibrated noise floor
- Interruptions while speaking (voice barge-in or Space/Enter key)
- 100% RAM-only audio streaming (zero disk clutter)
"""

import argparse
import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import msvcrt
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SpeechRecognizer
from src.executor import ActionExecutor
from src.vision_engine import LocalScreenVision
from brain_module.jarvis.brain import Brain

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)  # 100ms chunks
ACTIVE_WINDOW_SECS = 25.0

WAKE_RE = re.compile(
    r"^(?:hey|ok|okay|hi|hello|yo)?\s*,?\s*jarvis\b[\s,.:;!?-]*(?P<rest>.*)$",
    re.IGNORECASE,
)


def extract_wake_command(text: str) -> Tuple[bool, str]:
    """Checks if text invokes 'Hey Jarvis' and returns (triggered, remaining_command)."""
    clean = (text or "").strip()
    m = WAKE_RE.match(clean)
    if m:
        return True, (m.group("rest") or "").strip()
    return False, clean


def calibrate_noise_floor(seconds: float = 0.8) -> float:
    """Measures ambient room noise RMS and returns a speech threshold."""
    frames = []
    num_chunks = max(3, int(seconds / 0.1))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
        for _ in range(num_chunks):
            data, _ = stream.read(CHUNK_SAMPLES)
            frames.append(float(np.sqrt(np.mean(data ** 2))))
    ambient = float(np.median(frames)) if frames else 0.008
    return max(0.014, min(0.06, ambient * 2.6))


def _monitor_barge_in(stop_event: threading.Event, barge_threshold: float) -> None:
    """Stops TTS playback if user presses Space/Enter or speaks loudly over Jarvis."""
    start_t = time.monotonic()
    loud_chunks = 0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b" ", b"\r", b"\n"):
                        stop_event.set()
                        break

                data, _ = stream.read(CHUNK_SAMPLES)
                # Brief 0.45s guard to avoid speaker startup click
                if time.monotonic() - start_t < 0.45:
                    continue
                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms > barge_threshold:
                    loud_chunks += 1
                    if loud_chunks >= 3:  # ~300ms sustained speech
                        stop_event.set()
                        break
                else:
                    loud_chunks = 0
    except Exception:
        pass


def speak_reply(text: str, tts_engine=None, barge_threshold: float = 0.06) -> bool:
    """
    Speaks text in memory (RAM-only, zero disk files) with interruption support.
    Returns True if interrupted by the user, False if finished normally.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return False

    stop_event = threading.Event()
    interrupted = False
    monitor = threading.Thread(
        target=_monitor_barge_in, args=(stop_event, barge_threshold), daemon=True
    )
    monitor.start()

    try:
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
                stream = sd.get_stream()
                while stream is not None and stream.active:
                    if stop_event.is_set():
                        sd.stop()
                        interrupted = True
                        break
                    time.sleep(0.05)
                return interrupted
            except Exception:
                pass

        safe = clean.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 1; $s.Speak('{safe}')"
        )
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000,
        )
        while proc.poll() is None:
            if stop_event.is_set():
                proc.terminate()
                interrupted = True
                break
            time.sleep(0.05)
        return interrupted
    finally:
        stop_event.set()


def listen_utterance(threshold: float, max_seconds: float = 15.0) -> Optional[np.ndarray]:
    """Continuously listens in RAM until a complete spoken utterance is captured."""
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

    if not speech_frames:
        return None

    audio_data = np.concatenate(speech_frames, axis=0).flatten()
    max_val = np.abs(audio_data).max()
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95
    return audio_data


def record_audio_interactive(sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Push-to-talk fallback: records audio in RAM until Enter is pressed."""
    print("\n>>> [RECORDING] Speak now... [PRESS ENTER WHEN DONE] >>> ", end="", flush=True)
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
    parser = argparse.ArgumentParser(description="Live Conversational Jarvis")
    parser.add_argument("--push-to-talk", action="store_true", help="Use Enter-key push-to-talk instead of continuous wake-word listening")
    args = parser.parse_args()

    print("=" * 70)
    print("   AI-POWERED AUTONOMOUS DESKTOP AGENT: CONVERSATIONAL JARVIS")
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

    # 3. Brain + Executor + Screen Vision
    print("[3/3] Loading Brain, OS/Browser Executor & Screen Vision...")
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
    print(f"      [OK] Brain ready ({brain.mode} -> {engine_desc}, {time.time() - t0:.2f}s)")

    threshold = 0.02
    if not args.push_to_talk:
        print("\n[*] Calibrating microphone noise floor (stay quiet for 1s)...")
        threshold = calibrate_noise_floor()
        print(f"    [OK] Speech threshold: {threshold:.4f} (barge-in: {max(threshold * 3.0, 0.075):.4f})")

    barge_threshold = max(threshold * 3.0, 0.075)
    print("-" * 70)
    if args.push_to_talk:
        print("Mode: Push-to-Talk (Press Enter to speak, 'q' to quit)")
    else:
        print("Mode: Continuous Hands-Free | Wake Word: 'Hey Jarvis'")
        print("Tip : Press [Space] or speak loudly to interrupt Jarvis mid-sentence.")
        print("      Press Ctrl+C to exit.")
    print("-" * 70)

    speak_reply("Hello, I am Jarvis. Say Hey Jarvis whenever you need me.", tts_engine, barge_threshold)

    active_until = 0.0
    turn = 1

    while True:
        try:
            if args.push_to_talk:
                print(f"\n[TURN #{turn}] Press [ENTER] to speak (or 'q' to quit): ", end="", flush=True)
                choice = sys.stdin.readline().strip()
                if choice.lower() in ("q", "quit", "exit"):
                    speak_reply("Goodbye! Have a great day.", tts_engine, barge_threshold)
                    break
                audio_data = record_audio_interactive()
            else:
                in_active = time.monotonic() < active_until
                status_tag = "ACTIVE (listening...)" if in_active else "STANDBY (say 'Hey Jarvis')"
                print(f"\n[{status_tag}] ", end="", flush=True)
                audio_data = listen_utterance(threshold)

            if audio_data is None or len(audio_data) < 2400:
                continue

            t_stt = time.time()
            stt_result = stt_recognizer.transcribe_with_metadata(audio_data)
            recognized_text = stt_result["text"].strip()
            latency_stt = time.time() - t_stt
            if not recognized_text:
                continue

            # Wake word gating in continuous mode
            if not args.push_to_talk:
                in_active = time.monotonic() < active_until
                woke, remainder = extract_wake_command(recognized_text)
                if not in_active and not woke:
                    continue
                if woke and not remainder:
                    print(f"\n>>> WAKE WORD: \"{recognized_text}\"")
                    active_until = time.monotonic() + ACTIVE_WINDOW_SECS
                    speak_reply("Yes, I am listening.", tts_engine, barge_threshold)
                    continue
                if woke and remainder:
                    recognized_text = remainder

            print(f"\n>>> YOU    : \"{recognized_text}\" (STT: {latency_stt:.2f}s)")

            lower_cmd = recognized_text.lower().strip(".,!? ")
            if lower_cmd in ("exit", "quit", "shut down jarvis", "bye jarvis"):
                speak_reply("Goodbye! Have a great day.", tts_engine, barge_threshold)
                break
            if lower_cmd in ("go to sleep", "stop listening", "standby"):
                active_until = 0.0
                speak_reply("Going to standby. Say Hey Jarvis when you need me.", tts_engine, barge_threshold)
                continue

            # Brain Reasoning
            t_brain = time.time()
            brain_out = brain.plan(recognized_text, session_id="live_jarvis")
            plan = brain_out.get("plan", {})
            intent = plan.get("intent", "chat")
            spoken_reply = plan.get("reply") or "I processed your request."
            backend = brain_out.get("backend", "model")
            latency_brain = time.time() - t_brain

            steps = plan.get("steps") or []
            if any(s.get("action") == "see_screen" for s in steps):
                print("[*] Analyzing screen with local SmolVLM-256M...")
                spoken_reply = vision.analyze_screen(recognized_text)
                steps = [s for s in steps if s.get("action") != "see_screen"]
                plan["steps"] = steps

            print(f">>> JARVIS : \"{spoken_reply}\" ({backend} | {intent} | {latency_brain:.2f}s)")

            # Interruptible TTS
            interrupted = speak_reply(spoken_reply, tts_engine, barge_threshold)
            if interrupted:
                print("[*] Interrupted — listening...")

            # Execute OS / Browser steps
            if steps:
                if plan.get("needs_confirmation"):
                    speak_reply("Please type yes in the terminal to confirm.", tts_engine, barge_threshold)
                    confirm = input("[!] Confirm risky system action? (yes/no): ").strip().lower()
                    if confirm not in ("y", "yes"):
                        print("[*] Action cancelled.")
                        continue
                for log in executor.execute_plan(plan):
                    print(f"[ACTION] {log}")

            active_until = time.monotonic() + ACTIVE_WINDOW_SECS
            turn += 1

        except KeyboardInterrupt:
            print("\nShutting down Jarvis. Goodbye!")
            break


if __name__ == "__main__":
    main()
