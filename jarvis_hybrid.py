"""
Hybrid JARVIS - API + Local Fallback Hands-Free Terminal Assistant
Strict Microphone Lifecycle (No Wake Word):
  1. MIC ON  -> Wait for voice command (auto-stop on silence)
  2. MIC OFF -> Transcribe (Groq Whisper API -> Local Whisper)
             -> Brain (Gemini API -> Local Ollama, zero RulesPlanner)
             -> Execute Tasks / Screen Vision (Gemini Vision -> Local SmolVLM)
             -> Speak Reply (edge-tts en-IN-PrabhatNeural in RAM -> Local TTS)
  3. MIC ON  -> Automatically listen for next command
"""

import argparse
import asyncio
import io
import os
import queue
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Keep CPU thread pools bounded on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import av
import edge_tts
import numpy as np
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain_module.jarvis.brain import Brain
from inference import SpeechRecognizer
from src.autonomous import AutonomousGoalRunner, BackgroundTaskManager
from src.executor import ActionExecutor
from src.vision_engine import LocalScreenVision
from src.whisper_engine import build_hotwords

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)


def calibrate_noise_floor(seconds: float = 0.8) -> float:
    frames = []
    num_chunks = max(3, int(seconds / 0.1))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
        for _ in range(num_chunks):
            data, _ = stream.read(CHUNK_SAMPLES)
            frames.append(float(np.sqrt(np.mean(data ** 2))))
    ambient = float(np.median(frames)) if frames else 0.008
    return max(0.014, min(0.06, ambient * 2.6))


def listen_utterance(threshold: float, silence_timeout: float = 1.25, max_seconds: float = 15.0) -> Optional[np.ndarray]:
    """Keeps mic open until speech starts and finishes, then closes mic immediately."""
    q_audio: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        q_audio.put(indata.copy())

    pre_roll = []
    speech_chunks = []
    started = False
    silence_elapsed = 0.0
    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES, callback=_callback):
        while True:
            try:
                chunk = q_audio.get(timeout=0.5).flatten()
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if not started:
                pre_roll.append(chunk)
                if len(pre_roll) > 4:
                    pre_roll.pop(0)
                if rms >= threshold:
                    started = True
                    speech_chunks.extend(pre_roll)
                    silence_elapsed = 0.0
                    start_time = time.time()
                    print("      [Hearing speech...]", flush=True)
            else:
                speech_chunks.append(chunk)
                if rms < threshold * 0.75:
                    silence_elapsed += 0.1
                else:
                    silence_elapsed = 0.0
                if silence_elapsed >= silence_timeout or (time.time() - start_time) >= max_seconds:
                    break

    return np.concatenate(speech_chunks) if speech_chunks else None


def _speak_edge_tts_ram(text: str, voice: str = "en-IN-PrabhatNeural", rate: str = "+18%") -> bool:
    """Synthesizes Edge TTS audio in RAM and plays it synchronously via sounddevice."""
    async def _fetch() -> bytes:
        comm = edge_tts.Communicate(text, voice, rate=rate)
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    try:
        mp3_bytes = asyncio.run(_fetch())
        if not mp3_bytes:
            return False
        container = av.open(io.BytesIO(mp3_bytes))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=24000)
        pcm_parts = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray().flatten()
                if len(arr):
                    pcm_parts.append(arr)
        if not pcm_parts:
            return False
        audio = np.concatenate(pcm_parts).astype(np.float32)
        sd.play(audio, samplerate=24000, blocking=True)
        sd.wait()
        return True
    except Exception:
        return False


def speak_reply(text: str, fallback_tts=None) -> None:
    if not text:
        return
    if _speak_edge_tts_ram(text):
        return
    if fallback_tts is not None:
        try:
            fallback_tts.speak(text)
            return
        except Exception:
            pass
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
    except Exception:
        pass


def _init_fallback_tts():
    try:
        from tts_module.engine import TTSEngine
        t = TTSEngine()
        t.initialize()
        return t
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid JARVIS (API + Local Fallback)")
    parser.add_argument("--text", action="store_true", help="Type commands instead of microphone input")
    args = parser.parse_args()

    print("\n============================================================")
    print("  HYBRID JARVIS (API Primary + Local Fallback)")
    print("============================================================")

    brain = Brain(mode="auto")
    vision = LocalScreenVision()
    vision.start_live_watch(interval=2.5)
    executor = ActionExecutor()
    bg_manager = BackgroundTaskManager()
    goal_runner = AutonomousGoalRunner(brain, executor, vision=vision)
    stt = SpeechRecognizer(model_size="small.en")
    fallback_tts = _init_fallback_tts()

    stt_mode = "Groq Whisper large-v3-turbo (API) -> Local Whisper" if stt.groq_engine.is_configured else "Local Whisper-Small.en (244M)"
    if getattr(brain, "groq_available", False):
        brain_mode = "Groq GPT-OSS-20B (Fast API) -> Gemini -> Local Ollama"
    elif brain.gemini_available:
        brain_mode = "Gemini (API) -> Local Ollama"
    else:
        brain_mode = "Local Ollama"
    print(f"  STT   : {stt_mode}")
    print(f"  Brain : {brain_mode}")
    print("  Vision: Gemini Vision (API) -> Local SmolVLM-256M")
    print("  TTS   : Edge-TTS en-IN-PrabhatNeural (RAM) -> Local Fallback")
    print("============================================================\n")

    threshold = 0.02
    if not args.text:
        print("Calibrating room noise floor...", flush=True)
        threshold = calibrate_noise_floor()
        print(f"Ready (mic threshold={threshold:.4f}). Speak anytime; say 'exit' to quit.\n", flush=True)

    turn = 0
    while True:
        turn += 1
        try:
            if args.text:
                recognized_text = input(f"[TURN #{turn}] YOU > ").strip()
                if not recognized_text:
                    continue
            else:
                print(f"\n[TURN #{turn} | MIC ON] Listening...", flush=True)
                audio = listen_utterance(threshold)
                print("[MIC OFF] Processing...", flush=True)
                if audio is None or len(audio) < 2400:
                    continue

                mem_terms = []
                try:
                    for item in brain.memory._load_long_term():
                        mem_terms.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", item.get("fact", "")))
                except Exception:
                    pass
                res = stt.transcribe_with_metadata(audio, hotwords=build_hotwords(mem_terms))
                recognized_text = res["text"].strip()
                if not recognized_text:
                    continue
                print(f">>> YOU    : \"{recognized_text}\"")

            lower_cmd = recognized_text.lower().strip(".,!? ")
            if lower_cmd in ("exit", "quit", "bye jarvis", "goodbye", "shut down jarvis"):
                speak_reply("Goodbye! Have a great day.", fallback_tts)
                break

            if lower_cmd in ("task status", "background tasks", "what tasks are running"):
                summary = bg_manager.status_summary()
                print(f">>> JARVIS : \"{summary}\"")
                speak_reply(summary, fallback_tts)
                continue

            if lower_cmd.startswith(("run in background ", "in background ")):
                bg_obj = re.sub(r"^(?:run\s+)?in\s+background\s+", "", recognized_text, flags=re.IGNORECASE).strip()
                tid = bg_manager.submit(bg_obj[:32], lambda o=bg_obj: goal_runner.run_objective(o, session_id="bg")[0])
                msg = f"Started {tid} in the background. I will notify you when it finishes."
                print(f">>> JARVIS : \"{msg}\"")
                speak_reply(msg, fallback_tts)
                continue

            plan_res = brain.plan(recognized_text)
            plan = plan_res["plan"]
            reply = plan.get("reply") or "Okay."

            steps = plan.get("steps") or []
            if steps:
                if plan.get("needs_confirmation"):
                    confirm_q = plan.get("clarification_question") or f"{reply} Please say yes to confirm."
                    print(f">>> JARVIS : \"{confirm_q}\"")
                    speak_reply(confirm_q, fallback_tts)
                    if args.text:
                        ans = input("Confirm (yes/no) > ").strip().lower()
                    else:
                        print("[MIC ON] Waiting for confirmation...", flush=True)
                        c_audio = listen_utterance(threshold, silence_timeout=1.0, max_seconds=6.0)
                        print("[MIC OFF]", flush=True)
                        ans = stt.transcribe_with_metadata(c_audio)["text"].lower() if c_audio is not None and len(c_audio) >= 1600 else ""
                    if not any(w in ans for w in ("yes", "yeah", "confirm", "sure", "ok", "okay", "do it", "proceed")):
                        reply = "Cancelled."
                        print(f">>> JARVIS : \"{reply}\"")
                        speak_reply(reply, fallback_tts)
                        continue

                outputs = executor.execute_plan(plan, memory=brain.memory, vision=vision)
                if outputs:
                    reply = " ".join(o for o in outputs if o)

            print(f">>> JARVIS : \"{reply}\"")
            speak_reply(reply, fallback_tts)
            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nExiting JARVIS.")
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(0.5)

    vision.stop_live_watch()


if __name__ == "__main__":
    main()
