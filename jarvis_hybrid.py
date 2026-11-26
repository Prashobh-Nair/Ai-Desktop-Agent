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
from src.plugin_manager import PluginManager
from src.ui_bridge import JarvisUIBridge
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


def listen_utterance(
    threshold: float,
    silence_timeout: float = 1.25,
    max_seconds: float = 15.0,
    ui_bridge: Optional[JarvisUIBridge] = None,
) -> Optional[np.ndarray]:
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
            if ui_bridge is not None:
                norm_level = min(1.0, rms / max(threshold * 2.8, 0.04))
                ui_bridge.set_mic_level(norm_level)
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


def _speak_edge_tts_ram(
    text: str,
    voice: str = "en-IN-PrabhatNeural",
    rate: str = "+18%",
    ui_bridge: Optional[JarvisUIBridge] = None,
) -> bool:
    """Synthesizes Edge TTS audio in RAM, plays it via sounddevice, and streams live amplitude."""
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
        sd.play(audio, samplerate=24000, blocking=False)
        if ui_bridge is not None:
            hop = int(24000 * 0.033)
            total_samples = len(audio)
            t0 = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - t0
                pos = int(elapsed * 24000)
                if pos >= total_samples:
                    break
                seg = audio[pos : min(total_samples, pos + hop)]
                rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
                ui_bridge.set_tts_level(min(1.0, rms * 4.5))
                time.sleep(0.033)
            ui_bridge.set_tts_level(0.0)
        sd.wait()
        return True
    except Exception:
        return False


def speak_reply(
    text: str,
    fallback_tts=None,
    ui_bridge: Optional[JarvisUIBridge] = None,
) -> None:
    if not text:
        return
    if ui_bridge is not None:
        ui_bridge.set_state("SPEAKING")
    try:
        if _speak_edge_tts_ram(text, ui_bridge=ui_bridge):
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
    finally:
        if ui_bridge is not None:
            ui_bridge.set_state("IDLE")


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
    parser.add_argument("--no-ui", action="store_true", help="Run without launching the visual UI")
    parser.add_argument("--full-ui", action="store_true", help="Also open the full JARVIS orb window on startup")
    args = parser.parse_args()

    print("\n============================================================")
    print("  HYBRID JARVIS (API Primary + Local Fallback)")
    print("============================================================")

    ui_bridge: Optional[JarvisUIBridge] = None
    if not args.no_ui:
        ui_bridge = JarvisUIBridge()
        ui_bridge.start(open_window=args.full_ui, start_island=True)

    brain = Brain(mode="auto")
    vision = LocalScreenVision()
    vision.start_live_watch(interval=2.5)
    executor = ActionExecutor()
    plugin_manager = PluginManager(
        llm=getattr(brain, "groq", None) or getattr(brain, "gemini", None)
    )
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
                if ui_bridge is not None:
                    ui_bridge.set_state("IDLE")
                recognized_text = input(f"[TURN #{turn}] YOU > ").strip()
                if not recognized_text:
                    continue
                if ui_bridge is not None:
                    ui_bridge.set_state("THINKING")
            else:
                print(f"\n[TURN #{turn} | MIC ON] Listening...", flush=True)
                if ui_bridge is not None:
                    ui_bridge.set_state("LISTENING")
                audio = listen_utterance(threshold, ui_bridge=ui_bridge)
                print("[MIC OFF] Processing...", flush=True)
                if ui_bridge is not None:
                    ui_bridge.set_state("THINKING")
                if audio is None or len(audio) < 2400:
                    if ui_bridge is not None:
                        ui_bridge.set_state("IDLE")
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
                    if ui_bridge is not None:
                        ui_bridge.set_state("IDLE")
                    continue
                print(f">>> YOU    : \"{recognized_text}\"")

            lower_cmd = recognized_text.lower().strip(".,!? ")
            if lower_cmd in ("exit", "quit", "bye jarvis", "goodbye", "shut down jarvis"):
                speak_reply("Alright, catch you later!", fallback_tts, ui_bridge=ui_bridge)
                break

            if lower_cmd in ("task status", "background tasks", "what tasks are running"):
                summary = bg_manager.status_summary()
                print(f">>> JARVIS : \"{summary}\"")
                speak_reply(summary, fallback_tts, ui_bridge=ui_bridge)
                continue

            if lower_cmd.startswith(("run in background ", "in background ")):
                bg_obj = re.sub(r"^(?:run\s+)?in\s+background\s+", "", recognized_text, flags=re.IGNORECASE).strip()
                tid = bg_manager.submit(bg_obj[:32], lambda o=bg_obj: goal_runner.run_objective(o, session_id="bg")[0])
                msg = f"Got it, running {tid} in the background. I'll let you know when it's done."
                print(f">>> JARVIS : \"{msg}\"")
                speak_reply(msg, fallback_tts, ui_bridge=ui_bridge)
                continue

            plan_res = brain.plan(recognized_text)
            plan = plan_res["plan"]
            reply = plan.get("reply") or "Got it."

            steps = plan.get("steps") or []
            if steps:
                if plan.get("needs_confirmation"):
                    confirm_q = plan.get("clarification_question") or f"{reply} Just say yes to go ahead."
                    print(f">>> JARVIS : \"{confirm_q}\"")
                    speak_reply(confirm_q, fallback_tts, ui_bridge=ui_bridge)
                    if args.text:
                        ans = input("Confirm (yes/no) > ").strip().lower()
                    else:
                        print("[MIC ON] Waiting for confirmation...", flush=True)
                        if ui_bridge is not None:
                            ui_bridge.set_state("LISTENING")
                        c_audio = listen_utterance(threshold, silence_timeout=1.0, max_seconds=6.0, ui_bridge=ui_bridge)
                        print("[MIC OFF]", flush=True)
                        if ui_bridge is not None:
                            ui_bridge.set_state("THINKING")
                        ans = stt.transcribe_with_metadata(c_audio)["text"].lower() if c_audio is not None and len(c_audio) >= 1600 else ""
                    if not any(w in ans for w in ("yes", "yeah", "confirm", "sure", "ok", "okay", "do it", "proceed")):
                        reply = "No worries, cancelled."
                        print(f">>> JARVIS : \"{reply}\"")
                        speak_reply(reply, fallback_tts, ui_bridge=ui_bridge)
                        continue

                if ui_bridge is not None:
                    ui_bridge.start_task(recognized_text, steps)
                outputs = executor.execute_plan(
                    plan,
                    memory=brain.memory,
                    vision=vision,
                    plugin_manager=plugin_manager,
                    step_callback=(lambda i, s, l: ui_bridge.update_step(i, s, l)) if ui_bridge is not None else None,
                )
                err_outputs = [o for o in (outputs or []) if o and any(k in o.lower() for k in ("could not", "failed", "not found", "unavailable", "screen check:"))]
                if ui_bridge is not None:
                    ui_bridge.finish_task(had_error=bool(err_outputs), error_msg=(" ".join(err_outputs) if err_outputs else ""))
                if outputs:
                    # Handle on-demand plugin creation / reconnection interactively
                    setup_marker = next((o for o in outputs if o and o.startswith("__SETUP_PLUGIN__:")), None)
                    if setup_marker:
                        app_to_setup = setup_marker.split(":", 1)[1].strip() or "custom"
                        reply = plugin_manager.interactive_setup(
                            app_to_setup,
                            speak_fn=lambda m: (print(f">>> JARVIS : \"{m}\""), speak_reply(m, fallback_tts, ui_bridge=ui_bridge)),
                        )
                    else:
                        info_actions = {"see_screen", "get_time", "get_date", "web_fetch", "file_op", "git_op", "memory_op", "calendar_op", "run_terminal", "plugin_op"}
                        has_info = any(s.get("action") in info_actions for s in steps)
                        err_outputs = [o for o in outputs if o and any(k in o.lower() for k in ("could not", "failed", "not found", "unavailable", "screen check:"))]
                        if err_outputs:
                            reply = " ".join(err_outputs)
                        elif has_info:
                            info_outs = [o for s, o in zip(steps, outputs) if o and s.get("action") in info_actions]
                            reply = " ".join(info_outs) if info_outs else " ".join(o for o in outputs if o)

                        # If user tried to use an unconfigured plugin, offer to set it up now
                        if "integration set up yet. Want to configure it?" in reply:
                            print(f">>> JARVIS : \"{reply}\"")
                            speak_reply(reply, fallback_tts, ui_bridge=ui_bridge)
                            if args.text:
                                ans = input("Configure now (yes/no) > ").strip().lower()
                            else:
                                print("[MIC ON] Waiting for reply...", flush=True)
                                if ui_bridge is not None:
                                    ui_bridge.set_state("LISTENING")
                                c_audio = listen_utterance(threshold, silence_timeout=1.0, max_seconds=6.0, ui_bridge=ui_bridge)
                                print("[MIC OFF]", flush=True)
                                if ui_bridge is not None:
                                    ui_bridge.set_state("THINKING")
                                ans = stt.transcribe_with_metadata(c_audio)["text"].lower() if c_audio is not None and len(c_audio) >= 1600 else ""
                            if any(w in ans for w in ("yes", "yeah", "sure", "ok", "okay", "configure", "set it up")):
                                target_app = next((s.get("target") for s in steps if s.get("action") == "plugin_op"), "") or "custom"
                                reply = plugin_manager.interactive_setup(
                                    target_app,
                                    speak_fn=lambda m: (print(f">>> JARVIS : \"{m}\""), speak_reply(m, fallback_tts, ui_bridge=ui_bridge)),
                                )
                            else:
                                reply = "No worries, we can set it up anytime."

            print(f">>> JARVIS : \"{reply}\"")
            speak_reply(reply, fallback_tts, ui_bridge=ui_bridge)
            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nExiting JARVIS.")
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(0.5)

    vision.stop_live_watch()
    if ui_bridge is not None:
        ui_bridge.stop()


if __name__ == "__main__":
    main()
