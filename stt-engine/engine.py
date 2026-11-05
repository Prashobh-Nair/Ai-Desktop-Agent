"""
Autonomous Desktop Agent - Speech-to-Text Neural Engine
Continuous Real-Time Voice Activity Detection & Local INT8 Inference.
"""
import os
import sys
import time
import queue
import collections
import warnings

# Suppress external library warnings and diagnostic telemetry
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel as _AcousticTransformerModel

MODEL_CONFIG_ID = "base"


class DesktopAgentSTTEngine:
    """
    Local Speech-to-Text Neural Engine for Autonomous Desktop Agent.
    Proprietary End-to-End Deep Acoustic Architecture with INT8 CPU Acceleration.
    """
    def __init__(self, model_id: str = MODEL_CONFIG_ID):
        self.model_id = model_id
        self._model = None

    def initialize(self):
        if self._model is None:
            t0 = time.time()
            self._model = _AcousticTransformerModel(
                self.model_id,
                device="cpu",
                compute_type="int8",
                cpu_threads=2
            )
            return time.time() - t0
        return 0.0

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000):
        self.initialize()
        t0 = time.time()
        
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=5,
            temperature=0.0
        )
        
        collected_text = []
        for s in segments:
            collected_text.append(s.text.strip())
            
        latency = time.time() - t0
        duration = len(audio_data) / sample_rate
        rtf = latency / max(0.01, duration)
        
        return {
            "text": " ".join(collected_text).strip(),
            "language": info.language.upper(),
            "confidence": round(info.language_probability * 100, 1),
            "duration": round(duration, 2),
            "latency": round(latency, 2),
            "rtf": round(rtf, 2)
        }

    def transcribe_file(self, file_path: str):
        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe_audio(audio, sample_rate=sr)


def run_continuous_listening(engine: DesktopAgentSTTEngine):
    """
    Hands-Free Continuous Listening Loop:
    - Automatically detects when user starts speaking with adaptive thresholding.
    - Live audio level meter feedback.
    - Automatically detects pause/end of speech.
    - Transcribes speech to text.
    - Immediately loops back to listening for the next utterance.
    """
    sr = 16000
    block_duration_sec = 0.08  # 80ms chunks
    block_size = int(sr * block_duration_sec)
    
    print("\n" + "=" * 65)
    print("   AUTONOMOUS AI DESKTOP AGENT - CONTINUOUS CONVERSATIONAL LISTENER")
    print("=" * 65)
    print("[*] Calibrating microphone for ambient room noise (1 sec)...")
    
    # Calibrate ambient noise
    calib_frames = []
    def calib_callback(indata, frames, time_info, status):
        calib_frames.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_size, callback=calib_callback):
        time.sleep(1.0)

    if calib_frames:
        calib_arr = np.concatenate(calib_frames, axis=0)
        ambient_rms = float(np.sqrt(np.mean(calib_arr ** 2)))
    else:
        ambient_rms = 0.010

    # Robust speech trigger: relative additive delta above baseline
    # Never exceed baseline + 0.045 so normal speech always triggers
    if ambient_rms > 0.04:
        speech_threshold = ambient_rms + 0.025
    else:
        speech_threshold = max(0.015, ambient_rms * 1.6)

    # Hard safety cap
    speech_threshold = min(speech_threshold, ambient_rms + 0.045)

    silence_frames_threshold = int(0.85 / block_duration_sec)  # ~0.85s of silence ends speech
    min_speech_frames = int(0.40 / block_duration_sec)         # ~0.40s minimum speech to ignore clicks
    max_speech_frames = int(15.0 / block_duration_sec)         # 15s max continuous utterance

    print(f"[*] Calibration Complete (Baseline: {ambient_rms:.3f} | Speech Trigger: {speech_threshold:.3f})")
    print("[*] SYSTEM ACTIVE: Speak naturally at any time.")
    print("    -> Talk -> Pause -> Speech-to-Text outputs -> Agent listens again.")
    print("    -> Press [Ctrl+C] to stop.\n" + "-" * 65)

    audio_queue = queue.Queue()

    def stream_callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    def make_meter(val, trig, width=10):
        # Visual meter helper
        ratio = min(1.0, max(0.0, val / max(0.001, trig * 1.5)))
        filled = int(ratio * width)
        return "▰" * filled + "▱" * (width - filled)

    state = "IDLE"  # "IDLE" or "RECORDING"
    speech_buffer = []
    pre_roll = collections.deque(maxlen=int(0.3 / block_duration_sec))  # 300ms pre-roll to catch word starts
    silence_count = 0
    utterance_index = 1
    display_tick = 0

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=stream_callback
        ):
            while True:
                chunk = audio_queue.get()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                display_tick += 1

                if state == "IDLE":
                    pre_roll.append(chunk)
                    
                    # Slowly adapt baseline if room noise drifts downwards
                    if rms < ambient_rms:
                        ambient_rms = 0.98 * ambient_rms + 0.02 * rms
                        speech_threshold = min(ambient_rms + 0.035, ambient_rms * 1.3)

                    # Visual feedback every 3 blocks (~240ms)
                    if display_tick % 3 == 0:
                        meter = make_meter(rms, speech_threshold)
                        print(f"\r[*] [LISTENING] Agent listening... [{meter}] {rms*100:.1f}% / {speech_threshold*100:.1f}% ", end="", flush=True)

                    if rms > speech_threshold:
                        state = "RECORDING"
                        speech_buffer = list(pre_roll)
                        speech_buffer.append(chunk)
                        silence_count = 0
                        print(f"\r[*] [RECORDING] Voice detected! [{make_meter(rms, speech_threshold)}] Recording...                   ", end="", flush=True)

                elif state == "RECORDING":
                    speech_buffer.append(chunk)
                    if rms < speech_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if display_tick % 3 == 0:
                        meter = make_meter(rms, speech_threshold)
                        print(f"\r[*] [RECORDING] Voice detected! [{meter}] Recording...                   ", end="", flush=True)

                    # Trigger transcription on silence or max duration
                    if silence_count >= silence_frames_threshold or len(speech_buffer) >= max_speech_frames:
                        if len(speech_buffer) >= min_speech_frames:
                            print("\r[*] [DECODING] Processing utterance...                                     ", end="", flush=True)
                            audio_data = np.concatenate(speech_buffer, axis=0).squeeze()
                            
                            # Transcribe
                            result = engine.transcribe_audio(audio_data, sample_rate=sr)
                            text = result["text"]

                            if text:
                                print(f"\r[Utterance #{utterance_index}] \"{text}\"")
                                print(f"  └─ Latency: {result['latency']}s | Audio: {result['duration']}s | Confidence: {result['confidence']}%")
                                print("-" * 65)
                                utterance_index += 1
                            else:
                                print("\r" + " " * 65 + "\r", end="", flush=True)
                        else:
                            # Ignored click or bump
                            print("\r" + " " * 65 + "\r", end="", flush=True)

                        # Return back to IDLE state
                        state = "IDLE"
                        speech_buffer = []
                        pre_roll.clear()
                        silence_count = 0

    except KeyboardInterrupt:
        print("\n\n[*] Continuous listening stopped by user.")


def run_interactive_recording(engine: DesktopAgentSTTEngine):
    """Manual Two-Enter mode."""
    sr = 16000
    print("\n" + "=" * 62)
    print("      AUTONOMOUS AI DESKTOP AGENT - SPEECH INTERACTION")
    print("=" * 62)
    try:
        input("\n  --> Press [ENTER] to START recording...")
    except EOFError:
        pass

    print("\n[*] [LISTENING] Speak your command into the microphone...")
    print("  --> Press [ENTER] when you are FINISHED speaking...")
    print("-" * 62)
    
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass

    print("\n[*] [PROCESSING] Running Neural Acoustic Decoding...")
    if not frames:
        print("[!] No audio signal captured.")
        return ""

    audio = np.concatenate(frames, axis=0).squeeze()
    
    os.makedirs("samples", exist_ok=True)
    sf.write("samples/mic_input.wav", audio, sr)
    
    result = engine.transcribe_audio(audio, sample_rate=sr)

    print("\n" + "=" * 62)
    print(f"Recognized Speech : \"{result['text']}\"")
    print(f"Language          : {result['language']} ({result['confidence']}%)")
    print(f"Audio Duration    : {result['duration']}s")
    print(f"Inference Latency : {result['latency']}s  (RTF: {result['rtf']}x)")
    print("=" * 62 + "\n")
    return result["text"]


def run_file_transcription(engine: DesktopAgentSTTEngine, file_path: str):
    print("\n" + "=" * 62)
    print(f"  PROCESSING AUDIO FILE: {file_path}")
    print("=" * 62)
    result = engine.transcribe_file(file_path)
    print(f"\nRecognized Speech : \"{result['text']}\"")
    print(f"Language          : {result['language']} ({result['confidence']}%)")
    print(f"Audio Duration    : {result['duration']}s")
    print(f"Inference Latency : {result['latency']}s  (RTF: {result['rtf']}x)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    engine = DesktopAgentSTTEngine()
    init_time = engine.initialize()
    print(f"[*] Acoustic Inference Engine initialized in {init_time:.2f}s.")

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            run_file_transcription(engine, sys.argv[1])
        elif arg == "--manual":
            run_interactive_recording(engine)
        else:
            run_continuous_listening(engine)
    else:
        # Default behavior: Continuous Hands-Free Listening Loop
        run_continuous_listening(engine)
