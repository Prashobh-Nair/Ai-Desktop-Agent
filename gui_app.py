"""
Native Windows Desktop GUI for the AI-Powered Autonomous Desktop Agent.

Simple, clean, light-themed student GUI built with Tkinter:
- High-DPI sharp rendering on Windows.
- Preloads TTS -> STT -> Brain in optimal RAM order for maximum speed.
- Non-blocking execution: reply appears immediately, and TTS speaks asynchronously.
- Persistent SAPI5 fallback speaker (< 40ms) if Kokoro-82M hits low RAM.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import ctypes
from datetime import datetime
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext
from typing import Optional

import numpy as np
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain_module.jarvis.brain import Brain
from src.executor import ActionExecutor

# Enable crisp DPI scaling on Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


# Simple, calm, light-themed color palette (no dark theme, no bright colors)
BG_MAIN = "#f5f6f8"
BG_CARD = "#ffffff"
BG_SUBTLE = "#f0f2f5"
BORDER_COLOR = "#dfe3e8"
TEXT_PRIMARY = "#24292f"
TEXT_MUTED = "#656d76"
ACCENT = "#3b5b7d"
ACCENT_HOVER = "#2f4966"
ACCENT_TEXT = "#ffffff"
BTN_HOVER = "#e4e8ee"
PILL_WAIT_BG = "#f0f2f5"
PILL_WAIT_FG = "#656d76"
PILL_OK_BG = "#edf5ef"
PILL_OK_FG = "#2c663b"
RECORD_BG = "#f6eaea"
RECORD_FG = "#9c3b3b"


class FastSpeaker:
    """Uses Kokoro-82M when available, with a persistent Windows SAPI5 fallback (< 40ms)."""

    def __init__(self):
        self.kokoro = None
        self._sapi_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _get_sapi_proc(self) -> subprocess.Popen:
        if self._sapi_proc is None or self._sapi_proc.poll() is not None:
            init_ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 1; "
                "while (($line = [Console]::In.ReadLine()) -ne $null) { "
                "if ($line.Length -gt 0) { $s.Speak($line) } }"
            )
            self._sapi_proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", init_ps],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
        return self._sapi_proc

    def speak(self, text: str) -> None:
        clean = " ".join((text or "").split())
        if not clean:
            return

        if self.kokoro is not None:
            try:
                self.kokoro.speak(clean, blocking=True)
                return
            except Exception:
                pass

        with self._lock:
            try:
                proc = self._get_sapi_proc()
                if proc.stdin:
                    proc.stdin.write(clean + "\n")
                    proc.stdin.flush()
            except Exception:
                pass

    def stop(self) -> None:
        if self.kokoro is not None:
            try:
                self.kokoro.stop()
            except Exception:
                pass
        with self._lock:
            if self._sapi_proc is not None and self._sapi_proc.poll() is None:
                try:
                    self._sapi_proc.terminate()
                except Exception:
                    pass
                self._sapi_proc = None


def _bind_hover(widget: tk.Widget, normal_bg: str, hover_bg: str) -> None:
    widget.bind("<Enter>", lambda _e: widget.config(bg=hover_bg) if widget["state"] != tk.DISABLED else None)
    widget.bind("<Leave>", lambda _e: widget.config(bg=normal_bg) if widget["state"] != tk.DISABLED else None)


class JarvisDesktopGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-Powered Autonomous Desktop Agent — Jarvis")
        self.root.geometry("860x640")
        self.root.minsize(680, 500)
        self.root.configure(bg=BG_MAIN)

        self.brain: Optional[Brain] = None
        self.executor = ActionExecutor()
        self.stt = None
        self.speaker = FastSpeaker()

        self.speak_enabled = tk.BooleanVar(value=True)
        self.is_recording = False
        self.recorded_frames = []
        self.audio_stream = None
        self.speech_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._start_background_workers()

    def _build_ui(self) -> None:
        # Top Header Card
        header = tk.Frame(self.root, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header.pack(fill=tk.X, padx=18, pady=(16, 10))

        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT, padx=16, pady=12)
        tk.Label(
            title_frame,
            text="AI-Powered Autonomous Desktop Agent",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            title_frame,
            text="Semester 5 Minor Project  •  Local Voice, Reasoning & Desktop Automation",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        # Separate Module Status Pills
        pills = tk.Frame(header, bg=BG_CARD)
        pills.pack(side=tk.RIGHT, padx=16)

        self.pill_stt = self._make_pill(pills, "STT: Loading...")
        self.pill_brain = self._make_pill(pills, "Brain: Loading...")
        self.pill_tts = self._make_pill(pills, "TTS: Loading...")

        # Main Chat Card
        card = tk.Frame(self.root, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 8))

        # Toolbar inside Chat Card
        toolbar = tk.Frame(card, bg=BG_CARD)
        toolbar.pack(fill=tk.X, padx=16, pady=(10, 8))

        tk.Label(
            toolbar,
            text="Assistant Console",
            font=("Segoe UI", 10, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(side=tk.LEFT)

        clear_btn = tk.Button(
            toolbar,
            text="Clear",
            font=("Segoe UI", 9),
            bg=BG_SUBTLE,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=3,
            command=self._clear_chat,
        )
        clear_btn.pack(side=tk.RIGHT)
        _bind_hover(clear_btn, BG_SUBTLE, BTN_HOVER)

        stop_voice_btn = tk.Button(
            toolbar,
            text="Stop Audio",
            font=("Segoe UI", 9),
            bg=BG_SUBTLE,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=3,
            command=self.speaker.stop,
        )
        stop_voice_btn.pack(side=tk.RIGHT, padx=(0, 8))
        _bind_hover(stop_voice_btn, BG_SUBTLE, BTN_HOVER)

        tk.Checkbutton(
            toolbar,
            text="Speak replies",
            variable=self.speak_enabled,
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
            activebackground=BG_CARD,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 12))

        tk.Frame(card, bg=BORDER_COLOR, height=1).pack(fill=tk.X)

        # Chat History Area
        self.chat_box = scrolledtext.ScrolledText(
            card,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=18,
            pady=14,
            spacing1=2,
            spacing3=2,
            state=tk.DISABLED,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)
        self.chat_box.tag_configure("time", foreground="#8c959f", font=("Segoe UI", 8))
        self.chat_box.tag_configure("user", foreground="#2b4c70", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("jarvis", foreground="#2c5e3b", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("meta", foreground="#7a828c", font=("Segoe UI", 8))
        self.chat_box.tag_configure("action", foreground="#3b5b7d", font=("Segoe UI", 9))

        # Live Activity Bar
        activity_bar = tk.Frame(card, bg=BG_CARD)
        activity_bar.pack(fill=tk.X, padx=16, pady=(0, 6))
        self.activity_label = tk.Label(
            activity_bar,
            text="Ready — type a command or press Speak",
            font=("Segoe UI", 8),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        )
        self.activity_label.pack(side=tk.LEFT)

        tk.Frame(card, bg=BORDER_COLOR, height=1).pack(fill=tk.X)

        # Input Bar
        input_bar = tk.Frame(card, bg=BG_SUBTLE)
        input_bar.pack(fill=tk.X)

        entry_wrap = tk.Frame(input_bar, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        entry_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(14, 8), pady=12)

        self.entry = tk.Entry(
            entry_wrap,
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            bd=0,
        )
        self.entry.pack(fill=tk.X, padx=10, pady=7)
        self.entry.bind("<Return>", lambda _e: self._on_send())
        self.entry.focus_set()

        self.mic_btn = tk.Button(
            input_bar,
            text="Speak",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            cursor="hand2",
            padx=14,
            pady=7,
            command=self._toggle_recording,
        )
        self.mic_btn.pack(side=tk.LEFT, padx=(0, 8), pady=12)
        _bind_hover(self.mic_btn, BG_CARD, BTN_HOVER)

        self.send_btn = tk.Button(
            input_bar,
            text="Send",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT,
            fg=ACCENT_TEXT,
            relief=tk.FLAT,
            cursor="hand2",
            padx=18,
            pady=7,
            command=self._on_send,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(0, 14), pady=12)
        _bind_hover(self.send_btn, ACCENT, ACCENT_HOVER)

        # Footer
        footer = tk.Frame(self.root, bg=BG_MAIN)
        footer.pack(fill=tk.X, padx=18, pady=(0, 12))
        tk.Label(
            footer,
            text="Modules:  STT (Prashobh Nair)  •  Brain (Sahaj Patel)  •  TTS (Zeel Vasoya)  •  OS & UI (Mayank)",
            font=("Segoe UI", 8),
            bg=BG_MAIN,
            fg=TEXT_MUTED,
        ).pack(side=tk.LEFT)
        tk.Label(
            footer,
            text="100% Local Offline Execution",
            font=("Segoe UI", 8),
            bg=BG_MAIN,
            fg=TEXT_MUTED,
        ).pack(side=tk.RIGHT)

        self._append_chat(
            "Jarvis",
            "Hello! I am ready. Ask me any question or give a desktop/browser command.",
        )

    def _make_pill(self, parent: tk.Frame, text: str) -> tk.Label:
        lbl = tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 8),
            bg=PILL_WAIT_BG,
            fg=PILL_WAIT_FG,
            padx=8,
            pady=4,
        )
        lbl.pack(side=tk.LEFT, padx=(6, 0))
        return lbl

    def _start_background_workers(self) -> None:
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._init_models, daemon=True).start()

    def _safe_after(self, fn) -> None:
        try:
            self.root.after(0, fn)
        except (tk.TclError, RuntimeError):
            pass

    def _set_activity(self, text: str) -> None:
        self._safe_after(lambda: self.activity_label.config(text=text))

    def _init_models(self) -> None:
        # 1. Load TTS first (before Ollama allocates RAM)
        try:
            from tts_module.src.tts_engine import get_tts_engine
            self.speaker.kokoro = get_tts_engine()
            tts_txt = "TTS: Kokoro-82M"
        except Exception:
            self.speaker.kokoro = None
            tts_txt = "TTS: SAPI5 Fast"
        self._safe_after(lambda: self.pill_tts.config(text=tts_txt, bg=PILL_OK_BG, fg=PILL_OK_FG))

        # 2. Preload STT (Whisper 74M) so first voice turn has zero load delay
        try:
            from inference import SpeechRecognizer
            self.stt = SpeechRecognizer(engine="whisper", model_size="base")
            self.stt.whisper_engine.initialize()
            stt_txt = "STT: Whisper 74M"
        except Exception:
            stt_txt = "STT: Error"
        self._safe_after(lambda: self.pill_stt.config(text=stt_txt, bg=PILL_OK_BG, fg=PILL_OK_FG))

        # 3. Load Brain (prefers Gemini 3.1 Flash Lite if configured, else Ollama)
        try:
            self.brain = Brain(mode="auto")
            if getattr(self.brain, "gemini_available", False):
                brain_txt = "Brain: Gemini 3.1"
            elif self.brain.ollama_available:
                self.brain.model.warmup()
                brain_txt = "Brain: Hybrid (LLM)"
            else:
                brain_txt = "Brain: Rules"
        except Exception:
            brain_txt = "Brain: Error"
        self._safe_after(lambda: self.pill_brain.config(text=brain_txt, bg=PILL_OK_BG, fg=PILL_OK_FG))

    def _tts_worker(self) -> None:
        while True:
            text = self.speech_queue.get()
            if text:
                self._set_activity("Speaking reply...")
                self.speaker.speak(text)
                self._set_activity("Ready — type a command or press Speak")
            self.speech_queue.task_done()

    def _enqueue_speech(self, text: str) -> None:
        if self.speak_enabled.get() and text:
            self.speech_queue.put(text)

    def _append_chat(self, sender: str, text: str, actions=None, meta: str = "") -> None:
        self.chat_box.config(state=tk.NORMAL)
        now = datetime.now().strftime("%I:%M %p").lstrip("0")
        tag = "user" if sender == "You" else "jarvis"
        self.chat_box.insert(tk.END, f"[{now}] ", "time")
        self.chat_box.insert(tk.END, f"{sender}: ", tag)
        self.chat_box.insert(tk.END, f"{text}\n")
        if actions:
            for act in actions:
                self.chat_box.insert(tk.END, f"   [Action] {act}\n", "action")
        if meta:
            self.chat_box.insert(tk.END, f"   {meta}\n", "meta")
        self.chat_box.insert(tk.END, "\n")
        self.chat_box.see(tk.END)
        self.chat_box.config(state=tk.DISABLED)

    def _clear_chat(self) -> None:
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)
        self._append_chat("Jarvis", "Chat cleared. Ready for your commands.")

    def _on_send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._submit_text(text)

    def _submit_text(self, text: str) -> None:
        self._append_chat("You", text)
        self.send_btn.config(state=tk.DISABLED)
        self._set_activity("Thinking...")
        threading.Thread(target=self._process_command, args=(text,), daemon=True).start()

    def _process_command(self, text: str) -> None:
        if self.brain is None:
            self.brain = Brain(mode="auto")

        t0 = time.perf_counter()
        out = self.brain.plan(text, session_id="gui_app")
        plan = out.get("plan", {})
        reply = plan.get("reply") or "Done."
        intent = plan.get("intent", "unknown")
        backend = out.get("backend", "rules")
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        actions_log = []
        if plan.get("steps"):
            if plan.get("needs_confirmation"):
                approved = self._ask_confirmation_sync(reply)
                if approved:
                    actions_log = self.executor.execute_plan(plan)
                    reply = "Confirmed. Executing now."
                else:
                    actions_log = ["Cancelled by user."]
                    reply = "Action cancelled."
            else:
                actions_log = self.executor.execute_plan(plan)

        meta = f"Intent: {intent}  •  Engine: {backend}  •  {latency_ms} ms"
        self._safe_after(lambda: self._append_chat("Jarvis", reply, actions_log, meta))
        self._safe_after(lambda: self.send_btn.config(state=tk.NORMAL))
        self._set_activity("Ready — type a command or press Speak")
        self._enqueue_speech(reply)

    def _ask_confirmation_sync(self, prompt: str) -> bool:
        result = [False]
        ev = threading.Event()

        def ask():
            result[0] = messagebox.askyesno("Confirm System Action", prompt)
            ev.set()

        self._safe_after(ask)
        ev.wait()
        return result[0]

    def _toggle_recording(self) -> None:
        if not self.is_recording:
            try:
                self.recorded_frames = []

                def callback(indata, _frames, _time_info, _status):
                    self.recorded_frames.append(indata.copy())

                self.audio_stream = sd.InputStream(
                    samplerate=16000, channels=1, dtype="float32", callback=callback
                )
                self.audio_stream.start()
                self.is_recording = True
                self.mic_btn.config(text="Stop", bg=RECORD_BG, fg=RECORD_FG)
                self._set_activity("Listening... Speak now, then click Stop.")
            except Exception as e:
                messagebox.showerror("Microphone Error", f"Could not access microphone:\n{e}")
        else:
            self.is_recording = False
            self.mic_btn.config(text="Speak", bg=BG_CARD, fg=TEXT_PRIMARY)
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass

            if not self.recorded_frames:
                self._set_activity("Ready — type a command or press Speak")
                return

            audio = np.concatenate(self.recorded_frames, axis=0).flatten()
            max_val = np.abs(audio).max()
            if max_val > 1e-4:
                audio = audio / max_val * 0.95

            self.mic_btn.config(state=tk.DISABLED)
            self._set_activity("Transcribing speech...")
            threading.Thread(target=self._process_voice, args=(audio,), daemon=True).start()

    def _process_voice(self, audio: np.ndarray) -> None:
        try:
            if self.stt is None:
                from inference import SpeechRecognizer
                self.stt = SpeechRecognizer(engine="whisper", model_size="base")

            res = self.stt.transcribe_with_metadata(audio)
            recognized = (res.get("text") or "").strip()
        except Exception as e:
            self._safe_after(lambda: self._append_chat("Jarvis", f"STT error: {e}"))
            self._safe_after(lambda: self.mic_btn.config(state=tk.NORMAL))
            self._set_activity("Ready — type a command or press Speak")
            return

        self._safe_after(lambda: self.mic_btn.config(state=tk.NORMAL))
        if not recognized:
            self._safe_after(lambda: self._append_chat("Jarvis", "I didn't hear anything. Please try again."))
            self._set_activity("Ready — type a command or press Speak")
            return

        self._safe_after(lambda: self._append_chat("You", recognized))
        self._set_activity("Thinking...")
        threading.Thread(target=self._process_command, args=(recognized,), daemon=True).start()


def main():
    root = tk.Tk()
    JarvisDesktopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
