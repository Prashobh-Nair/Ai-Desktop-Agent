"""
Native Windows Desktop GUI for the AI-Powered Autonomous Desktop Agent.

Simple, clean, light-themed student GUI built with Tkinter:
- High-DPI sharp rendering on Windows.
- Non-blocking execution: text reply appears immediately, and TTS speaks asynchronously.
- Automatic Kokoro-82M TTS with Windows SAPI5 fallback if RAM is tight.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import ctypes
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
STATUS_OK_BG = "#edf5ef"
STATUS_OK_FG = "#2c663b"
RECORD_BG = "#f6eaea"
RECORD_FG = "#9c3b3b"


def speak_with_fallback(text: str, tts_engine=None) -> None:
    """Speaks text using Kokoro-82M, falling back to Windows SAPI5 if memory is tight."""
    clean = (text or "").strip()
    if not clean:
        return

    if tts_engine is not None:
        try:
            tts_engine.speak(clean, blocking=True)
            return
        except Exception:
            pass

    safe_text = clean.replace("'", "''")
    ps_cmd = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; $s.Speak('{safe_text}')"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
        capture_output=True,
    )


def _bind_hover(widget: tk.Widget, normal_bg: str, hover_bg: str) -> None:
    widget.bind("<Enter>", lambda _e: widget.config(bg=hover_bg) if widget["state"] != tk.DISABLED else None)
    widget.bind("<Leave>", lambda _e: widget.config(bg=normal_bg) if widget["state"] != tk.DISABLED else None)


class JarvisDesktopGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-Powered Autonomous Desktop Agent — Jarvis")
        self.root.geometry("820x600")
        self.root.minsize(660, 480)
        self.root.configure(bg=BG_MAIN)

        self.brain: Optional[Brain] = None
        self.executor = ActionExecutor()
        self.stt = None
        self.tts = None
        self.tts_mode = "Loading..."

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
            text="Semester 5 Minor Project  •  STT, Brain, OS/Browser Automation & TTS",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        self.status_label = tk.Label(
            header,
            text="Initializing modules...",
            font=("Segoe UI", 9),
            bg=BG_SUBTLE,
            fg=TEXT_MUTED,
            padx=12,
            pady=5,
        )
        self.status_label.pack(side=tk.RIGHT, padx=16)

        # Main Chat Card
        card = tk.Frame(self.root, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 8))

        # Toolbar inside Chat Card
        toolbar = tk.Frame(card, bg=BG_CARD)
        toolbar.pack(fill=tk.X, padx=16, pady=(10, 6))

        tk.Label(
            toolbar,
            text="Assistant Console",
            font=("Segoe UI", 10, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(side=tk.LEFT)

        clear_btn = tk.Button(
            toolbar,
            text="Clear Chat",
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

        tk.Checkbutton(
            toolbar,
            text="Speak replies aloud",
            variable=self.speak_enabled,
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
            activebackground=BG_CARD,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 14))

        # Divider
        tk.Frame(card, bg=BORDER_COLOR, height=1).pack(fill=tk.X)

        # Chat History Area
        self.chat_box = scrolledtext.ScrolledText(
            card,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=16,
            pady=12,
            spacing1=2,
            spacing3=2,
            state=tk.DISABLED,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)
        self.chat_box.tag_configure("user", foreground="#2b4c70", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("jarvis", foreground="#2c5e3b", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("meta", foreground="#7a828c", font=("Segoe UI", 8))
        self.chat_box.tag_configure("action", foreground="#3b5b7d", font=("Segoe UI", 9))

        # Divider above input
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

        # Clean Footer with Team Credits
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
            text="100% Local Execution",
            font=("Segoe UI", 8),
            bg=BG_MAIN,
            fg=TEXT_MUTED,
        ).pack(side=tk.RIGHT)

        self._append_chat(
            "Jarvis",
            "Hello! Type any question or desktop command below, or click Speak to talk to me.",
        )

    def _start_background_workers(self) -> None:
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._init_models, daemon=True).start()

    def _safe_after(self, fn) -> None:
        try:
            self.root.after(0, fn)
        except (tk.TclError, RuntimeError):
            pass

    def _init_models(self) -> None:
        try:
            from tts_module.src.tts_engine import get_tts_engine
            self.tts = get_tts_engine()
            self.tts_mode = "Kokoro-82M"
        except Exception:
            self.tts = None
            self.tts_mode = "Windows SAPI5"

        try:
            self.brain = Brain(mode="auto")
            brain_desc = f"{self.brain.mode} ({'LLM' if self.brain.ollama_available else 'Rules'})"
        except Exception:
            brain_desc = "Error"

        self._safe_after(
            lambda: self.status_label.config(
                text=f"Brain: {brain_desc}  |  TTS: {self.tts_mode}",
                bg=STATUS_OK_BG,
                fg=STATUS_OK_FG,
            )
        )

    def _tts_worker(self) -> None:
        while True:
            text = self.speech_queue.get()
            if text:
                speak_with_fallback(text, self.tts)
            self.speech_queue.task_done()

    def _enqueue_speech(self, text: str) -> None:
        if self.speak_enabled.get() and text:
            self.speech_queue.put(text)

    def _append_chat(self, sender: str, text: str, actions=None, meta: str = "") -> None:
        self.chat_box.config(state=tk.NORMAL)
        tag = "user" if sender == "You" else "jarvis"
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
                return

            audio = np.concatenate(self.recorded_frames, axis=0).flatten()
            max_val = np.abs(audio).max()
            if max_val > 1e-4:
                audio = audio / max_val * 0.95

            self.mic_btn.config(state=tk.DISABLED)
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
            return

        self._safe_after(lambda: self.mic_btn.config(state=tk.NORMAL))
        if not recognized:
            self._safe_after(lambda: self._append_chat("Jarvis", "I didn't hear anything. Please try again."))
            return

        self._safe_after(lambda: self._append_chat("You", recognized))
        threading.Thread(target=self._process_command, args=(recognized,), daemon=True).start()


def main():
    root = tk.Tk()
    JarvisDesktopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
