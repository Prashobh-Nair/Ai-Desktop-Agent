"""
Native Windows Desktop GUI for the AI-Powered Autonomous Desktop Agent.

Simple, light-themed student GUI built with Tkinter:
- Runs natively on Windows so OS & Browser automation work reliably.
- Non-blocking execution: text reply appears immediately, and TTS speaks asynchronously.
- Automatic Kokoro-82M TTS with Windows SAPI5 fallback if RAM is low.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

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


# Light, simple student-project color palette (no dark theme, no bright colors)
BG_MAIN = "#f4f6f8"
BG_CARD = "#ffffff"
BG_INPUT = "#fafbfc"
BORDER_COLOR = "#d0d7de"
TEXT_PRIMARY = "#24292f"
TEXT_MUTED = "#57606a"
ACCENT = "#3b5b7d"
ACCENT_TEXT = "#ffffff"
BTN_BG = "#eef2f6"
RECORD_BG = "#f6eaea"
RECORD_FG = "#a94442"


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

    # Fallback: built-in Windows SAPI5 speech (fast, zero extra RAM)
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


class JarvisDesktopGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-Powered Autonomous Desktop Agent — Jarvis")
        self.root.geometry("920x620")
        self.root.minsize(760, 520)
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
        # Top Header
        header = tk.Frame(self.root, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header.pack(fill=tk.X, padx=14, pady=(12, 8))

        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT, padx=14, pady=10)
        tk.Label(
            title_frame,
            text="AI-Powered Autonomous Desktop Agent",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            title_frame,
            text="Semester 5 Minor Project — STT + Brain + OS/Browser Automation + TTS",
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w")

        self.status_label = tk.Label(
            header,
            text="Status: Initializing modules...",
            font=("Segoe UI", 9),
            bg=BTN_BG,
            fg=TEXT_MUTED,
            padx=10,
            pady=4,
        )
        self.status_label.pack(side=tk.RIGHT, padx=14)

        # Body: Left Chat + Right Controls
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))

        # Left Panel (Chat)
        left = tk.Frame(body, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        chat_top = tk.Frame(left, bg=BG_CARD)
        chat_top.pack(fill=tk.X, padx=12, pady=(8, 4))
        tk.Label(
            chat_top, text="Assistant Console", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_PRIMARY
        ).pack(side=tk.LEFT)
        tk.Button(
            chat_top,
            text="Clear Chat",
            font=("Segoe UI", 8),
            bg=BTN_BG,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=self._clear_chat,
        ).pack(side=tk.RIGHT)

        self.chat_box = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=12,
            pady=8,
            state=tk.DISABLED,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.chat_box.tag_configure("user", foreground="#1f4e79", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("jarvis", foreground="#2e5939", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("meta", foreground=TEXT_MUTED, font=("Segoe UI", 8))
        self.chat_box.tag_configure("action", foreground=ACCENT, font=("Segoe UI", 9))

        # Bottom Input Bar
        input_bar = tk.Frame(left, bg=BG_INPUT, highlightbackground=BORDER_COLOR, highlightthickness=1)
        input_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(
            input_bar, font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_PRIMARY, relief=tk.SOLID, bd=1
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 6), pady=10, ipady=4)
        self.entry.bind("<Return>", lambda _e: self._on_send())

        self.mic_btn = tk.Button(
            input_bar,
            text="Speak",
            font=("Segoe UI", 9, "bold"),
            bg=BTN_BG,
            fg=TEXT_PRIMARY,
            relief=tk.SOLID,
            bd=1,
            padx=12,
            pady=4,
            command=self._toggle_recording,
        )
        self.mic_btn.pack(side=tk.LEFT, padx=(0, 6), pady=10)

        self.send_btn = tk.Button(
            input_bar,
            text="Send",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT,
            fg=ACCENT_TEXT,
            relief=tk.FLAT,
            padx=14,
            pady=5,
            command=self._on_send,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

        # Right Panel (Sidebar)
        right = tk.Frame(body, bg=BG_MAIN, width=250)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # Settings card
        settings_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        settings_card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            settings_card, text="Settings", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_PRIMARY
        ).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Checkbutton(
            settings_card,
            text="Speak replies aloud",
            variable=self.speak_enabled,
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            activebackground=BG_CARD,
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Sample Commands card
        samples_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        samples_card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            samples_card,
            text="Quick Test Commands",
            font=("Segoe UI", 10, "bold"),
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        samples = [
            ("Open Chrome", "open chrome"),
            ("Open New Tab", "open new tab"),
            ("Close Tab", "close tab"),
            ("Search Google: Poco X6 Neo", "search in google about poco x6 neo"),
            ("Play Believer on YouTube", "play believer on youtube"),
            ("What is the time?", "what is the time"),
            ("Explain how a plane flies", "Explain how a plane flies."),
            ("Tell me a joke", "Tell me a funny joke."),
        ]
        for label, cmd in samples:
            tk.Button(
                samples_card,
                text=label,
                anchor="w",
                font=("Segoe UI", 8),
                bg="#f9fafb",
                fg=TEXT_PRIMARY,
                relief=tk.SOLID,
                bd=1,
                padx=8,
                pady=4,
                command=lambda c=cmd: self._submit_text(c),
            ).pack(fill=tk.X, padx=10, pady=2)
        tk.Frame(samples_card, bg=BG_CARD, height=8).pack()

        # Team Credits card
        team_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        team_card.pack(fill=tk.X)
        tk.Label(
            team_card, text="Team Modules", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_PRIMARY
        ).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Label(
            team_card,
            text="Module 1 (STT): Prashobh Nair\nModule 2 (Brain): Sahaj Patel\nModule 3 (TTS): Zeel Vasoya\nModule 4 (OS / UI): Mayank",
            justify=tk.LEFT,
            font=("Segoe UI", 8),
            bg=BG_CARD,
            fg=TEXT_MUTED,
        ).pack(anchor="w", padx=12, pady=(0, 10))

        self._append_chat(
            "Jarvis",
            "Hello! Type a command, click a Quick Test button, or click Speak to use your microphone.",
        )

    def _start_background_workers(self) -> None:
        # Dedicated speaker thread so TTS never blocks the GUI or command execution
        threading.Thread(target=self._tts_worker, daemon=True).start()
        # Background model initializer (Kokoro first before Ollama reserves RAM)
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
                text=f"Brain: {brain_desc}  |  TTS: {self.tts_mode}", fg="#2e6f40"
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
                # Ask confirmation safely on GUI main thread
                approved = self._ask_confirmation_sync(reply)
                if approved:
                    actions_log = self.executor.execute_plan(plan)
                    reply = "Confirmed. Executing now."
                else:
                    actions_log = ["Cancelled by user."]
                    reply = "Action cancelled."
            else:
                actions_log = self.executor.execute_plan(plan)

        meta = f"Intent: {intent} | Engine: {backend} | {latency_ms} ms"
        self.root.after(0, lambda: self._append_chat("Jarvis", reply, actions_log, meta))
        self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
        self._enqueue_speech(reply)

    def _ask_confirmation_sync(self, prompt: str) -> bool:
        result = [False]
        ev = threading.Event()

        def ask():
            result[0] = messagebox.askyesno("Confirm System Action", prompt)
            ev.set()

        self.root.after(0, ask)
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
            self.mic_btn.config(text="Speak", bg=BTN_BG, fg=TEXT_PRIMARY)
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
            self.root.after(0, lambda: self._append_chat("Jarvis", f"STT error: {e}"))
            self.root.after(0, lambda: self.mic_btn.config(state=tk.NORMAL))
            return

        self.root.after(0, lambda: self.mic_btn.config(state=tk.NORMAL))
        if not recognized:
            self.root.after(0, lambda: self._append_chat("Jarvis", "I didn't hear anything. Please try again."))
            return

        self.root.after(0, lambda: self._append_chat("You", recognized))
        threading.Thread(target=self._process_command, args=(recognized,), daemon=True).start()


def main():
    root = tk.Tk()
    JarvisDesktopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
