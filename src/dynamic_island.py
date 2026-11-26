"""
Native Windows Dynamic Island Floating Overlay for JARVIS.

Lightweight, frameless, always-on-top transparent tkinter overlay positioned at
the bottom-centre of the screen. Reacts in real time to JarvisUIBridge states:
  IDLE, LISTENING, THINKING, EXECUTING, SPEAKING, COMPLETED, ERROR

Features:
  - True pixel-clean rounded pill via Windows -transparentcolor
  - Does not steal focus from active apps (WS_EX_TOOLWINDOW)
  - Smoothly animated width/height, mini 3D orb, live mic/TTS waveforms,
    thinking orbit particles, and task progress bar
  - Click to expand full task details, click outside to collapse, drag to move
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple


TRANSPARENT_KEY = "#010101"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _draw_rounded_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    **kwargs,
) -> int:
    """Draws a smooth rounded rectangle on a tk.Canvas using a polygon."""
    r = max(2.0, min(r, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, splinesteps=24, **kwargs)


class DynamicIslandOverlay:
    """Native always-on-top Dynamic Island pill driven by JarvisUIBridge."""

    def __init__(self, ui_bridge: Any):
        self.ui_bridge = ui_bridge
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._root: Optional[tk.Tk] = None

        # Current & target pill geometry
        self._cur_w = 204.0
        self._cur_h = 46.0
        self._expanded = False
        self._custom_pos: Optional[Tuple[int, int]] = None  # (center_x, bottom_y)

        # Drag state
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_origin_pos: Optional[Tuple[int, int]] = None
        self._moved = False

        # Animation phase
        self._phase = 0.0
        self._last_tick = time.perf_counter()
        self._last_was_pressed = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def _run_tk(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("JARVIS Island")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=TRANSPARENT_KEY)
        try:
            root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except Exception:
            pass

        # Prevent taskbar icon & focus stealing
        root.update_idletasks()
        try:
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        except Exception:
            pass

        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()

        canvas = tk.Canvas(
            root,
            width=400,
            height=300,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

        # Mouse bindings for click-to-expand and drag-to-move
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Double-Button-1>", self._on_double_click)

        self._tick()
        try:
            root.mainloop()
        except Exception:
            pass

    def _on_press(self, event) -> None:
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self._moved = False
        if self._custom_pos is None:
            cx = self._screen_w // 2
            by = self._screen_h - 46
            self._custom_pos = (cx, by)
        self._drag_origin_pos = self._custom_pos

    def _on_drag(self, event) -> None:
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        if abs(dx) > 4 or abs(dy) > 4:
            self._moved = True
        if self._drag_origin_pos is not None:
            nx = max(120, min(self._screen_w - 120, self._drag_origin_pos[0] + dx))
            ny = max(80, min(self._screen_h - 12, self._drag_origin_pos[1] + dy))
            self._custom_pos = (nx, ny)

    def _on_release(self, event) -> None:
        if not self._moved:
            # Check if user clicked the "Full UI" button inside expanded mode
            if self._expanded:
                items = self._canvas.find_withtag("full_ui_btn")
                if items:
                    bbox = self._canvas.bbox("full_ui_btn")
                    if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                        self._expanded = False
                        self.ui_bridge.launch_frontend_window()
                        return
            self._expanded = not self._expanded

    def _on_double_click(self, _event) -> None:
        # Reset position back to bottom-centre
        self._custom_pos = None

    def _check_click_outside(self, win_x: int, win_y: int, win_w: int, win_h: int) -> None:
        """Collapses expanded pill when the user clicks anywhere outside its bounds."""
        if not self._expanded:
            return
        try:
            VK_LBUTTON = 0x01
            pressed = bool(ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
            if pressed and not self._last_was_pressed:
                class _PT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = _PT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                inside = (win_x <= pt.x <= win_x + win_w) and (win_y <= pt.y <= win_y + win_h)
                if not inside:
                    self._expanded = False
            self._last_was_pressed = pressed
        except Exception:
            pass

    def _target_geometry(self, snap: Dict[str, Any]) -> Tuple[float, float]:
        st = (snap.get("state") or "IDLE").upper()
        task = snap.get("task") or {}
        steps = task.get("steps") or []

        if self._expanded:
            n_rows = max(1, min(6, len(steps))) if steps else 1
            return 368.0, float(92 + n_rows * 24)

        if st == "LISTENING":
            return 282.0, 50.0
        if st == "THINKING":
            return 244.0, 48.0
        if st == "EXECUTING":
            return 350.0, 56.0
        if st == "SPEAKING":
            return 282.0, 50.0
        if st == "COMPLETED":
            return 222.0, 48.0
        if st == "ERROR":
            return 320.0, 52.0
        return 198.0, 46.0

    def _tick(self) -> None:
        if self._stop_event.is_set() or self._root is None:
            return

        now = time.perf_counter()
        dt = min(0.05, now - self._last_tick)
        self._last_tick = now

        snap = self.ui_bridge.snapshot()
        st = (snap.get("state") or "IDLE").upper()
        mic = float(snap.get("mic_level") or 0.0)
        tts = float(snap.get("tts_level") or 0.0)
        task = snap.get("task") or {}

        speed_map = {
            "IDLE": 1.6,
            "LISTENING": 3.8 + mic * 4.0,
            "THINKING": 4.4,
            "EXECUTING": 4.0,
            "SPEAKING": 3.8 + tts * 4.5,
            "COMPLETED": 2.0,
            "ERROR": 2.0,
        }
        self._phase += dt * speed_map.get(st, 1.8)

        tw, th = self._target_geometry(snap)
        k = min(1.0, dt * 11.0)
        self._cur_w = _lerp(self._cur_w, tw, k)
        self._cur_h = _lerp(self._cur_h, th, k)

        w = int(round(self._cur_w))
        h = int(round(self._cur_h))

        # Position at bottom-centre (or custom dragged position)
        if self._custom_pos is not None:
            cx, by = self._custom_pos
        else:
            cx = self._screen_w // 2
            by = self._screen_h - 52

        x = max(8, min(self._screen_w - w - 8, cx - w // 2))
        y = max(8, min(self._screen_h - h - 8, by - h))

        self._check_click_outside(x, y, w, h)

        self._root.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.config(width=w, height=h)
        self._render_pill(w, h, snap)

        self._root.after(33, self._tick)  # ~30 fps

    def _render_pill(self, w: int, h: int, snap: Dict[str, Any]) -> None:
        c = self._canvas
        c.delete("all")

        st = (snap.get("state") or "IDLE").upper()
        mic = float(snap.get("mic_level") or 0.0)
        tts = float(snap.get("tts_level") or 0.0)
        status_text = snap.get("status_text") or "Ready"
        task = snap.get("task") or {}
        steps: List[Dict[str, Any]] = task.get("steps") or []

        # Border accent color per state (light theme + soft blue/purple/orange)
        accent_map = {
            "IDLE": "#cbdcf6",
            "LISTENING": "#7dd3fc",
            "THINKING": "#c4b5fd",
            "EXECUTING": "#93c5fd",
            "SPEAKING": "#fdba74",
            "COMPLETED": "#86efac",
            "ERROR": "#fca5a5",
        }
        border_col = accent_map.get(st, "#cbdcf6")
        radius = min(24.0, h / 2.0 - 1)

        # Outer translucent-style pill body (light pearl glass)
        _draw_rounded_rect(c, 2, 2, w - 2, h - 2, radius, fill="#f7f9fe", outline=border_col, width=1.6)
        # Subtle top inner highlight line
        c.create_line(radius, 3.5, w - radius, 3.5, fill="#ffffff", width=1.2)

        if self._expanded:
            self._render_expanded(c, w, h, snap, st, task, steps)
            return

        # --- Compact Pill Mode ---
        orb_cx = 25.0
        orb_cy = h / 2.0
        self._draw_mini_orb(c, orb_cx, orb_cy, st, mic, tts)

        # Activity label
        label = status_text
        if st == "EXECUTING" and steps:
            active_step = next((s for s in steps if s.get("status") == "active"), None)
            done_cnt = sum(1 for s in steps if s.get("status") == "done")
            if active_step:
                label = active_step.get("label") or status_text
            elif task.get("title"):
                label = task["title"]
            if len(label) > 28:
                label = label[:26] + "..."
            label = f"{label} ({done_cnt}/{len(steps)})"
        elif st == "ERROR":
            err = snap.get("error_message") or status_text
            label = err[:32] + ("..." if len(err) > 32 else "")

        text_x = 46
        c.create_text(
            text_x,
            orb_cy - (4 if st == "EXECUTING" and steps else 0),
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold" if st in ("EXECUTING", "LISTENING", "SPEAKING") else "normal"),
            fill="#162033",
        )

        # Right-side waveform for LISTENING / SPEAKING
        if st in ("LISTENING", "SPEAKING"):
            level = mic if st == "LISTENING" else tts
            bar_color = "#0ea5e9" if st == "LISTENING" else "#f97316"
            bx0 = w - 66
            for i in range(6):
                amp = 3.5 + (level * 12.0 + 3.0) * (0.5 + 0.5 * math.sin(self._phase * 1.9 + i * 0.9))
                amp = max(3.0, min(16.0, amp))
                bx = bx0 + i * 7
                c.create_line(bx, orb_cy - amp, bx, orb_cy + amp, fill=bar_color, width=3.0, capstyle=tk.ROUND)

        # Subtle progress bar inside bottom of EXECUTING pill
        elif st == "EXECUTING" and steps:
            done_cnt = sum(1 for s in steps if s.get("status") == "done")
            pct = max(0.12, done_cnt / max(1, len(steps)))
            bar_x1 = 46
            bar_x2 = w - 20
            bar_y = h - 12
            c.create_line(bar_x1, bar_y, bar_x2, bar_y, fill="#e2e8f0", width=3.5, capstyle=tk.ROUND)
            c.create_line(
                bar_x1,
                bar_y,
                bar_x1 + (bar_x2 - bar_x1) * pct,
                bar_y,
                fill="#3b82f6",
                width=3.5,
                capstyle=tk.ROUND,
            )

    def _render_expanded(
        self,
        c: tk.Canvas,
        w: int,
        h: int,
        snap: Dict[str, Any],
        st: str,
        task: Dict[str, Any],
        steps: List[Dict[str, Any]],
    ) -> None:
        # Header row
        self._draw_mini_orb(c, 26, 24, st, float(snap.get("mic_level") or 0.0), float(snap.get("tts_level") or 0.0))
        c.create_text(
            48,
            24,
            text=f"JARVIS  •  {st}",
            anchor="w",
            font=("Segoe UI", 9, "bold"),
            fill="#5a667d",
        )

        # Subtle "Full UI" button on top-right
        _draw_rounded_rect(c, w - 76, 13, w - 16, 35, 11, fill="#eef4ff", outline="#c7d8f8", width=1.0, tags="full_ui_btn")
        c.create_text(
            w - 46,
            24,
            text="Full UI",
            font=("Segoe UI", 8, "bold"),
            fill="#2563eb",
            tags="full_ui_btn",
        )

        # Task title or status
        title = task.get("title") or snap.get("status_text") or "Ready for voice command"
        if len(title) > 44:
            title = title[:42] + "..."
        c.create_text(
            20,
            52,
            text=title,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            fill="#141d2f",
        )

        # Step rows
        y = 78
        if steps:
            for s in steps[:6]:
                sst = s.get("status") or "pending"
                lbl = s.get("label") or ""
                if len(lbl) > 42:
                    lbl = lbl[:40] + "..."
                if sst == "done":
                    sym, sym_col, txt_col = "✓", "#10b981", "#2c4a3e"
                elif sst == "active":
                    sym, sym_col, txt_col = "●", "#0ea5e9", "#141d2f"
                elif sst == "error":
                    sym, sym_col, txt_col = "!", "#ef4444", "#b91c1c"
                else:
                    sym, sym_col, txt_col = "•", "#94a3b8", "#64748b"

                c.create_text(24, y, text=sym, font=("Segoe UI", 10, "bold"), fill=sym_col)
                c.create_text(40, y, text=lbl, anchor="w", font=("Segoe UI", 9), fill=txt_col)
                y += 23
        else:
            hint = "Speak naturally — I'm listening." if st == "LISTENING" else "Say a command anytime."
            c.create_text(20, y, text=hint, anchor="w", font=("Segoe UI", 9), fill="#64748b")

    def _draw_mini_orb(
        self,
        c: tk.Canvas,
        cx: float,
        cy: float,
        st: str,
        mic: float,
        tts: float,
    ) -> None:
        pulse = 0.0
        if st == "IDLE":
            pulse = math.sin(self._phase) * 1.1
        elif st == "LISTENING":
            pulse = 1.2 + mic * 3.2
        elif st == "SPEAKING":
            pulse = 1.2 + tts * 3.5
        elif st in ("THINKING", "EXECUTING"):
            pulse = 1.0 + math.sin(self._phase * 1.5) * 1.0

        r = 12.0 + pulse * 0.45

        # Outer soft halo
        halo_col = {
            "IDLE": "#bae6fd",
            "LISTENING": "#7dd3fc",
            "THINKING": "#ddd6fe",
            "EXECUTING": "#93c5fd",
            "SPEAKING": "#fed7aa",
            "COMPLETED": "#bbf7d0",
            "ERROR": "#fecaca",
        }.get(st, "#bae6fd")
        c.create_oval(cx - r - 2.5, cy - r - 2.5, cx + r + 2.5, cy + r + 2.5, fill=halo_col, outline="")

        # Sapphire optical sphere
        core_col = "#09254c" if st != "COMPLETED" else "#065f46"
        if st == "ERROR":
            core_col = "#7f1d1d"
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=core_col, outline="#e0f2fe", width=1.2)

        # Internal animated ribbons (icy cyan + warm orange)
        ang = self._phase * (1.8 if st == "THINKING" else 0.9)
        dx = math.cos(ang) * (r * 0.35)
        dy = math.sin(ang) * (r * 0.28)
        c.create_oval(
            cx - r * 0.65 + dx * 0.4,
            cy - r * 0.55 - dy * 0.3,
            cx + r * 0.45 + dx * 0.4,
            cy + r * 0.15 - dy * 0.3,
            fill="#38bdf8",
            outline="",
        )
        c.create_oval(
            cx - r * 0.35 - dx * 0.4,
            cy - r * 0.05 + dy * 0.3,
            cx + r * 0.68 - dx * 0.4,
            cy + r * 0.62 + dy * 0.3,
            fill="#fb923c" if st not in ("COMPLETED", "ERROR") else ("#34d399" if st == "COMPLETED" else "#f87171"),
            outline="",
        )

        # Specular glass highlight
        c.create_oval(cx - r * 0.58, cy - r * 0.62, cx - r * 0.05, cy - r * 0.22, fill="#ffffff", outline="")

        # Orbiting particle in THINKING state
        if st == "THINKING":
            px = cx + math.cos(self._phase * 2.4) * (r + 2.0)
            py = cy + math.sin(self._phase * 2.4) * (r + 2.0)
            c.create_oval(px - 2.2, py - 2.2, px + 2.2, py + 2.2, fill="#8b5cf6", outline="#ffffff", width=0.6)
