"""
Native Windows Dynamic Island Floating Overlay for JARVIS (v2 — PIL Anti-Aliased).

Lightweight, frameless, always-on-top transparent tkinter overlay positioned at
the bottom-centre of the screen. Uses 2x supersampled PIL rendering for crisp,
anti-aliased rounded glass surfaces, soft shadows, and a miniature 3D glass orb
with flowing icy-cyan, pearl-white, and vibrant-orange ribbons.

States:
  IDLE, LISTENING, THINKING, EXECUTING, SPEAKING, COMPLETED, ERROR
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageTk


TRANSPARENT_KEY = "#010101"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class DynamicIslandOverlay:
    """Native always-on-top Dynamic Island pill driven by JarvisUIBridge."""

    def __init__(self, ui_bridge: Any):
        self.ui_bridge = ui_bridge
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._root: Optional[tk.Tk] = None

        # Larger, well-proportioned pill geometry
        self._cur_w = 256.0
        self._cur_h = 60.0
        self._expanded = False
        self._custom_pos: Optional[Tuple[int, int]] = None

        # Drag state
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_origin_pos: Optional[Tuple[int, int]] = None
        self._moved = False

        # Animation & PIL photo reference
        self._phase = 0.0
        self._last_tick = time.perf_counter()
        self._last_was_pressed = False
        self._visible = False
        self._tk_img: Optional[ImageTk.PhotoImage] = None

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
            width=460,
            height=340,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Double-Button-1>", self._on_double_click)

        root.withdraw()
        self._visible = False
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
            self._custom_pos = (self._screen_w // 2, self._screen_h - 52)
        self._drag_origin_pos = self._custom_pos

    def _on_drag(self, event) -> None:
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        if abs(dx) > 4 or abs(dy) > 4:
            self._moved = True
        if self._drag_origin_pos is not None:
            nx = max(140, min(self._screen_w - 140, self._drag_origin_pos[0] + dx))
            ny = max(90, min(self._screen_h - 12, self._drag_origin_pos[1] + dy))
            self._custom_pos = (nx, ny)

    def _on_release(self, event) -> None:
        if not self._moved:
            if self._expanded:
                items = self._canvas.find_withtag("full_ui_btn")
                if items:
                    bbox = self._canvas.bbox("full_ui_btn")
                    if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                        self._expanded = False
                        self.ui_bridge.set_island_visible(False)
                        self.ui_bridge.launch_frontend_window()
                        return
            self._expanded = not self._expanded

    def _on_double_click(self, _event) -> None:
        self._custom_pos = None

    def _check_click_outside(self, win_x: int, win_y: int, win_w: int, win_h: int) -> None:
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
            return 424.0, float(108 + n_rows * 26)

        if st == "LISTENING":
            return 340.0, 64.0
        if st == "THINKING":
            return 292.0, 62.0
        if st == "EXECUTING":
            return 410.0, 72.0
        if st == "SPEAKING":
            return 340.0, 64.0
        if st == "COMPLETED":
            return 272.0, 62.0
        if st == "ERROR":
            return 370.0, 66.0
        return 256.0, 60.0

    def _tick(self) -> None:
        if self._stop_event.is_set() or self._root is None:
            return

        now = time.perf_counter()
        dt = min(0.05, now - self._last_tick)
        self._last_tick = now

        snap = self.ui_bridge.snapshot()
        should_show = bool(snap.get("island_visible", False))
        if not should_show:
            if self._visible:
                self._root.withdraw()
                self._visible = False
                self._expanded = False
            self._root.after(100, self._tick)
            return

        if not self._visible:
            self._root.deiconify()
            self._visible = True

        st = (snap.get("state") or "IDLE").upper()
        mic = float(snap.get("mic_level") or 0.0)
        tts = float(snap.get("tts_level") or 0.0)

        speed_map = {
            "IDLE": 1.6,
            "LISTENING": 3.8 + mic * 4.2,
            "THINKING": 4.4,
            "EXECUTING": 4.0,
            "SPEAKING": 3.8 + tts * 4.6,
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

        if self._custom_pos is not None:
            cx, by = self._custom_pos
        else:
            cx = self._screen_w // 2
            by = self._screen_h - 54

        x = max(8, min(self._screen_w - w - 8, cx - w // 2))
        y = max(8, min(self._screen_h - h - 8, by - h))

        self._check_click_outside(x, y, w, h)

        self._root.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.config(width=w, height=h)
        self._render_pill(w, h, snap)

        self._root.after(33, self._tick)

    def _render_pill(self, w: int, h: int, snap: Dict[str, Any]) -> None:
        c = self._canvas
        c.delete("all")

        st = (snap.get("state") or "IDLE").upper()
        mic = float(snap.get("mic_level") or 0.0)
        tts = float(snap.get("tts_level") or 0.0)
        status_text = snap.get("status_text") or "Ready"
        task = snap.get("task") or {}
        steps: List[Dict[str, Any]] = task.get("steps") or []

        # 2x supersampled PIL background + mini 3D orb for ultra-crisp anti-aliased edges
        S = 2
        img = Image.new("RGBA", (w * S, h * S), (1, 1, 1, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        border_rgba_map = {
            "IDLE": (186, 214, 248, 255),
            "LISTENING": (96, 198, 248, 255),
            "THINKING": (178, 152, 248, 255),
            "EXECUTING": (118, 182, 252, 255),
            "SPEAKING": (251, 168, 96, 255),
            "COMPLETED": (110, 228, 168, 255),
            "ERROR": (248, 138, 138, 255),
        }
        border_col = border_rgba_map.get(st, (186, 214, 248, 255))

        pad = 3 * S
        rad = min(28 * S, (h * S) // 2 - pad)
        # Outer luminous rim + pearl-white glass body
        draw.rounded_rectangle(
            [pad, pad, w * S - pad, h * S - pad],
            radius=rad,
            fill=(246, 249, 254, 255),
            outline=border_col,
            width=2 * S,
        )
        # Inner top specular highlight
        draw.rounded_rectangle(
            [pad + 2 * S, pad + 2 * S, w * S - pad - 2 * S, h * S - pad - 2 * S],
            radius=max(4, rad - 2 * S),
            fill=None,
            outline=(255, 255, 255, 230),
            width=1 * S,
        )

        # Mini 3D glass orb drawn in 2x PIL
        orb_cx = (34 if not self._expanded else 32) * S
        orb_cy = (h // 2 if not self._expanded else 28) * S
        self._draw_mini_orb_pil(draw, orb_cx, orb_cy, S, st, mic, tts)

        # Downscale cleanly with LANCZOS and composite onto transparent key
        img_small = img.resize((w, h), Image.Resampling.LANCZOS)
        # Ensure exact transparent-key background pixels stay (1,1,1)
        bg = Image.new("RGB", (w, h), (1, 1, 1))
        bg.paste(img_small, (0, 0), img_small)
        self._tk_img = ImageTk.PhotoImage(bg)
        c.create_image(0, 0, anchor="nw", image=self._tk_img)

        if self._expanded:
            self._render_expanded_text(c, w, h, snap, st, task, steps)
            return

        # --- Compact Pill Foreground Text & Waveforms ---
        label = status_text
        if st == "EXECUTING" and steps:
            active_step = next((s for s in steps if s.get("status") == "active"), None)
            done_cnt = sum(1 for s in steps if s.get("status") == "done")
            if active_step:
                label = active_step.get("label") or status_text
            elif task.get("title"):
                label = task["title"]
            if len(label) > 30:
                label = label[:28] + "..."
            label = f"{label}  ({done_cnt}/{len(steps)})"
        elif st == "ERROR":
            err = snap.get("error_message") or status_text
            label = err[:34] + ("..." if len(err) > 34 else "")

        text_x = 64
        cy = h / 2.0
        c.create_text(
            text_x,
            cy - (6 if st == "EXECUTING" and steps else 0),
            text=label,
            anchor="w",
            font=("Segoe UI", 11, "bold" if st in ("EXECUTING", "LISTENING", "SPEAKING") else "normal"),
            fill="#141d2f",
        )

        if st in ("LISTENING", "SPEAKING"):
            level = mic if st == "LISTENING" else tts
            bar_color = "#0ea5e9" if st == "LISTENING" else "#f97316"
            bx0 = w - 76
            for i in range(7):
                amp = 4.0 + (level * 14.0 + 3.5) * (0.5 + 0.5 * math.sin(self._phase * 2.0 + i * 0.85))
                amp = max(3.5, min(18.0, amp))
                bx = bx0 + i * 8
                c.create_line(bx, cy - amp, bx, cy + amp, fill=bar_color, width=3.2, capstyle=tk.ROUND)

        elif st == "EXECUTING" and steps:
            done_cnt = sum(1 for s in steps if s.get("status") == "done")
            pct = max(0.12, done_cnt / max(1, len(steps)))
            bar_x1 = 64
            bar_x2 = w - 24
            bar_y = h - 16
            c.create_line(bar_x1, bar_y, bar_x2, bar_y, fill="#e2e8f0", width=4.0, capstyle=tk.ROUND)
            c.create_line(
                bar_x1,
                bar_y,
                bar_x1 + (bar_x2 - bar_x1) * pct,
                bar_y,
                fill="#3b82f6",
                width=4.0,
                capstyle=tk.ROUND,
            )

    def _render_expanded_text(
        self,
        c: tk.Canvas,
        w: int,
        h: int,
        snap: Dict[str, Any],
        st: str,
        task: Dict[str, Any],
        steps: List[Dict[str, Any]],
    ) -> None:
        c.create_text(
            60,
            28,
            text=f"JARVIS  •  {st}",
            anchor="w",
            font=("Segoe UI", 9, "bold"),
            fill="#5a667d",
        )

        # Subtle "Full UI" pill button
        c.create_rectangle(w - 86, 15, w - 18, 39, fill="#eef4ff", outline="#c7d8f8", width=1, tags="full_ui_btn")
        c.create_text(
            w - 52,
            27,
            text="Full UI",
            font=("Segoe UI", 9, "bold"),
            fill="#2563eb",
            tags="full_ui_btn",
        )

        title = task.get("title") or snap.get("status_text") or "Ready for voice command"
        if len(title) > 46:
            title = title[:44] + "..."
        c.create_text(
            24,
            58,
            text=title,
            anchor="w",
            font=("Segoe UI", 11, "bold"),
            fill="#141d2f",
        )

        y = 86
        if steps:
            for s in steps[:6]:
                sst = s.get("status") or "pending"
                lbl = s.get("label") or ""
                if len(lbl) > 44:
                    lbl = lbl[:42] + "..."
                if sst == "done":
                    sym, sym_col, txt_col = "✓", "#10b981", "#1f513f"
                elif sst == "active":
                    sym, sym_col, txt_col = "●", "#0ea5e9", "#141d2f"
                elif sst == "error":
                    sym, sym_col, txt_col = "!", "#ef4444", "#b91c1c"
                else:
                    sym, sym_col, txt_col = "•", "#94a3b8", "#64748b"

                c.create_text(28, y, text=sym, font=("Segoe UI", 10, "bold"), fill=sym_col)
                c.create_text(46, y, text=lbl, anchor="w", font=("Segoe UI", 9), fill=txt_col)
                y += 25
        else:
            hint = "Listening to your voice..." if st == "LISTENING" else "Speak a command anytime."
            c.create_text(24, y, text=hint, anchor="w", font=("Segoe UI", 9), fill="#64748b")

    def _draw_mini_orb_pil(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        S: int,
        st: str,
        mic: float,
        tts: float,
    ) -> None:
        pulse = 0.0
        if st == "IDLE":
            pulse = math.sin(self._phase) * 1.2
        elif st == "LISTENING":
            pulse = 1.5 + mic * 3.8
        elif st == "SPEAKING":
            pulse = 1.5 + tts * 4.2
        elif st in ("THINKING", "EXECUTING"):
            pulse = 1.2 + math.sin(self._phase * 1.6) * 1.2

        r = int((18.0 + pulse * 0.55) * S)

        halo_map = {
            "IDLE": (186, 230, 253, 140),
            "LISTENING": (56, 189, 248, 175),
            "THINKING": (167, 139, 250, 175),
            "EXECUTING": (96, 165, 250, 175),
            "SPEAKING": (251, 146, 60, 175),
            "COMPLETED": (52, 211, 153, 165),
            "ERROR": (248, 113, 113, 165),
        }
        draw.ellipse([cx - r - 4 * S, cy - r - 4 * S, cx + r + 4 * S, cy + r + 4 * S], fill=halo_map.get(st, (186, 230, 253, 140)))

        # Sapphire glass core
        core_col = (8, 34, 72, 255)
        if st == "COMPLETED":
            core_col = (6, 95, 70, 255)
        elif st == "ERROR":
            core_col = (127, 29, 29, 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=core_col, outline=(224, 242, 254, 255), width=2 * S)

        # Sweeping internal ribbons (icy cyan + warm orange)
        ang = self._phase * (1.8 if st == "THINKING" else 0.95)
        dx = int(math.cos(ang) * (r * 0.25))
        dy = int(math.sin(ang) * (r * 0.22))

        draw.ellipse(
            [cx - int(r * 0.72) + dx, cy - int(r * 0.62) - dy, cx + int(r * 0.55) + dx, cy + int(r * 0.18) - dy],
            fill=(56, 189, 248, 210),
        )
        warm_col = (249, 115, 22, 230) if st not in ("COMPLETED", "ERROR") else ((52, 211, 153, 220) if st == "COMPLETED" else (248, 113, 113, 220))
        draw.ellipse(
            [cx - int(r * 0.48) - dx, cy - int(r * 0.08) + dy, cx + int(r * 0.75) - dx, cy + int(r * 0.68) + dy],
            fill=warm_col,
        )
        # Pearl white inner veil
        draw.ellipse(
            [cx - int(r * 0.52) + dx // 2, cy - int(r * 0.52), cx + int(r * 0.28) + dx // 2, cy - int(r * 0.02)],
            fill=(240, 249, 255, 200),
        )
        # Crisp glass specular highlight
        draw.ellipse(
            [cx - int(r * 0.58), cy - int(r * 0.65), cx - int(r * 0.05), cy - int(r * 0.22)],
            fill=(255, 255, 255, 235),
        )

        # Orbiting violet particle when THINKING
        if st == "THINKING":
            px = int(cx + math.cos(self._phase * 2.4) * (r + 2 * S))
            py = int(cy + math.sin(self._phase * 2.4) * (r + 2 * S))
            draw.ellipse([px - 3 * S, py - 3 * S, px + 3 * S, py + 3 * S], fill=(139, 92, 246, 255), outline=(255, 255, 255, 255), width=1 * S)
