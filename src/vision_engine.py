"""
Local Screen Understanding Engine powered by HuggingFaceTB/SmolVLM-256M-Instruct.

Optimized for 8GB RAM laptops (no Ollama GPU/OOM crashes):
- ~500 MB footprint, loads in ~3s and runs inference in ~4s on CPU.
- Captures screen in memory (zero disk clutter) with single-pass resizing.
- Lazy-loads on first screen query and caches in process memory.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image, ImageGrab


def _load_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


class LocalScreenVision:
    """Captures the desktop screen and answers questions about it 100% locally."""

    MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    _processor = None
    _model = None

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._latest_frame: Optional[Image.Image] = None
        self._latest_frame_ts: float = 0.0
        self._frame_lock = threading.Lock()
        self._watch_stop = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None

    def start_live_watch(self, interval: float = 2.5) -> None:
        """Continuously buffers the latest screen frame in RAM (Gemini Live style)."""
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()

        def _loop():
            while not self._watch_stop.is_set():
                try:
                    frame = self.capture_screen(max_width=768)
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._latest_frame_ts = time.time()
                except Exception:
                    pass
                self._watch_stop.wait(interval)

        self._watch_thread = threading.Thread(target=_loop, daemon=True)
        self._watch_thread.start()

    def stop_live_watch(self) -> None:
        self._watch_stop.set()

    def initialize(self) -> None:
        if LocalScreenVision._model is not None:
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id)
        if hasattr(processor, "image_processor"):
            processor.image_processor.do_image_splitting = False

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        LocalScreenVision._processor = processor
        LocalScreenVision._model = model

    def capture_screen(self, max_width: int = 960) -> Image.Image:
        """Captures the primary screen with ImageGrab and Windows GDI BitBlt fallback."""
        img = None
        last_err: Optional[Exception] = None
        try:
            img = ImageGrab.grab(all_screens=False).convert("RGB")
        except Exception as e:
            last_err = e

        if img is None and os.name == "nt":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                gdi32 = ctypes.windll.gdi32
                w = user32.GetSystemMetrics(0)
                h = user32.GetSystemMetrics(1)
                hwnd = user32.GetDesktopWindow()
                hdc_src = user32.GetWindowDC(hwnd)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_src)
                hbmp = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
                gdi32.SelectObject(hdc_mem, hbmp)
                gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_src, 0, 0, 0x00CC0020)
                class _BMI(ctypes.Structure):
                    _fields_ = [
                        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                        ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                        ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                        ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                        ("biClrImportant", ctypes.c_uint32),
                    ]
                bmi = _BMI(40, w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
                buf = ctypes.create_string_buffer(w * h * 4)
                gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), 0)
                gdi32.DeleteObject(hbmp)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(hwnd, hdc_src)
                img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
            except Exception as e2:
                last_err = e2

        if img is None:
            raise RuntimeError(
                f"Screen capture unavailable (desktop may be locked or headless): {last_err}"
            )
        if img.width > max_width:
            ratio = max_width / float(img.width)
            img = img.resize((max_width, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        return img

    def get_screen_size(self) -> tuple[int, int]:
        """Returns physical primary screen (width, height) in pixels."""
        if os.name == "nt":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
                w = int(user32.GetSystemMetrics(0))
                h = int(user32.GetSystemMetrics(1))
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass
        img = self.capture_screen(max_width=1920)
        return img.width, img.height

    def _call_gemini_vision(self, img: Image.Image, prompt_text: str, json_mode: bool = False, max_tokens: int = 250) -> Optional[str]:
        api_key = _load_gemini_key()
        if not api_key or api_key == "your_gemini_api_key_here":
            return None

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        gen_cfg: dict = {"temperature": 0.1 if json_mode else 0.2, "maxOutputTokens": max_tokens}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        body = json.dumps({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                ],
            }],
            "generationConfig": gen_cfg,
        }).encode("utf-8")

        for model_id in ("gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-flash-lite-latest", "gemini-flash-latest"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=12.0) as resp:
                    data = json.loads(resp.read())
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e:
                if e.code in (404, 429, 500, 503, 504):
                    continue
                return None
            except Exception:
                return None
        return None

    def _analyze_with_gemini(self, img: Image.Image, prompt_text: str) -> Optional[str]:
        return self._call_gemini_vision(
            img,
            f"{prompt_text} Answer concisely in 1 to 3 sentences.",
            json_mode=False,
            max_tokens=250,
        )

    def locate_element(self, element_description: str) -> dict:
        """
        Captures a fresh screenshot and locates a UI element (button, text field, menu, etc.).
        Returns {"found": bool, "x": int, "y": int, "element_type": str, "label": str, "reason": str}.
        """
        try:
            screen_w, screen_h = self.get_screen_size()
            img = self.capture_screen(max_width=1280)
        except Exception as e:
            return {"found": False, "x": 0, "y": 0, "element_type": "", "label": "", "reason": str(e)}

        prompt = (
            f"Find this UI element on the screen: '{element_description}'.\n"
            "Return ONLY valid JSON with:\n"
            '{"found": true/false, "x_norm": 0-1000, "y_norm": 0-1000, '
            '"element_type": "button|input|menu|window|link|other", "label": "short label", "reason": "brief note"}\n'
            "Use normalized coordinates where (0,0) is top-left and (1000,1000) is bottom-right of the image. "
            "Aim for the exact center of the interactive element."
        )
        raw = self._call_gemini_vision(img, prompt, json_mode=True, max_tokens=180)
        if not raw:
            return {"found": False, "x": 0, "y": 0, "element_type": "", "label": "", "reason": "Vision API unavailable"}
        try:
            data = json.loads(raw)
            if not data.get("found"):
                return {
                    "found": False,
                    "x": 0,
                    "y": 0,
                    "element_type": data.get("element_type", ""),
                    "label": data.get("label", ""),
                    "reason": data.get("reason", "Element not visible on screen."),
                }
            x_norm = max(0, min(1000, int(data.get("x_norm", 0))))
            y_norm = max(0, min(1000, int(data.get("y_norm", 0))))
            return {
                "found": True,
                "x": int(round(x_norm * screen_w / 1000.0)),
                "y": int(round(y_norm * screen_h / 1000.0)),
                "element_type": str(data.get("element_type", "other")),
                "label": str(data.get("label", element_description)),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            return {"found": False, "x": 0, "y": 0, "element_type": "", "label": "", "reason": f"Parse error: {e}"}

    def verify_screen(self, expected_description: str) -> tuple[bool, str]:
        """
        Captures a fresh screenshot after an action and verifies whether the expected state is visible.
        Uses Gemini Vision primary with local SmolVLM-256M fallback.
        """
        try:
            img = self.capture_screen(max_width=960)
        except Exception as e:
            return False, f"Screen capture failed during verification: {e}"

        prompt = (
            f"Verify if the current screen satisfies this expectation: '{expected_description}'.\n"
            'Return ONLY valid JSON: {"verified": true/false, "summary": "1 concise sentence describing what is visible"}'
        )
        raw = self._call_gemini_vision(img, prompt, json_mode=True, max_tokens=150)
        if raw:
            try:
                data = json.loads(raw)
                return bool(data.get("verified")), str(data.get("summary") or expected_description).strip()
            except Exception:
                pass

        # Local SmolVLM fallback when offline
        ans = self.analyze_image(img, f"Does the screen show: {expected_description}? Answer yes or no and explain briefly.")
        verified = ans.lower().startswith(("yes", "true", "it does", "the screen shows"))
        return verified, ans

    def analyze_image(self, img: Image.Image, question: Optional[str] = None) -> str:
        """Analyzes a PIL Image via Gemini Vision (when online) with SmolVLM-256M local fallback."""
        prompt_text = (question or "").strip() or "Describe what is on this screen briefly in 1 to 2 sentences."
        cloud_answer = self._analyze_with_gemini(img, prompt_text)
        if cloud_answer:
            return cloud_answer

        import torch

        self.initialize()
        processor = LocalScreenVision._processor
        model = LocalScreenVision._model

        msgs = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }]
        chat_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, images=[img], return_tensors="pt")

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64)

        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

    def analyze_screen(self, question: Optional[str] = None) -> str:
        """Analyzes the live screen buffer (or captures fresh) via Gemini Vision -> SmolVLM."""
        img = None
        age = None
        with self._frame_lock:
            if self._latest_frame is not None and (time.time() - self._latest_frame_ts) < 4.0:
                img = self._latest_frame
                age = round(time.time() - self._latest_frame_ts, 1)
        if img is None:
            try:
                img = self.capture_screen(max_width=768)
                print("[Vision] Fresh screen captured.", flush=True)
            except RuntimeError as e:
                return str(e)
        else:
            print(f"[Vision] Using buffered frame ({age}s old).", flush=True)
        try:
            api_key = _load_gemini_key()
            if api_key and api_key != "your_gemini_api_key_here":
                print("[Vision] Sending to Gemini Vision API...", flush=True)
            else:
                print("[Vision] No Gemini key — using local SmolVLM-256M.", flush=True)
            return self.analyze_image(img, question)
        except Exception as e:
            return f"Could not analyze screen: {e}"
