"""
Web Frontend Server for the AI-Powered Autonomous Desktop Agent.
Connects Module 1 (Whisper STT), Module 2 (Jarvis Brain), Module 3 (Kokoro TTS),
and Module 4 (OS & Browser ActionExecutor) to a simple, light-themed student UI.
"""

from __future__ import annotations

import os
import sys

# Prevent OpenBLAS/MKL thread-pool over-allocation on 8GB RAM / multi-core Windows laptops
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SpeechRecognizer
from src.executor import ActionExecutor
from brain_module.jarvis.brain import Brain
from tts_module.src.tts_engine import generate_speech
from tts_module.src.config import AUDIO_OUTPUT_DIR

app = FastAPI(title="AI Desktop Agent - Student UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded shared instances
_stt: Optional[SpeechRecognizer] = None
_brain: Optional[Brain] = None
_executor: Optional[ActionExecutor] = None
_pending_confirmations: Dict[str, Dict] = {}


def get_brain_and_executor():
    global _brain, _executor
    if _brain is None:
        _brain = Brain(mode="auto")
    if _executor is None:
        _executor = ActionExecutor()
    return _brain, _executor


def get_stt():
    global _stt
    if _stt is None:
        _stt = SpeechRecognizer(engine="whisper", model_size="base")
    return _stt


class ChatRequest(BaseModel):
    text: str
    speak_reply: bool = True
    session_id: str = "web_ui"


class ConfirmRequest(BaseModel):
    confirm_id: str
    approved: bool
    speak_reply: bool = True


def _run_brain_and_execute(text: str, speak_reply: bool, session_id: str) -> Dict:
    brain, executor = get_brain_and_executor()
    t0 = time.perf_counter()
    brain_out = brain.plan(text, session_id=session_id)
    plan = brain_out.get("plan", {})
    reply = plan.get("reply") or "Done."
    intent = plan.get("intent", "unknown")
    backend = brain_out.get("backend", "rules")
    needs_confirmation = bool(plan.get("needs_confirmation"))

    actions_log = []
    confirm_id = None

    if plan.get("steps"):
        if needs_confirmation:
            confirm_id = f"cfm_{int(time.time() * 1000)}"
            _pending_confirmations[confirm_id] = plan
        else:
            actions_log = executor.execute_plan(plan)

    audio_url = None
    if speak_reply and reply:
        try:
            wav_path = generate_speech(reply)
            audio_url = f"/api/audio/{Path(wav_path).name}"
        except Exception:
            audio_url = None

    return {
        "user_text": text,
        "reply": reply,
        "intent": intent,
        "backend": backend,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "actions": actions_log,
        "needs_confirmation": needs_confirmation,
        "confirm_id": confirm_id,
        "audio_url": audio_url,
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_path = PROJECT_ROOT / "frontend" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    brain, _ = get_brain_and_executor()
    return {
        "stt": "Whisper-Base (74M)",
        "brain_mode": brain.mode,
        "ollama_active": brain.ollama_available,
        "tts": "Kokoro-82M",
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")
    return _run_brain_and_execute(text, req.speak_reply, req.session_id)


@app.post("/api/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    speak_reply: bool = Form(True),
    session_id: str = Form("web_ui"),
):
    stt = get_stt()
    suffix = Path(audio.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        stt_res = stt.transcribe_with_metadata(tmp_path)
        recognized = (stt_res.get("text") or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not recognized:
        return {
            "user_text": "",
            "reply": "I couldn't hear any speech. Please try speaking again.",
            "intent": "unknown",
            "backend": "stt",
            "latency_ms": 0,
            "actions": [],
            "needs_confirmation": False,
            "confirm_id": None,
            "audio_url": None,
        }

    return _run_brain_and_execute(recognized, speak_reply, session_id)


@app.post("/api/confirm")
def confirm_endpoint(req: ConfirmRequest):
    _, executor = get_brain_and_executor()
    plan = _pending_confirmations.pop(req.confirm_id, None)
    if not plan:
        raise HTTPException(status_code=404, detail="Confirmation expired or not found")

    if req.approved:
        actions = executor.execute_plan(plan)
        reply = "Confirmed. Executing the command now."
    else:
        actions = ["Cancelled by user."]
        reply = "Action cancelled."

    audio_url = None
    if req.speak_reply:
        try:
            wav_path = generate_speech(reply)
            audio_url = f"/api/audio/{Path(wav_path).name}"
        except Exception:
            pass

    return {"reply": reply, "actions": actions, "audio_url": audio_url}


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    safe_name = Path(filename).name
    file_path = Path(AUDIO_OUTPUT_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(file_path), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  AI Desktop Agent - Student Web UI")
    print("  Open in browser: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
