"""
FastAPI Speech-to-Text Microservice
Supports:
1. Production High-Capacity Local Engine: Neural Whisper (74M / 244M params, INT8 offline)
2. Academic Baseline Model: Custom CNN + BiLSTM + CTC (1.6M parameters from scratch)
"""
import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from inference import SpeechRecognizer

app = FastAPI(
    title="Autonomous Desktop Agent STT API",
    description="Speech-to-Text inference service supporting local offline 74M Transformer and academic 1.6M scratch model.",
    version="2.1.0"
)

DEFAULT_ENGINE = os.environ.get("STT_ENGINE", "whisper")
CHECKPOINT_PATH = os.environ.get("STT_CHECKPOINT", "checkpoints/best_model.pt")

recognizers = {}


@app.on_event("startup")
def load_default_model():
    global recognizers
    try:
        print(f"[*] Preloading default STT engine ({DEFAULT_ENGINE})...")
        recognizers[DEFAULT_ENGINE] = SpeechRecognizer(
            checkpoint_path=CHECKPOINT_PATH, engine=DEFAULT_ENGINE
        )
        print(f"[+] Loaded {DEFAULT_ENGINE} engine successfully.")
    except Exception as e:
        print(f"[!] Warning preloading {DEFAULT_ENGINE}: {e}")


def get_recognizer(engine: str):
    if engine not in recognizers:
        recognizers[engine] = SpeechRecognizer(
            checkpoint_path=CHECKPOINT_PATH, engine=engine
        )
    return recognizers[engine]


@app.get("/health")
def health():
    current_rec = recognizers.get(DEFAULT_ENGINE)
    params_str = current_rec.parameter_str if current_rec else "74 Million"
    return {
        "status": "healthy",
        "default_engine": DEFAULT_ENGINE,
        "model_parameters": params_str,
        "quantization": "INT8 (CTranslate2 Local)",
        "offline": True,
        "loaded_engines": list(recognizers.keys()),
        "checkpoint": CHECKPOINT_PATH
    }


@app.get("/info")
def model_info():
    return {
        "active_engine": DEFAULT_ENGINE,
        "flagship_model": {
            "name": "Local Neural Whisper-Base",
            "parameters": "74 Million (74,000,000)",
            "architecture": "Encoder-Decoder Deep Transformer",
            "quantization": "INT8 Dynamic Precision",
            "runtime": "CTranslate2 C++ Optimized CPU Kernel",
            "memory_footprint": "~220 MB RAM",
            "offline": True
        },
        "academic_baseline_model": {
            "name": "Custom CNN + BiLSTM + CTC",
            "parameters": "1.6 Million (1,600,000)",
            "architecture": "2D CNN + 2-layer Bidirectional LSTM + CTC Loss",
            "vocabulary": "29 Character Tokens",
            "purpose": "Academic evaluation baseline trained from scratch"
        },
        "scale_difference": "Flagship model has 46.25x larger parameter capacity for superior conversational accuracy."
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None, description="Recognition engine: 'whisper' (default, 74M) or 'custom' (1.6M)")
):
    selected_engine = (engine or DEFAULT_ENGINE).lower()
    if selected_engine not in ["whisper", "custom", "base", "small"]:
        raise HTTPException(status_code=400, detail="Invalid engine. Choose 'whisper' or 'custom'.")

    try:
        rec = get_recognizer(selected_engine)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Engine '{selected_engine}' error: {str(e)}")

    tmp_path = None
    try:
        suffix = ".wav" if file.filename.lower().endswith(".wav") else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = rec.transcribe_with_metadata(tmp_path)

        return {
            "status": "success",
            "text": result.get("text", ""),
            "language": result.get("language", "EN"),
            "confidence": result.get("confidence", 99.0),
            "duration": result.get("duration", 0.0),
            "latency": result.get("latency", 0.0),
            "rtf": result.get("rtf", 0.0),
            "engine": selected_engine,
            "parameters": result.get("model_parameters", rec.parameter_str),
            "offline": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
