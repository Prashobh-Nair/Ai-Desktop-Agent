"""
FastAPI Speech-to-Text Microservice supporting both:
1. Production High-Accuracy Engine: Wav2Vec2-CTC
2. Academic Custom Engine: Custom CNN + BiLSTM + CTC
"""
import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from inference import SpeechRecognizer

app = FastAPI(
    title="Autonomous Desktop Agent STT API",
    description="Speech-to-Text inference service supporting custom PyTorch CNN+BiLSTM and Wav2Vec2 CTC.",
    version="2.0.0"
)

DEFAULT_ENGINE = os.environ.get("STT_ENGINE", "wav2vec2")
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
    return {
        "status": "healthy",
        "default_engine": DEFAULT_ENGINE,
        "loaded_engines": list(recognizers.keys()),
        "checkpoint": CHECKPOINT_PATH
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None, description="Recognition engine: 'wav2vec2' or 'custom'")
):
    selected_engine = (engine or DEFAULT_ENGINE).lower()
    if selected_engine not in ["wav2vec2", "custom"]:
        raise HTTPException(status_code=400, detail="Invalid engine. Choose 'wav2vec2' or 'custom'.")

    try:
        rec = get_recognizer(selected_engine)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Engine '{selected_engine}' error: {str(e)}")

    suffix = ".wav" if file.filename.lower().endswith(".wav") else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        transcribed_text = rec.transcribe(tmp_path)
        
        return {
            "status": "success",
            "text": transcribed_text,
            "engine": selected_engine,
            "language": "en"
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
