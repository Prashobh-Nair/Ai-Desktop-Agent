"""
Pretrained Wav2Vec2-CTC Speech Recognition Engine.
Uses facebook/wav2vec2-base-960h (100% free, open-source CTC model).
"""
import os
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Recognizer:
    """
    Wav2Vec2-based CTC Speech-to-Text inference engine.
    Pretrained on 960 hours of English speech for high production accuracy.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing Wav2Vec2 CTC engine ({model_name}) on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def load_audio(self, audio_input):
        """Loads and standardizes audio to 16kHz mono float32 numpy array."""
        if isinstance(audio_input, str):
            data, sr = sf.read(audio_input, dtype="float32")
        elif isinstance(audio_input, np.ndarray):
            data = audio_input.astype(np.float32)
            sr = 16000
        elif isinstance(audio_input, torch.Tensor):
            data = audio_input.detach().cpu().numpy().astype(np.float32)
            sr = 16000
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz if necessary
        target_sr = 16000
        if sr != target_sr:
            import math
            gcd = math.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            data = resample_poly(data, up, down).astype(np.float32)

        return data

    def transcribe(self, audio_input) -> str:
        """Transcribes audio array or WAV file path to English text."""
        audio_data = self.load_audio(audio_input)
        
        # Process input features
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode using CTC processor
        transcription = self.processor.decode(predicted_ids[0])
        return transcription.strip().lower()

