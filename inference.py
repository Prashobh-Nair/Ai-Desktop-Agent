"""
Unified Inference module supporting both:
1. Production High-Accuracy Engine: Wav2Vec2-CTC (facebook/wav2vec2-base-960h)
2. Academic Custom Engine: Custom CNN + BiLSTM + CTC (from scratch checkpoint)
"""
import os
import argparse
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from src.text import Tokenizer, CTCGreedyDecoder
from src.model import SpeechRecognitionModel


class SpeechRecognizer:
    """
    Speech Recognizer supporting both custom-trained checkpoint and Wav2Vec2 CTC.
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        engine: str = "wav2vec2",
        device: str = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.engine = engine.lower()

        if self.engine == "custom":
            ckpt = checkpoint_path or "checkpoints/best_model.pt"
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Custom model checkpoint not found at: {ckpt}")
                
            self.tokenizer = Tokenizer()
            self.decoder = CTCGreedyDecoder(self.tokenizer)
            checkpoint = torch.load(ckpt, map_location=self.device)
            config = checkpoint.get("config", {})
            
            self.model = SpeechRecognitionModel(
                n_mels=config.get("n_mels", 80),
                vocab_size=config.get("vocab_size", self.tokenizer.vocab_size),
                cnn_channels=config.get("cnn_channels", [32, 64]),
                rnn_hidden=config.get("rnn_hidden", 192),
                rnn_layers=config.get("rnn_layers", 2),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.checkpoint_epoch = checkpoint.get("epoch", "unknown")
            self.best_val_loss = checkpoint.get("best_val_loss", None)
            self.wav2vec_rec = None
        else:
            from src.wav2vec2_engine import Wav2Vec2Recognizer
            self.wav2vec_rec = Wav2Vec2Recognizer(device=str(self.device))
            self.checkpoint_epoch = "pretrained"
            self.best_val_loss = None

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

        target_sr = 16000
        if sr != target_sr:
            import math
            gcd = math.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            data = resample_poly(data, up, down).astype(np.float32)

        return data

    def transcribe(self, audio_path: str) -> str:
        """Transcribes a WAV file or audio array to English text."""
        if self.engine == "custom":
            audio_data = self.load_audio(audio_path)
            waveform = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, output_lens = self.model(waveform)
                decoded = self.decoder.decode(logits, output_lens)
            return decoded[0] if decoded else ""
        else:
            return self.wav2vec_rec.transcribe(audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with SpeechRecognizer")
    parser.add_argument("--audio", type=str, default="samples/sample1.wav")
    parser.add_argument("--engine", type=str, choices=["wav2vec2", "custom"], default="wav2vec2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    recognizer = SpeechRecognizer(checkpoint_path=args.checkpoint, engine=args.engine)
    print(f"[*] Engine: {args.engine.upper()}")
    
    if os.path.exists(args.audio):
        result = recognizer.transcribe(args.audio)
        print(f"[*] Audio: {args.audio}")
        print(f"[*] Transcription: '{result}'")
    else:
        print(f"[!] Audio file {args.audio} not found.")
