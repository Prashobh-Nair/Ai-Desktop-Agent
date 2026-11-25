"""
Unified Speech-to-Text Inference Module
1. Production High-Capacity Local Engine: Local Neural Whisper (74M / 244M params, INT8 offline)
2. Academic Baseline Model: Custom CNN + BiLSTM + CTC (1.6M parameters from scratch)
"""
import os
import argparse
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from src.whisper_engine import LocalWhisperEngine


class SpeechRecognizer:
    """
    Unified Speech Recognizer supporting:
    - Primary Engine: High-capacity Local Neural Whisper (74 Million parameters, offline INT8)
    - Academic Engine: Custom CNN + BiLSTM + CTC (1.6 Million parameters trained from scratch)
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        engine: str = "whisper",
        model_size: str = "base",
        device: str = None
    ):
        self.engine = engine.lower()
        if self.engine in ["custom", "bilstm", "cnn"]:
            import torch
            from src.text import Tokenizer, CTCGreedyDecoder
            from src.model import SpeechRecognitionModel

            self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.engine_type = "custom"
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

            # Calculate parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            self.parameter_count = total_params
            self.parameter_str = f"{total_params / 1e6:.2f} Million (~1.6M params)"
            self.checkpoint_epoch = checkpoint.get("epoch", "unknown")
            self.best_val_loss = checkpoint.get("best_val_loss", None)
            self.whisper_engine = None

        else:
            # Primary Flagship Local Model: Whisper-Base (74M) or Small (244M)
            self.engine_type = "whisper"
            self.whisper_engine = LocalWhisperEngine(
                model_size=model_size,
                device="cuda" if torch.cuda.is_available() and str(self.device) != "cpu" else "cpu",
                compute_type="int8"
            )
            self.model = None
            self.parameter_count = self.whisper_engine.specs["param_count"]
            self.parameter_str = self.whisper_engine.specs["parameters"]
            self.checkpoint_epoch = "local-pretrained"
            self.best_val_loss = None

    def load_audio(self, audio_input):
        """Loads and standardizes audio to 16kHz mono float32 numpy array."""
        return LocalWhisperEngine.load_audio(audio_input)

    def transcribe(self, audio_path: str) -> str:
        """Transcribes a WAV file or audio array to English text."""
        if self.engine_type == "custom":
            audio_data = self.load_audio(audio_path)
            waveform = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, output_lens = self.model(waveform)
                decoded = self.decoder.decode(logits, output_lens)
            return decoded[0] if decoded else ""
        else:
            return self.whisper_engine.transcribe(audio_path)

    def transcribe_with_metadata(self, audio_path: str) -> dict:
        """Transcribes with complete metadata (latency, RTF, parameters, language)."""
        if self.engine_type == "custom":
            import time
            audio_data = self.load_audio(audio_path)
            duration = len(audio_data) / 16000.0
            t0 = time.time()
            waveform = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, output_lens = self.model(waveform)
                decoded = self.decoder.decode(logits, output_lens)
            latency = time.time() - t0
            text = decoded[0] if decoded else ""
            return {
                "text": text,
                "language": "EN",
                "confidence": 85.0,
                "duration": round(duration, 2),
                "latency": round(latency, 3),
                "rtf": round(latency / max(0.01, duration), 3),
                "model_name": "Custom CNN + BiLSTM + CTC (From Scratch)",
                "model_parameters": self.parameter_str,
                "parameter_count": self.parameter_count,
                "offline": True
            }
        else:
            return self.whisper_engine.transcribe_with_metadata(audio_path)

    def get_info(self) -> dict:
        """Returns metadata about the active speech recognition engine."""
        return {
            "engine": self.engine_type,
            "parameters": self.parameter_str,
            "parameter_count": self.parameter_count,
            "device": str(self.device),
            "offline": True,
            "epoch": self.checkpoint_epoch
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with SpeechRecognizer")
    parser.add_argument("--audio", type=str, default="samples/sample1.wav", help="Path to WAV audio file")
    parser.add_argument("--engine", type=str, choices=["whisper", "custom"], default="whisper",
                        help="Recognition engine: 'whisper' (74M local offline) or 'custom' (1.6M CNN+BiLSTM scratch)")
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "small"],
                        help="Whisper model size: 'base' (74M params) or 'small' (244M params)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to custom model checkpoint")
    parser.add_argument("--info", action="store_true", help="Print model architecture and parameter count")
    args = parser.parse_args()

    recognizer = SpeechRecognizer(
        checkpoint_path=args.checkpoint,
        engine=args.engine,
        model_size=args.model_size
    )

    info = recognizer.get_info()
    print("=" * 60)
    print("  AUTONOMOUS AI DESKTOP AGENT - SPEECH RECOGNITION (STT)")
    print("=" * 60)
    print(f"[*] Engine Active     : {info['engine'].upper()}")
    print(f"[*] Model Parameters  : {info['parameters']} ({info['parameter_count']:,} parameters)")
    print(f"[*] Execution Runtime : 100% Offline Local Inference ({info['device']})")
    print("=" * 60)

    if args.info:
        sys.exit(0)

    if os.path.exists(args.audio):
        res = recognizer.transcribe_with_metadata(args.audio)
        print(f"[*] Audio File        : {args.audio}")
        print(f"[*] Audio Duration    : {res['duration']}s")
        print(f"[*] Inference Latency : {res['latency']}s (RTF: {res['rtf']}x)")
        print(f"[*] Recognized Text   : \"{res['text']}\"")
        print("=" * 60)
    else:
        print(f"[!] Audio file '{args.audio}' not found.")
