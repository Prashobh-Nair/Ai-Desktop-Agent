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
    """Standalone Local Speech Recognizer backed by Local Neural Whisper (INT8)."""

    def __init__(
        self,
        checkpoint_path: str = None,
        engine: str = "whisper",
        model_size: str = "small.en",
        device: str = None,
    ):
        self.engine = "whisper"
        self.engine_type = "whisper"
        self.device = device or "cpu"
        self.whisper_engine = LocalWhisperEngine(
            model_size=model_size,
            device="cuda" if str(self.device) == "cuda" else "cpu",
            compute_type="int8",
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
        return self.whisper_engine.transcribe(audio_path)

    def transcribe_with_metadata(
        self,
        audio_path: str,
        initial_prompt: str = None,
        hotwords: str = None,
    ) -> dict:
        """Transcribes with complete metadata (latency, RTF, parameters, language)."""
        kwargs = {}
        if initial_prompt is not None:
            kwargs["initial_prompt"] = initial_prompt
        if hotwords is not None:
            kwargs["hotwords"] = hotwords
        return self.whisper_engine.transcribe_with_metadata(audio_path, **kwargs)

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
