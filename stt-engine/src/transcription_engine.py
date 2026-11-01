"""
Autonomous Desktop Agent - Speech-to-Text Neural Decoding Engine
Features INT8 CPU optimization, Voice Activity Detection, timestamp alignment, and latency telemetry.
"""
import time
from typing import Dict, Any, Optional, List
import numpy as np
from config import settings
from src.model_manager import AcousticModelManager
from src.audio_processor import AudioProcessor
from src.vad_engine import VADEngine
from src.logger import log_info, log_success, log_error, get_process_memory_mb


class TranscriptionEngine:
    """
    High-performance, local Neural Speech-to-Text Transcription Engine.
    """
    def __init__(
        self,
        model_manager: Optional[AcousticModelManager] = None,
        vad_engine: Optional[VADEngine] = None,
    ):
        self.model_manager = model_manager or AcousticModelManager()
        self.vad_engine = vad_engine or VADEngine()
        self.model = None

    def initialize(self, offline: bool = True):
        """Preloads model into memory."""
        if self.model is None:
            self.model = self.model_manager.load_model(offline=offline)

    def transcribe(
        self,
        audio_input,
        language: Optional[str] = settings.DEFAULT_LANGUAGE,
        beam_size: int = settings.BEAM_SIZE,
        use_vad: bool = settings.VAD_FILTER,
        word_timestamps: bool = settings.WORD_TIMESTAMPS,
    ) -> Dict[str, Any]:
        """
        Transcribes audio array or file path using INT8 acoustic transformer decoding.
        """
        self.initialize()

        # Load and normalize audio
        audio_array = AudioProcessor.load_audio(audio_input)
        audio_duration = AudioProcessor.get_duration_seconds(audio_array)

        if len(audio_array) == 0:
            return {
                "text": "",
                "segments": [],
                "language": language or "unknown",
                "duration": 0.0,
                "inference_time": 0.0,
                "real_time_factor": 0.0,
                "ram_usage_mb": get_process_memory_mb()
            }

        vad_params = self.vad_engine.get_vad_parameters() if use_vad else None

        t0 = time.perf_counter()

        # Run Neural Acoustic INT8 Inference
        segments_gen, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=beam_size,
            temperature=settings.TEMPERATURE,
            vad_filter=use_vad,
            vad_parameters=vad_params,
            word_timestamps=word_timestamps
        )

        # Collect segment data
        collected_segments = []
        full_text_parts = []

        for seg in segments_gen:
            full_text_parts.append(seg.text.strip())
            seg_dict = {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            if word_timestamps and seg.words:
                seg_dict["words"] = [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3)
                    }
                    for w in seg.words
                ]
            collected_segments.append(seg_dict)

        inference_time = time.perf_counter() - t0
        rtf = round(inference_time / max(0.01, audio_duration), 3)
        final_text = " ".join(full_text_parts)

        return {
            "text": final_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(audio_duration, 2),
            "inference_time": round(inference_time, 3),
            "real_time_factor": rtf,
            "ram_usage_mb": round(get_process_memory_mb(), 1),
            "segments": collected_segments
        }
