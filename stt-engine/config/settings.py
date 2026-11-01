"""
Autonomous Desktop Agent - Speech-to-Text Engine Configuration
Centralized configuration for offline INT8 neural acoustic transcription.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_WEIGHTS_DIR = MODELS_DIR / "whisper-small"

# Model Configuration
MODEL_SIZE = "base"
LOCAL_MODEL_PATH = str(MODELS_DIR / "acoustic-model")
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 2   # Optimized for 8GB RAM system & Intel Core Ultra 5 125H
NUM_WORKERS = 1

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
DEFAULT_RECORD_DURATION = 5.0  # seconds

# Voice Activity Detection (VAD)
VAD_FILTER = True
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION_MS = 500
SPEECH_PAD_MS = 300

# Acoustic Decoding Parameters
BEAM_SIZE = 5
TEMPERATURE = 0.0
DEFAULT_LANGUAGE = None  # None for auto-detection; or "en", "es", "fr", etc.
WORD_TIMESTAMPS = True
