# Custom Speech-to-Text (STT) Neural Network

An end-to-end, trainable Speech-to-Text neural network built from scratch using **PyTorch**, **Log-Mel Spectrograms**, **2D CNN**, **Bidirectional LSTM**, and **CTC Loss (Connectionist Temporal Classification)**.

Designed for embedded desktop integration, local edge inference, and academic evaluation without reliance on third-party speech APIs or pretrained models (such as Whisper or Google Speech).

---

## Architecture Overview

```
Raw Audio Waveform (16 kHz Mono)
              │
              ▼
┌───────────────────────────────────────────┐
│ Log-Mel Spectrogram Extractor             │  (80 Mel Bins, 25ms Window, 10ms Hop)
└───────────────────────────────────────────┘
              │  [Batch, 1, 80, Time]
              ▼
┌───────────────────────────────────────────┐
│ 2D CNN Feature Extractor                  │
│  - Conv2D (1 -> 32), Stride (2, 2) + BN   │  Downsamples Time x2, Frequency x2
│  - Conv2D (32 -> 64), Stride (2, 1) + BN  │  Downsamples Frequency x2
└───────────────────────────────────────────┘
              │  [Batch, Time/2, 1280]
              ▼
┌───────────────────────────────────────────┐
│ Linear Feature Projection + Dropout       │  (1280 -> 256)
└───────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ 2-Layer Bidirectional LSTM                │  (Hidden size: 192, Bidirectional = 384)
└───────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ Layer Normalization & Linear Classifier   │  (384 -> 29 Vocabulary Tokens)
└───────────────────────────────────────────┘
              │  Logits [Batch, Time/2, 29]
              ▼
┌───────────────────────────────────────────┐
│ PyTorch CTC Loss / Greedy CTC Decoder     │  Collapses repeats & removes blanks
└───────────────────────────────────────────┘
              │
              ▼
Decoded Character Transcription String
```

### Model Specifications
- **Input Sampling Rate:** 16,000 Hz (mono)
- **Feature Extraction:** 80-channel Log-Mel Spectrogram with per-utterance instance normalization
- **Acoustic Backbone:** 2D CNN + 2-Layer Bidirectional LSTM + LayerNorm
- **Vocabulary Size:** 29 tokens (blank `<blank>` at index 0, space `' '`, apostrophe `'\'`, letters `a-z`)
- **Total Trainable Parameters:** **1,937,789** (~1.94 Million parameters)
- **Footprint:** ~34 MB checkpoint size

---

## Directory Structure

```
REAL/
├── checkpoints/
│   ├── best_model.pt          # Lowest validation loss checkpoint
│   ├── last_checkpoint.pt     # Latest epoch state for resume
│   └── training_log.json      # Full history (loss, CER, WER per epoch)
├── data/
│   ├── wavs/                  # 16kHz mono WAV audio files
│   ├── train.csv              # Training set manifest (audio_path, transcript, duration)
│   ├── val.csv                # Validation set manifest
│   ├── test.csv               # Unseen test set manifest
│   └── all.csv                # Complete manifest
├── samples/
│   ├── sample1.wav            # Standalone test sample 1
│   ├── sample2.wav            # Standalone test sample 2
│   └── mic_input.wav          # Output from microphone utility
├── src/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset & dynamic length bucketing
│   ├── features.py            # Log-Mel Spectrogram extraction
│   ├── metrics.py             # WER & CER computation
│   ├── model.py               # Custom CNN + BiLSTM + CTC PyTorch Module
│   └── text.py                # Tokenizer & CTC Greedy Decoder
├── app.py                     # FastAPI web service (GET /health, POST /transcribe)
├── data_prep.py               # Dataset extraction & preparation pipeline
├── inference.py               # SpeechRecognizer class & CLI inference
├── mic_record.py              # Microphone recording & live transcription tool
├── requirements.txt           # Project dependencies
├── test_stt.py                # Automated 5-stage test suite
└── train.py                   # Full training & validation pipeline with resume support
```

---

## Getting Started

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Downloads and extracts a manageable, curated subset of LibriSpeech (`dev-clean`) directly from OpenSLR, converts audio to 16 kHz Mono WAV format, and produces train/validation/test splits:

```bash
python data_prep.py
```

### 3. Model Training

Train the custom neural network from scratch:

```bash
python train.py --epochs 30 --batch-size 8 --lr 1e-3
```

To resume training from an existing checkpoint:
```bash
python train.py --resume --epochs 40
```

### 4. Running Inference

Transcribe any 16 kHz WAV audio file using the trained model checkpoint:

```bash
python inference.py --audio samples/sample1.wav
```

Programmatic Python usage:
```python
from inference import SpeechRecognizer

recognizer = SpeechRecognizer("checkpoints/best_model.pt")
text = recognizer.transcribe("samples/sample1.wav")
print("Recognized Text:", text)
```

### 5. Microphone Recording & Live Transcription

Interactive recording (Press Enter to start, Press Enter to stop):
```bash
python mic_record.py
```

Timed recording (e.g. record for 4 seconds):
```bash
python mic_record.py --duration 4.0
```

List available input microphones:
```bash
python mic_record.py --list-devices
```

### 6. FastAPI Web Service

Start the REST API server:
```bash
python app.py
```
*(Server listens on `http://127.0.0.1:8000`)*

#### Endpoints:
- **`GET /health`**: Returns service status and loaded model info.
  ```bash
  curl http://127.0.0.1:8000/health
  ```
  Response:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "checkpoint": "checkpoints/best_model.pt"
  }
  ```

- **`POST /transcribe`**: Upload a WAV file and get the transcription.
  ```bash
  curl -X POST "http://127.0.0.1:8000/transcribe" -F "file=@samples/sample1.wav"
  ```
  Response:
  ```json
  {
    "status": "success",
    "text": "h sn th  b for  o tnhor bhe inth t",
    "language": "en"
  }
  ```

### 7. Running the Automated Test Suite

Run the full end-to-end verification suite:
```bash
python test_stt.py
```

Tests verified:
1. Model checkpoint loading
2. Direct audio file inference
3. Unseen test set CER & WER evaluation
4. FastAPI endpoints (`GET /health` & `POST /transcribe`)
5. Microphone input availability

---

## Training & Evaluation Results

Experiment trained on local Intel Core Ultra 5 125H CPU across 30 epochs:

| Metric | Starting Epoch (1) | Mid Epoch (15) | Final Best Epoch (30) |
|---|---|---|---|
| **Training Loss** | `5.9617` | `2.9002` | **`1.9581`** |
| **Validation Loss** | `4.2638` | `2.8849` | **`2.0309`** |
| **Validation CER** | `98.68%` | `100.0%` | **`62.22%`** |
| **Unseen Test CER** | `100.0%` | `100.0%` | **`62.07%`** |
| **Unseen Test WER** | `100.0%` | `100.0%` | **`96.03%`** |

### Academic Notes:
1. **CTC Blank Token Dynamics:** In standard CTC training from scratch, early unconstrained optimization often falls into an "all-blank" local minimum ($L \approx -\ln(1/29) \approx 2.88$). By applying an initial negative bias to the blank token (`classifier.bias[0] = -3.0`) and introducing Layer Normalization, the network is compelled to align acoustic features with character tokens, bringing validation CER down to ~62% in only 30 CPU epochs.
2. **Dynamic Length Bucketing:** Speech datasets exhibit large duration variances. Using `BucketBatchSampler` to cluster similar audio durations per batch eliminated unnecessary zero-padding, cutting CPU training time per epoch from $142\text{ s}$ down to $\sim 10\text{ s}$ ($>10\times$ speedup).

