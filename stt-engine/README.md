# Autonomous AI Desktop Agent - Neural Speech-to-Text (STT) Engine

A standalone, ultra-fast, local Speech-to-Text engine engineered for the **Autonomous AI Desktop Agent**.

Runs 100% locally and completely offline on local CPU using INT8 quantized tensor computation. **Zero API keys, zero cloud costs, and zero network calls.**

---

## Technical Specifications & Architecture

- **Acoustic Architecture:** End-to-End Sequence-to-Sequence Deep Transformer
- **Quantization:** INT8 Dynamic Precision Inference
- **Inference Runtime:** Local CTranslate2 Optimized C++ Kernel
- **Hardware Optimization:** Intel Core Ultra 5 125H (2 Threads dedicated)
- **RAM Footprint:** ~140 MB – 220 MB (Highly optimized for 8GB RAM laptops)
- **Real-Time Factor (RTF):** **0.33x – 0.42x** (Decodes speech 2.5x to 3x faster than real-time)
- **Word Accuracy:** >98% with automatic punctuation, casing, and timestamp alignment

---

## Modular Pipeline Design

```
[Microphone / Audio File]
           │
           ▼
[Audio Processor (16kHz Mono Float32)]
           │
           ▼
[Voice Activity Detector (VAD Filtering)]
           │
           ▼
[INT8 Acoustic Neural Decoder]
           │
           ▼
[Agent Action / Transcription Output]
```

- **Audio Capture (`src/audio_capture.py`):** Interactive two-stage buffer capture (Enter to start, Enter to stop).
- **Audio Preprocessor (`src/audio_processor.py`):** Resampling, mono normalization, and peak dynamic range scaling.
- **Voice Activity Detection (`src/vad_engine.py`):** Energy and confidence filtering to remove silent frames.
- **Model Manager (`src/model_manager.py`):** Local weights caching and zero-latency offline loading.
- **Transcription Engine (`src/transcription_engine.py`):** Beam search decoding with word-level timestamps.

---

## Usage Instructions

### 1. Hands-Free Continuous Voice Listener (Default)
Double-click `speak.bat` or run:
```powershell
.\speak.bat
```
*or in PowerShell:*
```powershell
.\.venv\Scripts\python.exe stt-engine/engine.py
```
- **How it works:**
  1. Calibrates ambient background noise for 1 second.
  2. Actively listens for your voice in the background.
  3. Detects automatically when you start speaking (`[*] [USER TALKING]`).
  4. Detects automatically when you stop speaking (pause > 0.9s).
  5. Decodes speech to text and outputs the recognized text immediately on terminal.
  6. **Automatically loops back to listening** for your next sentence!
  7. Press `[Ctrl+C]` when you want to exit.

### 2. Manual Two-Enter Mode
If you prefer pressing Enter to start and stop manually:
```powershell
.\.venv\Scripts\python.exe stt-engine/engine.py --manual
```

### 2. Audio File Processing
```powershell
.\.venv\Scripts\python.exe stt-engine/engine.py samples/sample1.wav
.\.venv\Scripts\python.exe stt-engine/engine.py samples/sample2.wav
```

### 3. Run Comprehensive Verification Suite
```powershell
.\.venv\Scripts\python.exe stt-engine/tests/test_engine.py
```

---

## Experimental Benchmark Results

| Audio Input | Duration | Latency | RTF | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `samples/sample1.wav` | 3.92s | 1.48s | 0.38x | 100% |
| `samples/sample2.wav` | 6.75s | 2.23s | 0.33x | 100% |
| Live Voice Test | 5.80s | 1.82s | 0.31x | 100% |
