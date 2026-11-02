# AI-Powered Autonomous Desktop Agent

An intelligent desktop assistant allowing users to interact with their computers through natural voice commands, real-time speech processing, autonomous task execution, and studio-grade voice synthesis.

> **Development Team & Architecture Modules:**
> - **Module 1: STT (Speech-to-Text)** – Prashobh Nair *(Custom CNN + BiLSTM + CTC Neural Network)*
> - **Module 2: Brain / LLM** – Sahaj Patel *(Intent Understanding, Agent Planning & OS Operations)*
> - **Module 3: TTS (Text-to-Speech)** – Custom Local Engine *(Kokoro-82M Neural Speech Synthesis)*

---

## Overall System Architecture & Workflow

```
       USER VOICE (Microphone)
                  │
                  ▼
┌──────────────────────────────────────────┐
│ MODULE 1: SPEECH-TO-TEXT (STT)           │  (Prashobh Nair)
│  - 80-channel Log-Mel Spectrogram        │  16 kHz Audio -> Text Transcription
│  - 2D CNN + 2-Layer Bidirectional LSTM   │  1.94M Parameters, Offline PyTorch
└──────────────────────────────────────────┘
                  │  Transcribed Text Command
                  ▼
┌──────────────────────────────────────────┐
│ MODULE 2: BRAIN / LLM                    │  (Sahaj Patel)
│  - Natural Language Intent Understanding │  Autonomous Task Reasoning
│  - Desktop OS Action Dispatcher          │  Generates Conversational Response
└──────────────────────────────────────────┘
                  │  Response Text
                  ▼
┌──────────────────────────────────────────┐
│ MODULE 3: TEXT-TO-SPEECH (TTS)           │  (Local Kokoro-82M Engine)
│  - Intelligent Preprocessing & Chunking  │  82M Parameters, Multi-Voice
│  - CPU-Optimized ONNX Runtime Inference  │  WAV Audio Output (24 kHz)
└──────────────────────────────────────────┘
                  │  Synthesized Speech Audio
                  ▼
       AUDIO OUTPUT (Windows Speakers)
```

---

## Directory Structure

```text
Ai-Desktop-Agent/
│
├── checkpoints/                       # Module 1: Saved STT model weights
│   ├── best_model.pt                  # Lowest validation loss checkpoint
│   ├── last_checkpoint.pt             # Latest epoch state for resume
│   └── training_log.json              # STT training history (loss, CER, WER)
│
├── data/                              # Module 1: Audio dataset and manifests
│   ├── wavs/                          # 16kHz mono WAV audio files
│   ├── train.csv                      # Training set manifest
│   ├── val.csv                        # Validation set manifest
│   └── test.csv                       # Unseen test set manifest
│
├── samples/                           # Audio test recordings
│   ├── sample1.wav                    # Standalone test sample 1
│   ├── sample2.wav                    # Standalone test sample 2
│   └── mic_input.wav                  # Output from microphone utility
│
├── src/                               # Module 1: STT core architecture
│   ├── dataset.py                     # Dynamic length bucketing dataset
│   ├── features.py                    # Log-Mel Spectrogram extraction
│   ├── metrics.py                     # WER & CER computation
│   ├── model.py                       # CNN + BiLSTM + CTC PyTorch Module
│   └── text.py                        # Tokenizer & CTC Greedy Decoder
│
├── tts_module/                        # Module 3: Local Neural TTS Engine (Kokoro-82M)
│   ├── models/
│   │   └── kokoro/                    # Pre-trained Kokoro ONNX model & voice bins
│   ├── audio/
│   │   └── output/                    # Synthesized WAV audio outputs
│   ├── src/
│   │   ├── tts_engine.py              # Core neural synthesis engine & public API
│   │   ├── audio_player.py            # Windows hardware playback & interruption controls
│   │   ├── text_processor.py          # LLM markdown cleaning & normalization
│   │   └── config.py                  # Voice registry & hardware configurations
│   ├── tests/
│   │   ├── test_text_processor.py     # Unit tests for text normalization
│   │   ├── test_tts_engine.py         # Integration tests for speech synthesis
│   │   └── benchmark.py               # Performance, latency & RAM benchmarks
│   ├── download_model.py              # Automatic Kokoro model downloader
│   ├── demo_pipeline.py               # Full STT -> LLM -> TTS pipeline demo
│   ├── main.py                        # TTS CLI & interactive console
│   └── README.md                      # Detailed TTS technical documentation
│
├── app.py                             # Module 1: STT FastAPI web service
├── data_prep.py                       # Module 1: Dataset extraction & preparation
├── inference.py                       # Module 1: CLI SpeechRecognizer
├── mic_record.py                      # Module 1: Microphone live transcription
├── test_stt.py                        # Module 1: Automated 5-stage STT test suite
├── train.py                           # Module 1: STT training pipeline
└── requirements.txt                   # Complete project dependencies
```

---

# MODULE 1: SPEECH-TO-TEXT (STT)
*Developed by Prashobh Nair*

An end-to-end, trainable Speech-to-Text neural network built from scratch using **PyTorch**, **Log-Mel Spectrograms**, **2D CNN**, **Bidirectional LSTM**, and **CTC Loss (Connectionist Temporal Classification)**.

Designed for embedded desktop integration, local edge inference, and academic evaluation without reliance on third-party speech APIs or external cloud services.

### STT Architecture
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

### STT Model Specifications
- **Input Sampling Rate:** 16,000 Hz (mono)
- **Feature Extraction:** 80-channel Log-Mel Spectrogram with per-utterance instance normalization
- **Acoustic Backbone:** 2D CNN + 2-Layer Bidirectional LSTM + LayerNorm
- **Vocabulary Size:** 29 tokens (blank `<blank>` at index 0, space `' '`, apostrophe `'\'`, letters `a-z`)
- **Total Trainable Parameters:** **1,937,789** (~1.94 Million parameters)
- **Footprint:** ~34 MB checkpoint size

### STT Usage
```bash
# Transcribe single WAV file
python inference.py --audio samples/sample1.wav

# Record and transcribe from microphone
python mic_record.py

# Launch FastAPI speech server
python app.py
```

---

# MODULE 3: TEXT-TO-SPEECH (TTS)
*Custom Local Kokoro-82M Engine*

Converts responses generated by Sahaj's LLM brain into natural, expressive, human-like speech played through the user's computer speakers.

### Key Capabilities
- **Neural Architecture:** **Kokoro-82M** (82 million parameters, Apache 2.0 license).
- **100% Offline & Private:** Operates entirely locally with zero external API keys or cloud dependencies.
- **54+ High-Fidelity Voice Profiles:** American (`af_heart`, `am_adam`, `af_bella`, `am_michael`) and British (`bf_emma`, `bm_george`, etc.) voices.
- **Adjustable Speaking Speed:** Seamless rate control from `0.5x` (slow) to `2.0x` (rapid).
- **Intelligent Text Normalization:** Built-in normalization for numbers, currencies (`$`, `£`, `€`, `₹`), dates, times, units (`GB`, `MB`, `GHz`, `°C`), tech acronyms (`AI`, `LLM`, `STT`, `TTS`, `RAM`, `CPU`), and markdown stripping.
- **Long LLM Response Chunking:** Automatically divides lengthy multi-sentence responses into natural clauses and inserts acoustic pauses.
- **Low-Latency & Lightweight:** Model RAM footprint is only **~340 MB**, and CPU Real-Time Factor (RTF) averages **~0.55x** (faster than real-time).
- **Zero-Latency Audio Caching:** Instant (<1 ms) retrieval for repetitive assistant phrases (*"Yes sir"*, *"Command completed"*).
- **Reliable Windows Playback:** Dual-layer audio driver utilizing `sounddevice` with automatic fallback to Windows native `winsound`.

### Python Integration Interface (For Module 2 Brain)
```python
from tts_module import generate_speech, play_audio, speak

# 1. Generate speech from LLM response text
audio_path = generate_speech("System diagnostics report: All background services are running normally.")

# 2. Play audio through Windows speakers
play_audio(audio_path, blocking=True)

# 3. All-in-one assistant call (Synthesize & Speak)
speak("Hello! I am your AI desktop assistant.")
```

### TTS Performance Benchmarks (Windows CPU)
| Benchmark Scenario | Word Count | Audio Duration | Generation Time | Real-Time Factor (RTF)* | RAM Usage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Short Command** (*"Command executed successfully."*) | 3 | 1.90 s | 1.36 s | **0.71x** | 450 MB |
| **Interactive Question** (*"Would you like me to open the report...?"*) | 19 | 5.46 s | 2.93 s | **0.54x** | 530 MB |
| **Numbers & Currency** (*"$1,450.75 on Nov 2nd, 2026 at 99.8% in 250 ms..."*) | 16 | 12.78 s | 7.04 s | **0.55x** | 735 MB |
| **Technical Metrics** (*"CPU at 45% and memory at 3.2 GB out of 8 GB RAM..."*) | 18 | 8.68 s | 4.91 s | **0.56x** | 893 MB |
| **Long LLM Response** (*Full multi-sentence technical paragraph*) | 49 | 26.78 s | 14.75 s | **0.55x** | 898 MB |
| **Complex Punctuation** (*"Wait! What? No way... Are you sure?! Yes, 100%."*) | 11 | 4.42 s | 2.93 s | **0.66x** | 1211 MB |

*\*Real-Time Factor (RTF) = Generation Time / Audio Duration. An RTF < 1.0 indicates synthesis is faster than real-time speech playback.*

### TTS Usage
```bash
# Verify or download pre-trained Kokoro weights locally (runs once)
python tts_module/download_model.py

# Test speech synthesis and playback
python tts_module/main.py --text "Hello Prashobh and Sahaj, the desktop agent voice engine is operational." --play

# Launch interactive speech console
python tts_module/main.py --interactive

# Run performance benchmarks
python tts_module/main.py --benchmark
```

---

## End-to-End System Demonstration

To test the complete three-module pipeline (`STT` -> `LLM Brain` -> `Kokoro-82M TTS` -> `Speakers`):
```bash
python tts_module/demo_pipeline.py
```
This executes the full pipeline across four desktop agent workflows:
1. **System Performance Query:** STT recognizes speech, Brain queries metrics, TTS speaks CPU & RAM diagnostics.
2. **Application Launch:** Brain dispatches desktop action (*"Open Google Chrome"*), TTS confirms operation.
3. **Calendar Scheduling:** Brain schedules reminder for 10:00 AM, TTS confirms with natural date/time inflection.
4. **Team Credits:** Demonstrates speech synthesis across team module responsibilities.

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/Prashobh-Nair/Ai-Desktop-Agent.git
cd Ai-Desktop-Agent

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies for both STT and TTS modules
pip install -r requirements.txt
```

### 2. Verify Models
```bash
# Setup Kokoro-82M TTS weights locally
python tts_module/download_model.py
```

### 3. Run Automated Test Suites
```bash
# Run STT test suite
python test_stt.py

# Run TTS test suites (26 tests)
python -m unittest discover -s tts_module/tests -p "test_*.py"
```

---

## License & Credits

- **STT Module:** Prashobh Nair
- **LLM Brain Module:** Sahaj Patel
- **TTS Module:** Custom Kokoro-82M Engine
- **Kokoro-82M:** Apache 2.0 License by hexgrad
