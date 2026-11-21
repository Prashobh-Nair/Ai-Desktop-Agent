# Module 2: Brain / LLM (Intent Understanding & Task Planning)
*Developed by Sahaj Patel*

The **Brain** serves as the central cognitive engine of the AI-Powered Autonomous Desktop Agent. It receives natural language transcriptions from Module 1 (Speech-to-Text), normalizes noisy or Hinglish speech, reasons over user intent, formulates a deterministic and safe execution plan, and generates spoken dialogue text for synthesis by Module 3 (Text-to-Speech).

---

## 🧠 Architectural Overview

The Brain implements a robust **dual-mode architecture**:
1. **Fine-Tuned Local Neural LLM (Ollama / GGUF):** A specialized compact LLM fine-tuned to output strict JSON execution plans conforming to Pydantic schemas. Operates 100% locally with zero cloud API keys or external network dependencies.
2. **Deterministic Rule-Based Fallback Planner:** High-speed keyword and pattern matcher (average latency < 0.2 ms) with multi-turn session awareness and confirmation barriers for critical operations. Ensures 100% operational reliability even if the local neural model is offline.

```
       TRANSCRIBED SPEECH (from Module 1: STT)
                         │
                         ▼
             ┌───────────────────────┐
             │  Text Normalization   │ (Cleans speech noise, maps app aliases,
             │   & Clause Splitter   │  translates Hinglish/Hindi command verbs)
             └───────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │     JARVIS Brain      │
             │ ┌───────────────────┐ │
             │ │ Fine-Tuned Model  │ │ (Runs locally via Ollama INT8 GGUF)
             │ └─────────┬─────────┘ │
             │           │ (Fallback on offline / invalid JSON)
             │ ┌─────────▼─────────┐ │
             │ │ Rule-Based Engine │ │ (Sub-millisecond fallback & safety barrier)
             │ └───────────────────┘ │
             └───────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Validated Plan (JSON) │ -> [Action Steps for OS Dispatcher]
             │   & Spoken Reply      │ -> [Natural Reply Text to Module 3: TTS]
             └───────────────────────┘
```

---

## 📁 Module Directory Structure

```
brain_module/
├── jarvis/
│   ├── __init__.py
│   ├── brain.py            # Master Brain coordinator (Model + Rules fallback)
│   ├── llm.py              # Ollama REST client with Pydantic JSON schema enforcement
│   ├── normalize.py        # Speech text cleaning, app resolution & Hinglish parser
│   ├── planner.py          # Deterministic rule-based fallback planner
│   └── schema.py           # Pydantic V2 schemas (Intent, Action, Step, Plan)
│
├── data/                   # Fine-tuning and evaluation datasets
│   ├── train.jsonl         # 1.6 MB fine-tuning dataset
│   ├── val.jsonl           # Validation dataset
│   └── heldout.jsonl       # Out-of-domain evaluation set
│
├── tests/
│   └── test_set.json       # 87 hand-crafted benchmark test cases
│
├── jarvis_gguf/
│   ├── Modelfile.gguf      # Ollama model definition file for GGUF
│   └── jarvis-q8.gguf      # Quantized model weights (downloaded via download_model.py)
│
├── jarvis_merged/
│   ├── Modelfile           # Ollama model definition file for merged checkpoint
│   ├── config.json         # Hugging Face model configuration
│   ├── generation_config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json      # BPE Tokenizer definition
│   └── chat_template.jinja # Model chat prompt template
│
├── chat.py                 # Interactive console for testing the Brain directly
├── download_model.py       # Automated downloader for jarvis-q8.gguf from Google Drive
├── evaluate.py             # Quantitative benchmark evaluation script
├── make_dataset.py         # Synthetic training dataset generation pipeline
├── test_stage1.py          # Schema & Pydantic validation tests
├── test_stage2.py          # Text normalization & Hinglish parsing tests
├── test_stage3.py          # End-to-end multi-intent planning tests
└── README.md               # Technical documentation
```

---

## ⚡ Key Features

- **Hinglish & Multi-Dialect Support:** Built-in normalization for conversational Indian English and Hinglish idioms (e.g. *"yaar youtube kholo aur arijit ka song chala do"* -> `open youtube`, `play arijit`).
- **Contextual Session Memory:** Remembers the currently active application across multi-turn interactions (e.g. *"play believer"* followed by *"pause"* correctly resolves to Spotify).
- **Safety Confirmation Enforcement:** Critical system-level commands (such as `shutdown`, `restart`, `sleep`, or file deletion) automatically trigger `needs_confirmation=True`.
- **Zero Hallucination Guarantee:** Pydantic validation rejects fabricated intents, targets, or actions. If the model hallucinates an invalid schema, the system automatically falls back to the deterministic planner.
- **Ultra-Low Latency:**
  - Rule-based planner: **< 0.2 ms** average latency.
  - Local neural model: **~150 - 300 ms** on standard laptop CPU.

---

## 🚀 Setup & Usage

### 1. Download Pre-Trained Model Weights
To download the fine-tuned 1.53 GB quantized model weights (`jarvis-q8.gguf`):
```bash
python brain_module/download_model.py
```

### 2. Register with Local Ollama
```bash
ollama create jarvis -f brain_module/jarvis_gguf/Modelfile.gguf
```

### 3. Interactive Testing
Test the brain interactively with typed or simulated speech input:
```bash
python brain_module/chat.py
```

### 4. Run Benchmark Evaluation
Evaluate the brain against the 87 benchmark test commands:
```bash
# Evaluate rule-based baseline
python brain_module/evaluate.py

# Evaluate fine-tuned local neural model
python brain_module/evaluate.py --mode model
```

### 5. Run Unit & Integration Tests
```bash
python brain_module/test_stage1.py
python brain_module/test_stage2.py
python brain_module/test_stage3.py
```

---

## 📊 Benchmark Results

Evaluated on `tests/test_set.json` (87 real-world desktop voice queries):

| Metric | Rule-Based Baseline | Fine-Tuned Local Neural Model |
| :--- | :---: | :---: |
| **Intent Accuracy** | **80.5%** | **94.2%** |
| **Step Accuracy** | **83.9%** | **92.0%** |
| **Safety Confirmation** | **98.9%** | **100.0%** |
| **Average Latency** | **0.1 ms** | **~220 ms** |
| **Offline Execution** | 100% Local | 100% Local (Ollama) |
