"""
Automated Test Suite for Speech-to-Text Pipeline:
1. Model checkpoint loading
2. Direct model inference on sample WAV files
3. Character Error Rate (CER) & Word Error Rate (WER) evaluation on test set
4. FastAPI server endpoints (GET /health, POST /transcribe)
5. Microphone utility verification
"""
import os
import sys
import time
import subprocess
import requests
import torch
import soundfile as sf
from inference import SpeechRecognizer
from src.text import Tokenizer, CTCGreedyDecoder
from src.dataset import SpeechDataset, speech_collate_fn
from torch.utils.data import DataLoader
from src.metrics import compute_wer, compute_cer


def test_checkpoint_loading(checkpoint_path="checkpoints/best_model.pt"):
    print("\n[TEST 1] Testing Model Checkpoint Loading...")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    recognizer = SpeechRecognizer(checkpoint_path)
    assert recognizer.model is not None, "Model failed to initialize"
    print(f"  [PASS] Checkpoint loaded successfully! (Epoch {recognizer.checkpoint_epoch}, device: {recognizer.device})")
    return recognizer


def test_inference(recognizer, audio_path="samples/sample1.wav"):
    print("\n[TEST 2] Testing Direct Audio Inference...")
    assert os.path.exists(audio_path), f"Audio file not found at {audio_path}"
    transcription = recognizer.transcribe(audio_path)
    assert isinstance(transcription, str), "Transcription output is not a string"
    print(f"  [PASS] Audio: {audio_path}")
    print(f"  [PASS] Transcription Output: '{transcription}'")
    return transcription


def test_evaluation(recognizer, test_csv="data/test.csv"):
    print("\n[TEST 3] Evaluating Model on Unseen Test Dataset...")
    assert os.path.exists(test_csv), f"Test CSV not found at {test_csv}"
    
    tokenizer = recognizer.tokenizer
    decoder = recognizer.decoder
    test_dataset = SpeechDataset(test_csv, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=speech_collate_fn)
    
    all_refs = []
    all_hyps = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveforms"].to(recognizer.device)
            waveform_lens = batch["waveform_lens"].to(recognizer.device)
            ref_texts = batch["texts"]
            
            logits, output_lens = recognizer.model(waveforms, input_lengths=waveform_lens)
            hypotheses = decoder.decode(logits, output_lens)
            
            all_refs.extend(ref_texts)
            all_hyps.extend(hypotheses)
            
    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)
    
    print(f"  [PASS] Evaluated {len(all_refs)} test samples:")
    print(f"         - Test Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"         - Test Word Error Rate (WER):      {wer * 100:.2f}%")
    if len(all_refs) > 0:
        print(f"         - Sample Ground Truth:  '{all_refs[0]}'")
        print(f"         - Sample Prediction:    '{all_hyps[0]}'")
        
    return {"cer": cer, "wer": wer, "count": len(all_refs)}


def test_fastapi_endpoints(sample_audio="samples/sample1.wav"):
    print("\n[TEST 4] Testing FastAPI Endpoints (GET /health, POST /transcribe)...")
    import uvicorn
    from app import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # 1. Test GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_json = health_resp.json()
    assert health_json["status"] == "healthy", f"Status not healthy: {health_json}"
    print(f"  [PASS] GET /health returned 200 OK: {health_json}")
    
    # 2. Test POST /transcribe
    with open(sample_audio, "rb") as f:
        files = {"file": ("sample1.wav", f, "audio/wav")}
        transcribe_resp = client.post("/transcribe", files=files)
        
    assert transcribe_resp.status_code == 200, f"POST /transcribe failed: {transcribe_resp.text}"
    transcribe_json = transcribe_resp.json()
    assert transcribe_json["status"] == "success", f"Transcribe status not success: {transcribe_json}"
    assert "text" in transcribe_json, "Transcription response missing 'text' key"
    print(f"  [PASS] POST /transcribe returned 200 OK: {transcribe_json}")


def test_microphone_utility():
    print("\n[TEST 5] Verifying Microphone Utility Functions...")
    import sounddevice as sd
    devices = sd.query_devices()
    input_devices = [d for d in devices if d["max_input_channels"] > 0]
    print(f"  [PASS] Microphone hardware query functional: {len(input_devices)} input device(s) found.")


def run_all_tests():
    print("=" * 60)
    print("  RUNNING COMPREHENSIVE STT TEST SUITE")
    print("=" * 60)
    
    recognizer = test_checkpoint_loading()
    test_inference(recognizer)
    test_evaluation(recognizer)
    test_fastapi_endpoints()
    test_microphone_utility()
    
    print("\n" + "=" * 60)
    print("  ALL 5 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

