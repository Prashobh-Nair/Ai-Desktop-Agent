"""
PyTorch Dataset and Collate Function for Speech-to-Text Training.
"""
import os
import csv
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
from src.text import Tokenizer


class SpeechDataset(Dataset):
    """
    Loads 16kHz WAV audio and corresponding transcription.
    Samples are sorted by duration to minimize padding in batches.
    """
    def __init__(self, csv_file: str, tokenizer: Tokenizer, max_duration: float = 8.0, min_duration: float = 1.0):
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.samples = []
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dur = float(row.get("duration", 0.0))
                # Filter manageable durations for efficient training
                if self.min_duration <= dur <= self.max_duration and len(row["transcript"].strip()) > 0:
                    self.samples.append(row)

        # Sort by duration so batches have homogeneous sequence lengths
        self.samples.sort(key=lambda x: float(x.get("duration", 0.0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        wav_path = sample["audio_path"]
        raw_text = sample["transcript"]

        # Read audio
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Convert to PyTorch tensor
        waveform = torch.from_numpy(audio)
        
        # Encode transcription
        clean_text = self.tokenizer.clean_text(raw_text)
        token_ids = self.tokenizer.encode(clean_text)
        targets = torch.tensor(token_ids, dtype=torch.long)
        
        return {
            "waveform": waveform,
            "waveform_len": len(waveform),
            "targets": targets,
            "target_len": len(targets),
            "text": clean_text,
            "audio_path": wav_path,
        }


def speech_collate_fn(batch):
    """
    Collate function to batch variable-length waveforms and targets.
    Pads waveforms with 0 and targets with 0 (blank).
    """
    # Sort batch by waveform length descending (optimal for RNN packing & efficiency)
    batch = sorted(batch, key=lambda x: x["waveform_len"], reverse=True)

    waveforms = [item["waveform"] for item in batch]
    waveform_lens = torch.tensor([item["waveform_len"] for item in batch], dtype=torch.long)
    
    target_list = [item["targets"] for item in batch]
    target_lens = torch.tensor([item["target_len"] for item in batch], dtype=torch.long)
    
    texts = [item["text"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    # Pad waveforms: [batch, max_waveform_len]
    max_wave_len = max(len(w) for w in waveforms)
    padded_waveforms = torch.zeros(len(waveforms), max_wave_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :len(w)] = w

    # Flatten targets for CTC Loss (or pad them)
    # PyTorch CTCLoss accepts 1D concatenated targets when target_lengths is given
    concatenated_targets = torch.cat(target_list)

    return {
        "waveforms": padded_waveforms,
        "waveform_lens": waveform_lens,
        "targets": concatenated_targets,
        "target_lens": target_lens,
        "texts": texts,
        "audio_paths": audio_paths
    }

