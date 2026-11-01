"""
Dataset preparation script:
Downloads a manageable subset of LibriSpeech dev-clean,
converts to 16kHz mono WAV format, pairs with ground-truth transcripts,
and creates train / val / test CSV splits.
"""
import os
import io
import csv
import random
import tarfile
import urllib.request
import soundfile as sf
import numpy as np


def prepare_dataset(
    target_samples: int = 250,
    data_dir: str = "data",
    seed: int = 42,
    url: str = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
):
    random.seed(seed)
    wavs_dir = os.path.join(data_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print(f"[*] Connecting to OpenSLR LibriSpeech stream: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    extracted_transcripts = {} # utt_id -> text
    extracted_audios = {}      # utt_id -> bytes
    
    with urllib.request.urlopen(req, timeout=30) as response:
        tar = tarfile.open(fileobj=response, mode="r|gz")
        print("[*] Streaming archive and collecting audio samples & transcripts...")
        
        for member in tar:
            if not member.isfile():
                continue
                
            base_name = os.path.basename(member.name)
            
            # Transcript text file
            if base_name.endswith(".trans.txt"):
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode("utf-8", errors="ignore")
                    for line in content.strip().splitlines():
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            extracted_transcripts[utt_id] = text
                            
            # FLAC audio file
            elif base_name.endswith(".flac"):
                utt_id = base_name.replace(".flac", "")
                f = tar.extractfile(member)
                if f is not None:
                    extracted_audios[utt_id] = f.read()
                    
            # Stop once we have matched target_samples
            matched = set(extracted_transcripts.keys()).intersection(extracted_audios.keys())
            if len(matched) >= target_samples:
                print(f"[+] Reached target count of {len(matched)} matched audio-transcript pairs.")
                break

    # Find matched pairs
    matched_ids = list(set(extracted_transcripts.keys()).intersection(extracted_audios.keys()))
    matched_ids.sort()
    matched_ids = matched_ids[:target_samples]
    print(f"[*] Processing and saving {len(matched_ids)} WAV files...")

    manifest = []
    for idx, utt_id in enumerate(matched_ids):
        raw_bytes = extracted_audios[utt_id]
        transcript = extracted_transcripts[utt_id]
        
        # Read FLAC from memory
        audio_io = io.BytesIO(raw_bytes)
        data, sample_rate = sf.read(audio_io)
        
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        duration = len(data) / sample_rate
        wav_filename = f"{utt_id}.wav"
        wav_path = os.path.join(wavs_dir, wav_filename)
        
        # Save as standard 16-bit 16kHz PCM WAV
        sf.write(wav_path, data, sample_rate, subtype="PCM_16")
        
        manifest.append({
            "id": utt_id,
            "audio_path": os.path.relpath(wav_path, ".").replace("\\", "/"),
            "transcript": transcript,
            "duration": round(duration, 2)
        })

    # Shuffle for train/val/test splitting
    random.shuffle(manifest)
    
    n_total = len(manifest)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_set = manifest[:n_train]
    val_set = manifest[n_train:n_train + n_val]
    test_set = manifest[n_train + n_val:]

    def write_csv(items, filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "transcript", "duration"])
            writer.writeheader()
            writer.writerows(items)

    write_csv(manifest, os.path.join(data_dir, "all.csv"))
    write_csv(train_set, os.path.join(data_dir, "train.csv"))
    write_csv(val_set, os.path.join(data_dir, "val.csv"))
    write_csv(test_set, os.path.join(data_dir, "test.csv"))

    # Copy sample WAV files for quick testing
    if len(test_set) >= 2:
        import shutil
        shutil.copy(test_set[0]["audio_path"], "samples/sample1.wav")
        shutil.copy(test_set[1]["audio_path"], "samples/sample2.wav")
        print(f"[+] Created samples/sample1.wav (ref: '{test_set[0]['transcript']}')")
        print(f"[+] Created samples/sample2.wav (ref: '{test_set[1]['transcript']}')")

    print(f"\n[✓] Dataset Preparation Completed Successfully:")
    print(f"    - Total Samples: {n_total}")
    print(f"    - Train Samples: {len(train_set)} (data/train.csv)")
    print(f"    - Val Samples:   {len(val_set)} (data/val.csv)")
    print(f"    - Test Samples:  {len(test_set)} (data/test.csv)")


if __name__ == "__main__":
    prepare_dataset()

