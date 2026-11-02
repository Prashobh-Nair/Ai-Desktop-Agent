"""
Model Downloader for Kokoro-82M (Local TTS Module)
Downloads and verifies pre-trained model weights and voice configuration files.
Ensures 100% offline local execution once downloaded.
"""

import os
import sys
import time
import requests
from pathlib import Path

# Paths
MODULE_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODULE_DIR / "models" / "kokoro"

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

# Direct release URLs
DOWNLOAD_URLS = {
    MODEL_FILE: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    VOICES_FILE: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
}

# Expected minimum file sizes in bytes for sanity check
EXPECTED_MIN_SIZES = {
    MODEL_FILE: 300_000_000,   # ~310 MB
    VOICES_FILE: 25_000_000    # ~27 MB
}


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """
    Downloads a file with streaming and console progress output.
    """
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Downloading {dest_path.name}...")
    print(f"    From: {url}")
    
    start_time = time.time()
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        elapsed = max(time.time() - start_time, 0.001)
                        speed_mb = mb_done / elapsed
                        print(f"\r    Progress: {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%) @ {speed_mb:.2f} MB/s", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r    Downloaded: {mb_done:.1f} MB", end="", flush=True)
                        
        print("\n    Download finished! Finalizing file...")
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            
        print(f"[+] Successfully saved {dest_path.name} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"\n[-] Error downloading {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def verify_models(models_dir: Path = MODELS_DIR) -> bool:
    """
    Checks if all required model files exist and meet expected size thresholds.
    """
    all_ok = True
    for filename, min_size in EXPECTED_MIN_SIZES.items():
        filepath = models_dir / filename
        if not filepath.exists():
            print(f"[-] Missing: {filename}")
            all_ok = False
        elif filepath.stat().st_size < min_size:
            print(f"[-] Incomplete file: {filename} ({filepath.stat().st_size} bytes, expected > {min_size})")
            all_ok = False
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"[+] Found: {filename} ({size_mb:.2f} MB) - OK")
    return all_ok


def ensure_models_downloaded(models_dir: Path = MODELS_DIR) -> bool:
    """
    Downloads any missing model files. Returns True if all models are present.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Kokoro-82M Model Verification & Download")
    print("=" * 60)
    print(f"Target Directory: {models_dir}")
    
    missing_files = []
    for filename, min_size in EXPECTED_MIN_SIZES.items():
        filepath = models_dir / filename
        if not filepath.exists() or filepath.stat().st_size < min_size:
            missing_files.append(filename)
            
    if not missing_files:
        print("[+] All Kokoro-82M model files are already present and verified.")
        return True
        
    print(f"[*] Need to download {len(missing_files)} file(s): {', '.join(missing_files)}")
    
    for filename in missing_files:
        url = DOWNLOAD_URLS[filename]
        dest = models_dir / filename
        success = download_file(url, dest)
        if not success:
            print(f"[-] Failed to download {filename}.")
            return False
            
    return verify_models(models_dir)


if __name__ == "__main__":
    success = ensure_models_downloaded()
    if success:
        print("\n[SUCCESS] Kokoro-82M environment is ready for local offline inference!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Kokoro-82M model setup failed. Please check your network connection.")
        sys.exit(1)

