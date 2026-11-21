"""
Downloader utility for Module 2: Brain / LLM (Sahaj Patel)
Downloads the fine-tuned Jarvis GGUF model (jarvis-q8.gguf) directly from Google Drive.
"""

import os
import sys
import re
from pathlib import Path
import requests

MODEL_ID = "1fw4fXLdk7z33cHBtiOXe8Gv5kQhtdPvw" # jarvis-q8.gguf (1.53 GB)
BASE_DIR = Path(__file__).resolve().parent
TARGET_FILE = BASE_DIR / "jarvis_gguf" / "jarvis-q8.gguf"

def download_gguf_model(force: bool = False):
    """Downloads jarvis-q8.gguf if not already present."""
    if TARGET_FILE.exists() and not force:
        size_gb = TARGET_FILE.stat().st_size / (1024 ** 3)
        print(f"[OK] Jarvis GGUF model already exists: {TARGET_FILE} ({size_gb:.2f} GB)")
        return str(TARGET_FILE)

    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading pre-trained Jarvis model (jarvis-q8.gguf) from Google Drive...")
    print(f"[*] Target location: {TARGET_FILE}")

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

    r = session.get(url, headers=headers, stream=True)

    if "Google Drive - Virus scan warning" in r.text:
        action_match = re.search(r'action="([^"]+)"', r.text)
        if action_match:
            action = action_match.group(1)
            params = {}
            for m in re.finditer(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text):
                params[m.group(1)] = m.group(2)
            r = session.get(action, params=params, headers=headers, stream=True)

    total_downloaded = 0
    with open(TARGET_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=2 * 1024 * 1024): # 2MB chunks
            if chunk:
                f.write(chunk)
                total_downloaded += len(chunk)
                if total_downloaded % (50 * 1024 * 1024) == 0:
                    mb = total_downloaded / (1024 * 1024)
                    print(f"    ... {mb:.1f} MB downloaded", flush=True)

    size_gb = TARGET_FILE.stat().st_size / (1024 ** 3)
    print(f"[SUCCESS] Download complete! Model size: {size_gb:.2f} GB")
    print(f"[*] To register in Ollama, run: ollama create jarvis -f {BASE_DIR / 'jarvis_gguf' / 'Modelfile.gguf'}")
    return str(TARGET_FILE)

if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_gguf_model(force=force_download)
