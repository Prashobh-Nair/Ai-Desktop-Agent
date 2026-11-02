"""
Root CLI entrypoint for the AI Desktop Agent TTS Engine.
Forwards to tts_module.main.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tts_module.main import main

if __name__ == "__main__":
    main()

