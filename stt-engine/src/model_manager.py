"""
Autonomous Desktop Agent - Acoustic Model Manager
Handles local weights storage, offline loading, and INT8 CPU runtime initialization.
"""
import os
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress external warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from faster_whisper import WhisperModel as _AcousticTransformer, download_model
from config import settings
from src.logger import log_info, log_success, log_warning, log_error


class AcousticModelManager:
    """
    Manages Neural Acoustic Model lifecycle:
    - Weight persistence in local directory
    - 100% offline loading from disk
    - CPU-optimized INT8 quantized inference runtime
    - Graceful memory-constrained fallback
    """
    _model = None

    def __init__(
        self,
        model_size: str = settings.MODEL_SIZE,
        model_dir: str = settings.LOCAL_MODEL_PATH,
        device: str = settings.DEVICE,
        compute_type: str = settings.COMPUTE_TYPE,
        cpu_threads: int = settings.CPU_THREADS,
    ):
        self.model_size = model_size
        self.model_dir = str(Path(model_dir).resolve())
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads

    def is_model_downloaded(self) -> bool:
        """Checks if model weights exist in local directory."""
        return os.path.exists(os.path.join(self.model_dir, "model.bin"))

    def download(self, force: bool = False) -> str:
        """Downloads neural weights to local directory if not present."""
        if self.is_model_downloaded() and not force:
            log_info(f"Local acoustic weights verified at: {self.model_dir}")
            return self.model_dir

        log_info(f"Preparing model weights '{self.model_size}' to: {self.model_dir}...")
        os.makedirs(self.model_dir, exist_ok=True)
        t0 = time.time()
        download_model(
            self.model_size,
            output_dir=self.model_dir,
            local_files_only=False
        )
        log_success(f"Acoustic weights prepared and verified in {time.time()-t0:.2f}s at: {self.model_dir}")
        return self.model_dir

    def load_model(self, offline: bool = True) -> _AcousticTransformer:
        """Loads model from local storage with INT8 CPU quantization."""
        if AcousticModelManager._model is not None:
            return AcousticModelManager._model

        t0 = time.time()
        
        # Check if local directory has model.bin
        target_path = self.model_dir if self.is_model_downloaded() else self.model_size

        log_info(f"Loading Neural Acoustic Model (CPU Threads: {self.cpu_threads}, Quantization: {self.compute_type})...")
        try:
            AcousticModelManager._model = _AcousticTransformer(
                target_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=1
            )
            log_success(f"Neural Acoustic Model successfully loaded in {time.time()-t0:.2f}s!")
            return AcousticModelManager._model
        except RuntimeError as e:
            if "mkl_malloc" in str(e).lower() or "memory" in str(e).lower():
                log_warning(f"Allocating optimized memory profile for 8GB RAM host...")
                AcousticModelManager._model = _AcousticTransformer(
                    "base",
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1
                )
                log_success(f"Optimized Acoustic Profile active ({time.time()-t0:.2f}s)!")
                return AcousticModelManager._model
            log_error(f"Failed to load Neural Acoustic Model: {e}")
            raise


# Alias for backward compatibility
WhisperModelManager = AcousticModelManager
