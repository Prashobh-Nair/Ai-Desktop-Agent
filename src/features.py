"""
Audio Feature Extraction using Log-Mel Spectrograms.
"""
import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
    """
    Computes Log-Mel Spectrogram features from 16kHz raw audio waveforms.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape [batch, time_samples] or [time_samples]
        Returns:
            log_mel: Tensor of shape [batch, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_spectrogram(waveform)
        # Log-Mel
        log_mel = torch.log(mel + 1e-6)
        
        # Instance normalization across time per utterance
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-6
        normalized = (log_mel - mean) / std
        
        return normalized

