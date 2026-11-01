"""
Custom Speech-to-Text Model: CNN + BiLSTM + CTC
"""
import torch
import torch.nn as nn
from src.features import LogMelSpectrogram
from src.text import Tokenizer, CTCGreedyDecoder


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm2d, ReLU, and Dropout."""
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class SpeechRecognitionModel(nn.Module):
    """
    End-to-end Speech Recognition Architecture:
    1. Log-Mel Spectrogram Front-end
    2. 2D CNN Feature Extractor (spectral-temporal feature learning & downsampling)
    3. Bidirectional Multi-layer LSTM (temporal sequence modeling)
    4. Linear Output Projection to character vocabulary
    5. Integrated Greedy CTC Decoding
    """
    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 28,
        cnn_channels: list = [32, 64],
        rnn_hidden: int = 192,
        rnn_layers: int = 3,
        rnn_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.feature_extractor = LogMelSpectrogram(n_mels=n_mels)
        
        # CNN Frontend
        # Input shape: [batch, 1, n_mels, time_frames]
        # Layer 1: stride (2, 2) -> freq: n_mels//2 = 40, time: (T+1)//2
        self.cnn1 = ConvBlock(1, cnn_channels[0], stride=(2, 2), dropout=0.1)
        # Layer 2: stride (2, 1) -> freq: 40//2 = 20, time: preserved
        self.cnn2 = ConvBlock(cnn_channels[0], cnn_channels[1], stride=(2, 1), dropout=0.1)
        
        # Frequency dimension after 2 downsampling steps
        freq_dim = n_mels // 4 # 80 -> 20
        cnn_out_dim = cnn_channels[1] * freq_dim # 64 * 20 = 1280
        
        # Projection from flattened CNN features to RNN input dim
        self.fc_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0
        )
        
        # LayerNorm and final classification layer
        self.layer_norm = nn.LayerNorm(rnn_hidden * 2)
        self.classifier = nn.Linear(rnn_hidden * 2, vocab_size)

        # Initialize weights: suppress blank token initially to prevent blank dominance
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        # Suppress blank token (index 0) so network is forced to learn character emissions
        self.classifier.bias.data[0] = -3.0

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates output temporal length after CNN stride-2 temporal downsampling.
        """
        # Downsampled by factor of 2 in layer 1
        return (input_lengths + 1) // 2

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor = None):
        """
        Args:
            x: Raw waveform [batch, samples] OR spectrogram [batch, n_mels, time]
            input_lengths: Length of each spectrogram sequence before CNN
        Returns:
            logits: [batch, out_time, vocab_size]
            output_lengths: [batch]
        """
        # If input is raw waveform, convert to log-mel spectrogram
        if x.dim() == 2: # [batch, samples]
            spec = self.feature_extractor(x) # [batch, n_mels, time]
            if input_lengths is not None:
                # hop_length is 160
                spec_lengths = (input_lengths // 160) + 1
            else:
                spec_lengths = torch.full((x.size(0),), spec.size(-1), dtype=torch.long, device=x.device)
        elif x.dim() == 3: # [batch, n_mels, time]
            spec = x
            if input_lengths is None:
                spec_lengths = torch.full((x.size(0),), spec.size(-1), dtype=torch.long, device=x.device)
            else:
                spec_lengths = input_lengths
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        # Add channel dimension: [batch, 1, n_mels, time]
        x = spec.unsqueeze(1)
        
        # CNN feature extraction
        x = self.cnn1(x) # [batch, c1, n_mels//2, time//2]
        x = self.cnn2(x) # [batch, c2, n_mels//4, time//2]
        
        # Permute to [batch, time//2, c2, freq]
        batch_size, channels, freq, time_dim = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() # [batch, time, channels, freq]
        x = x.view(batch_size, time_dim, channels * freq) # [batch, time, 1280]
        
        # Linear projection
        x = self.fc_proj(x) # [batch, time, 256]
        
        # BiLSTM
        lstm_out, _ = self.lstm(x) # [batch, time, rnn_hidden * 2]
        
        # Layer normalization & classifier logits
        lstm_norm = self.layer_norm(lstm_out)
        logits = self.classifier(lstm_norm) # [batch, time, vocab_size]
        
        output_lengths = self.get_output_lengths(spec_lengths)
        # Clamp output_lengths to actual logits time dimension
        output_lengths = torch.clamp(output_lengths, max=logits.size(1))
        
        return logits, output_lengths

