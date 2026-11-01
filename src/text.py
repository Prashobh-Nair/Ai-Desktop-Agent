"""
Text Tokenizer and CTC Greedy Decoder for Custom Speech-to-Text Model.
"""
import re
from typing import List, Union
import torch


class Tokenizer:
    """
    Character-level tokenizer for CTC-based Speech-to-Text.
    Blank token index is 0.
    """
    def __init__(self, chars: str = " 'abcdefghijklmnopqrstuvwxyz"):
        self.blank_token = "<blank>"
        self.blank_id = 0
        
        # Build token-to-id and id-to-token mappings
        self.char_list = [self.blank_token] + list(chars)
        self.char_to_id = {char: idx for idx, char in enumerate(self.char_list)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.char_list)}
        self.vocab_size = len(self.char_list)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize text to lowercase and remove unsupported punctuation.
        Keeps lowercase letters, spaces, and apostrophes.
        """
        text = text.lower().strip()
        # Replace dashes/hyphens with space
        text = text.replace("-", " ")
        # Keep letters, spaces, apostrophes
        text = re.sub(r"[^a-z' ]", "", text)
        # Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def encode(self, text: str) -> List[int]:
        """Convert string to list of character token indices."""
        clean = self.clean_text(text)
        return [self.char_to_id[char] for char in clean if char in self.char_to_id]

    def decode(self, indices: List[int]) -> str:
        """Convert list of token indices to string, omitting blank token."""
        return "".join(
            self.id_to_char[idx] for idx in indices 
            if idx in self.id_to_char and idx != self.blank_id
        )

    def __len__(self) -> int:
        return self.vocab_size


class CTCGreedyDecoder:
    """
    Standard CTC Greedy (Best Path) Decoder.
    Takes model emission log-probabilities or logits, finds argmax at each frame,
    merges repeated consecutive non-blank tokens, and removes blanks.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.blank_id = tokenizer.blank_id

    def decode(self, logits: torch.Tensor, lengths: torch.Tensor = None) -> List[str]:
        """
        Args:
            logits: Tensor of shape [batch_size, time_steps, vocab_size]
            lengths: Optional 1D Tensor of shape [batch_size] containing actual sequence lengths
        Returns:
            List of decoded transcript strings.
        """
        # Argmax along vocabulary dimension
        best_path = torch.argmax(logits, dim=-1) # [batch_size, time_steps]
        batch_size, time_steps = best_path.shape
        decoded_transcripts = []

        for b in range(batch_size):
            t_max = int(lengths[b].item()) if lengths is not None else time_steps
            indices = best_path[b, :t_max].tolist()
            
            # CTC collapse: remove consecutive duplicates and blanks
            collapsed_indices = []
            prev_token = None
            for token in indices:
                if token != prev_token:
                    if token != self.blank_id:
                        collapsed_indices.append(token)
                    prev_token = token
            
            decoded_text = self.tokenizer.decode(collapsed_indices)
            decoded_transcripts.append(decoded_text)

        return decoded_transcripts

