# Sinusoidal positional encodings

import torch
from torch import nn
from typing import Optional
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    A Sinusoidal Positional Encoding layer that adds sinusoidal positional encodings to the input.
    """
    def __init__(self, d_model: int, max_seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer("pe", self.make_sinusoidal_embeddings(max_seq_len, d_model))

    def make_sinusoidal_embeddings(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """
        Make sinusoidal embeddings for the positional encoding.
        """
        position = torch.arange(max_seq_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.empty(max_seq_len, d_model, device=self.device, dtype=self.dtype)
        pe[:, ::2] = torch.sin(position.unsqueeze(1) * div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term.unsqueeze(0))
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sinusoidal positional encodings for the given position indices.

        Args:
            x (Int[Tensor, " ... sequence_length"]): Position indices for which to get positional encodings.

        Returns:
            Float[Tensor, " ... sequence_length d_model"]: The positional encodings for the given positions.
        """
        return self.pe[x]