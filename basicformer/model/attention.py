# Multi-head and Scaled Dot-Product attention
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from basicformer.utils.Softmax import Softmax
from basicformer.model.modules import TruncNormalNoBiasLinear


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention.
    """
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor of shape (..., seq_len_q, d_k)
            K: Key tensor of shape (..., seq_len_k, d_k)
            V: Value tensor of shape (..., seq_len_v, d_v)
            mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k) where True indicates
                   positions to attend to and False indicates positions to mask out.

        Returns:
            Attention output of shape (..., seq_len_q, d_v)
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to additive mask: False -> -inf, True -> 0
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_probs = self.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, V)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer that applies the multi-head attention mechanism to the input.
    """
    def __init__(self, d_model: int, num_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.causal = causal
        
        # Query, Key, Value projection layers
        self.q_proj = TruncNormalNoBiasLinear(d_model, d_model)
        self.k_proj = TruncNormalNoBiasLinear(d_model, d_model)
        self.v_proj = TruncNormalNoBiasLinear(d_model, d_model)
        
        # Output projection
        self.output_proj = TruncNormalNoBiasLinear(d_model, d_model)
        
        # Scaled dot-product attention
        self.sdpa = ScaledDotProductAttention()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention to the input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)  # (batch, seq_len, d_model)
        V = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape and transpose for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Create causal mask if needed
        mask = None
        if self.causal:
            # Create lower triangular mask: True for positions we can attend to, False for future positions
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            # Expand for batch and heads: (1, 1, seq_len, seq_len)
            mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply scaled dot-product attention
        attn_output = self.sdpa(Q, K, V, mask)  # (batch, num_heads, seq_len, d_v)
        
        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.output_proj(attn_output)  # (batch, seq_len, d_model)
        
        return output