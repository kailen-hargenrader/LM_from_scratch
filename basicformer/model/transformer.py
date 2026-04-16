# Main TransformerLM and Block definitions

import torch
from torch import nn
from typing import Optional
from basicformer.model.attention import MultiHeadAttention
from basicformer.model.modules import PositionwiseFeedForward, Embedding, TruncNormalNoBiasLinear
from basicformer.model.normalization import LayerNorm
from basicformer.model.positional import SinusoidalPositionalEncoding


class TransformerBlock(nn.Module):
    """
    A pre-norm Transformer block that applies multi-head self-attention followed by
    a position-wise feed-forward network, with residual connections and layer normalization.
    
    The structure is:
    1. Layer Norm
    2. Multi-head Self-Attention
    3. Residual Connection
    4. Layer Norm
    5. Feed-forward Network
    6. Residual Connection
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer norms
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
        
        # Attention and FFN
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        Apply the Transformer block to the input.
        
        Args:
            x (Float[Tensor, "batch seq_len d_model"]): Input tensor
            
        Returns:
            Float[Tensor, "batch seq_len d_model"]: Output tensor
        """
        # Pre-norm attention block with residual connection
        x = x + self.attn(self.ln1(x) if norm else x)
        
        # Pre-norm FFN block with residual connection
        x = x + self.ffn(self.ln2(x) if norm else x)
        
        return x


class TransformerLM(nn.Module):
    """
    A Transformer language model that combines token embeddings, sinusoidal positional embeddings,
    stacked pre-norm Transformer blocks, a final LayerNorm, and an LM head.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Token embeddings
        self.token_embeddings = Embedding(d_model, vocab_size, device=device, dtype=dtype)
        
        # Positional embeddings
        self.pos_embeddings = SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = LayerNorm(d_model, device=device, dtype=dtype)
        
        # LM head
        self.lm_head = TruncNormalNoBiasLinear(d_model, vocab_size)
    
    def forward(self, in_indices: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        Apply the Transformer language model to input token indices.
        
        Args:
            in_indices (Int[Tensor, "batch seq_len"]): Token indices
            
        Returns:
            Float[Tensor, "batch seq_len vocab_size"]: Logits for each token position
        """
        batch_size, seq_len = in_indices.shape
        
        # Get token embeddings
        x = self.token_embeddings(in_indices)  # (batch, seq_len, d_model)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=in_indices.device)
        pos_embs = self.pos_embeddings(positions)  # (seq_len, d_model)
        x = x + pos_embs.unsqueeze(0)  # Broadcast to (batch, seq_len, d_model)
        
        # Apply Transformer blocks
        for layer in self.layers:
            x = layer(x, norm)
        
        # Apply final layer norm
        x = self.ln_final(x)
        
        # Apply LM head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
