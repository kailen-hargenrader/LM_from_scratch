# Custom Linear and Embedding modules

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from typing import Optional

class TruncNormalNoBiasLinear(nn.Module):
    """
    A Linear layer with truncated normal initialization and no bias.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        trunc_scale = (2.0 / (d_in + d_out)) ** 0.5
        self.weight = nn.Parameter(
            trunc_normal_(
                torch.empty(d_out, d_in), 
                mean=0.0, 
                std=trunc_scale, 
                a = -3.0 * trunc_scale, 
                b = 3.0 * trunc_scale
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of a Linear layer, compute the transformation of a batched input.

        Args:
            x (Float[Tensor, "... d_in"]): The input tensor to the linear layer.

        Returns:
            Float[Tensor, "... d_out"]: The output tensor of the linear layer.
        """
        self.weight = self.weight.to(x.device)
        return x @ self.weight.T


class PositionwiseFeedForward(nn.Module):
    """
    A Positionwise FeedForward layer that applies a 2-layer ReLU FFN to the input.
    """
    def __init__(self, d_model: int, d_ff: int = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = TruncNormalNoBiasLinear(d_model, d_ff)
        self.fc2 = TruncNormalNoBiasLinear(d_ff, d_model)
        self.device = device
        self.dtype = dtype
    
    def _relu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input of a ReLU layer, compute the output of the ReLU layer.
        """
        return x.clamp_min(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of a Positionwise FeedForward layer, compute the output of the FFN.

        Args:
            x (Float[Tensor, "... d_model"]): The input tensor to the FFN.

        Returns:
            Float[Tensor, "... d_model"]: The output tensor of the FFN.
        """
        xdevice = x.device
        xdtype = x.dtype
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.fc1(x)
        x = self._relu(x)
        x = self.fc2(x)
        return x.to(device=xdevice, dtype=xdtype)

class Embedding(nn.Module):
    """
    An embedding layer that gets the vector for a single token id.
    """
    def __init__(self, d_model: int, vocab_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.embedding_table = nn.Parameter(
            trunc_normal_(
                torch.empty(vocab_size, d_model, device=device, dtype=dtype), 
                mean=0.0, 
                std=1.0, 
                a=-3.0, 
                b=3.0
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
        
        Args:
            x (Int[Tensor, "..."]): The token ids to get the embeddings for.

        Returns:
            Float[Tensor, " ... d_model"]: The embeddings for the token ids.
        """
        
        return self.embedding_table[x.to(self.embedding_table.device)].to(x.device)

if __name__ == "__main__":
    d_in = 10
    d_out = 20
    x = torch.randn(1, d_in)
    linear = TruncNormalNoBiasLinear(d_in, d_out)
    print(f"Linear input shape: {x.shape}")
    print(f"Linear weight shape: {linear.weight.shape}")
    print(f"Linear output shape: {linear(x).shape}")