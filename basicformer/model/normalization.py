# LayerNorm implementation

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from typing import Optional

class LayerNorm(nn.Module):
    """
    A LayerNorm module that applies layer normalization to the input.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the weights of a LayerNorm layer, compute the output of running LayerNorm on the input.

        Args:
            x (Float[Tensor, " ... d_model"]): The input tensor to the LayerNorm layer.

        Returns:
            Float[Tensor, " ... d_model"]: The output tensor of the LayerNorm layer.
        """
        xdevice = x.device
        xdtype = x.dtype
        x = x.to(device=self.device, dtype=torch.float32)
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = x.to(dtype=self.dtype)
        x = x * self.weight + self.bias
        return x.to(device=xdevice, dtype=xdtype)