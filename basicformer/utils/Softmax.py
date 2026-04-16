import torch
from torch import nn

class Softmax(nn.Module):
    """
    A softmax layer that applies the softmax function to the input.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """
        Given the input of a Softmax layer, compute the output of the Softmax layer.

        Args:
            x (Float[Tensor, "..."]): The input tensor to the Softmax layer.

        Returns:
            Float[Tensor, "..."]: The output tensor of the Softmax layer.
        """
        max_x = x.max(dim=dim, keepdim=True).values
        return torch.exp(x - max_x) / torch.exp(x - max_x).sum(dim=dim, keepdim=True)