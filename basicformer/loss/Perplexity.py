import torch
from torch import nn

class Perplexity(nn.Module):
    """
    A perplexity layer that computes the perplexity of the input.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Given the input of a CrossEntropy layer, compute the output of the CrossEntropy layer.

        Args:
            inputs (Float[Tensor, "batch_size ... vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.

            targets (Long[Tensor, "batch_size ..."]): Tensor of shape (batch_size, ...) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

        Returns:
            Float[Tensor, "batch_size ..."]: The output of the CrossEntropy layer.
        """
        # We assume targets are integer class indices for each sample.
        # Efficient cross-entropy for integer targets and unnormalized logits (no unnecessary log/exp).
        log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Select the log probabilities of the correct class for each sample.
        loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return torch.exp(loss.mean())