import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    LayerNorm

    This module implements layer normalization, which normalizes the input across the feature dimension.
    It helps stabilize and accelerate training by ensuring that the inputs to each layer have a consistent distribution.

    Args:
    embed_dim (int): The dimensionality of the input embeddings.
    eps (float): A small value added to the denominator for numerical stability.

    Returns:
    torch.Tensor: The normalized output, which has the same shape as the input.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LayerNorm.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Normalized output with the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized_x + self.beta
