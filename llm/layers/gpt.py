import torch
import torch.nn as nn

from .attention import MultiHeadAttentionBlock
from .mlp import FeedForwardBlock
from .norm import LayerNorm


class GPTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_expansion_factor: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
    ):
        super(GPTBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_expansion_factor = mlp_expansion_factor
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.attention = MultiHeadAttentionBlock(
            embed_dim, num_heads, dropout, max_seq_length
        )
        self.norm1 = LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(embed_dim, mlp_expansion_factor, dropout)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))  # Residual connection around attention
        x = x + self.mlp(self.norm2(x))  # Residual connection around MLP
        return x
