import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlock(nn.Module):
    """SwiGLU feed-forward block with a parameter budget near a 4x MLP."""

    def __init__(
        self,
        embed_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super(FeedForwardBlock, self).__init__()
        self.embed_dim = embed_dim
        # SwiGLU has three projections instead of two. 2/3 preserves the
        # parameter budget of an expansion_factor x standard MLP.
        unaligned_dim = 2 * embed_dim * expansion_factor // 3
        self.hidden_dim = (unaligned_dim + 63) // 64 * 64
        self.gate_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.up_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(x))
