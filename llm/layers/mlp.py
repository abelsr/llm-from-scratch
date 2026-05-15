import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super(FeedForwardBlock, self).__init__()
        self.embed_dim = embed_dim
        self.expanded_dim = embed_dim * expansion_factor
        self.fc1 = nn.Linear(embed_dim, self.expanded_dim)
        self.fc2 = nn.Linear(self.expanded_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'gelu'.")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
