import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    """
    MultiHeadAttentionBlock

    This module implements the multi-head attention mechanism,
    which allows the model to attend to different parts of the input sequence simultaneously.
    It splits the input embeddings into multiple heads, computes attention for each head,
    and then concatenates the results before applying a final linear transformation.

    Args:
    embed_dim (int): The dimensionality of the input embeddings.
    num_heads (int): The number of attention heads. Must divide embed_dim evenly.
    dropout (float): The dropout rate to apply to the attention weights.

    Returns:
    torch.Tensor: The output of the multi-head attention mechanism,
    which has the same shape as the input embeddings.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
    ) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_seq_length, max_seq_length)).bool()
        )  # We use max_seq_length as a maximum sequence length for the mask, which can be adjusted as needed.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttentionBlock.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output of the multi-head attention mechanism,
            with the same shape as the input.
        """

        batch_size, seq_length, _ = x.size()
        Q = (
            self.w_q(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, head_dim)
        K = (
            self.w_k(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, head_dim)
        V = (
            self.w_v(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, head_dim)

        mask = self.mask[:seq_length, :seq_length].to(
            x.device
        )  # (seq_length, seq_length)

        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # (batch_size, num_heads, seq_length, seq_length)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_scores = (
            attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]
        )
        attention = F.softmax(attention_scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(
            attention, V
        )  # (batch_size, num_heads, seq_length, head_dim)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embed_dim)
        )  # (batch_size, seq_length, embed_dim)
        output = self.w_o(output)  # (batch_size, seq_length, embed_dim)
        return output
