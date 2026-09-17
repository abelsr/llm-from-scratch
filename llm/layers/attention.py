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
        num_kv_heads: int | None = None,
    ) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even to use RoPE")
        self.num_groups = self.num_heads // self.num_kv_heads
        self.q_dim = num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.proj_qkv = nn.Linear(embed_dim, self.q_dim + 2 * self.kv_dim)

        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(max_seq_length, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer(
            "rope_cos", torch.repeat_interleave(angles.cos(), 2, dim=-1), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.repeat_interleave(angles.sin(), 2, dim=-1), persistent=False
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)

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
        qkv = self.proj_qkv(x)
        Q, K, V = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if seq_length > self.rope_cos.size(0):
            raise ValueError(
                f"Sequence length {seq_length} exceeds RoPE limit "
                f"{self.rope_cos.size(0)}"
            )
        cos = self.rope_cos[:seq_length].to(dtype=Q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:seq_length].to(dtype=Q.dtype).unsqueeze(0).unsqueeze(0)
        Q = Q * cos + self._rotate_half(Q) * sin
        K = K * cos + self._rotate_half(K) * sin
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

        output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embed_dim)
        )  # (batch_size, seq_length, embed_dim)
        output = self.w_o(output)  # (batch_size, seq_length, embed_dim)
        return output
