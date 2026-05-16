import torch
import torch.nn as nn

from .layers.gpt import GPTBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
    ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_expansion = mlp_expansion
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                GPTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_expansion_factor=mlp_expansion,
                    max_seq_length=max_seq_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input of the model is a tensor of shape (batch_size, seq_length) containing token indices.
        batch_size, seq_length = x.size()

        # Create position indices and get token and position embeddings.
        position_indices = torch.arange(seq_length, device=x.device).unsqueeze(
            0
        )  # (1, seq_length)

        # Get token and position embeddings, and sum them to get the input to the transformer blocks.
        token_embeds = self.token_embedding(x)  # (batch_size, seq_length, embed_dim)
        position_embeds = self.position_embedding(
            position_indices
        )  # (1, seq_length, embed_dim)
        x = token_embeds
        x = x + position_embeds  # (batch_size, seq_length, embed_dim)
        x = self.embed_dropout(x)

        # Pass through the transformer blocks.
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_length, embed_dim)

        # Final layer normalization and output projection to vocabulary size.
        x = self.ln_f(x)  # (batch_size, seq_length, embed_dim)
        logits = self.head(x)  # (batch_size, seq_length, vocab_size)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        top_k: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new tokens given an input sequence of token indices.

        Args:
            input_ids (torch.Tensor): Input token indices of shape (batch_size, seq_length).
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Generated token indices of shape (batch_size, seq_length + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            input_ids = input_ids[
                :, -self.max_seq_length :
            ]  # Ensure input does not exceed max_seq_length
            logits = self.forward(input_ids)  # (batch_size, seq_length, vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[
                    next_token_logits < top_k_values[:, -1, None]
                ] = -float("Inf")

            next_token_probs = torch.softmax(
                next_token_logits, dim=-1
            )  # (batch_size, vocab_size)
            next_token = torch.multinomial(next_token_probs, num_samples=1)  #

            input_ids = torch.cat(
                (input_ids, next_token), dim=1
            )  # (batch_size, seq_length + 1)
        return input_ids
