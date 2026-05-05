import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    AttentionBlock
    
    This module implements **ONLY** the attention mechanism, 
    without the feedforward network or layer normalization. 
    It computes the attention scores and applies them to the input embeddings.
    
    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        
    Returns:
        torch.Tensor: The output of the attention mechanism, 
        which has the same shape as the input embeddings.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.
        
        Args:
            x (torch.Tensor): Input embeddings of 
            shape (batch_size, seq_length, embed_dim).
            
        Returns:
            torch.Tensor: Output of the attention mechanism, 
            with the same shape as the input.
        """
        Q = self.w_q(x)  # (batch_size, seq_length, embed_dim)
        K = self.w_k(x)  # (batch_size, seq_length, embed_dim)
        V = self.w_v(x)  # (batch_size, seq_length, embed_dim)
        
        # Causal Attention Mask
        T = x.size(1) # seq_length
        mask = self.mask[:T, :T]  # type: ignore # (seq_length, seq_length)
        mask = mask.to(x.device)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, seq_length, seq_length)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]
        attention = F.softmax(attention_scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)  # (batch_size, seq_length, embed_dim)
        output = self.w_o(output)  # (batch_size, seq_length, embed_dim)
        return output