"""
    attention .py

    Implement Scaled Dot-Product Self Attention

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from commodity_gpt_v2.config import GPTConfig

class SelfAttention(nn.Module):
    """
    Implements single-head self-attention.

    Input:
        (batch, seq_len, embedding_dim)

    Output:
        (batch, seq_len, embedding_dim)
    """

    def __init__(self, config:GPTConfig):
        super().__init__()

        self.query = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.key = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.values= nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, embedding_dim)
        returns: (batch, seq_len, embedding_dim)
        """
        # project inputs to query, key, value
        q = self.query(x)   # (B, T, C)
        k = self.key(x)     # (B, T, C)
        v = self.value(x)   # (B, T, C)

        # -----------------------------
        # Step 2
        # Attention Scores
        # -----------------------------

        scores = q @ k.transpose(-2,-1)

         # -----------------------------
        # Step 3
        # Scaling
        # -----------------------------

        # scaled dot-product attention
        dk = q.size(-1)
        scores = scores / math.sqrt(dk)  # (B, T, T)
        # -----------------------------
        # Step 4
        # Probabilities
        # -----------------------------
        attn = F.softmax(scores, dim=-1)  # (B, T, T)

        out = torch.matmul(attn, v)  # (B, T, C)
        
        return out