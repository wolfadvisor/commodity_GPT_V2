"""
gpt.py

Complete CommodityGPT model.

Architecture

Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Position Embedding
      │
      ▼
Add
      │
      ▼
Transformer Stack
      │
      ▼
LayerNorm
      │
      ▼
LM Head
      │
      ▼
Vocabulary Logits

Author:
    Carlos Ribeiro

Project:
    CommodityGPT v2
"""

import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.embeddings.token_embedding import TokenEmbedding
from commodity_gpt_v2.embeddings.positional_embedding import PositionalEmbedding
from commodity_gpt_v2.layers.transformers_stack import TransformerStack

class CommodityGPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token Embeddings

        self.token_embedding = TokenEmbedding(config)

        # Position Embedding

        self.position_embedding = PositionalEmbedding(config)

        # Trasnformer Stack

        self.transformer = TransformerStack(config)

        # Final LayerNorm

        self.final_norm = nn.LayerNorm(config.embedding_dim)

        # Language Modeling Head

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def forward(
            self,
            tokens: torch.Tensor,
        )-> torch.Tensor:

        token_embeddings = self.token_embedding(tokens)

        position_embeddings = self.position_embedding(tokens)

        x = token_embeddings + position_embeddings  

        x = self.transformer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits