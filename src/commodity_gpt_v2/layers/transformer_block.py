"""
transformer_block.py

Implements a complete GPT Transformer Block.

Architecture:

Input
   │
   ▼
LayerNorm
   │
   ▼
Multi-Head Attention
   │
   ▼
Residual Connection
   │
   ▼
LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Connection
   │
   ▼
Output

Author:
    Carlos Ribeiro

Project:
    CommodityGPT v2
"""

import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.attention.multi_head_attention import MultiHeadAttention
from commodity_gpt_v2.layers.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """
    Implements one Transformer Block.

    Input Shape:
        (batch_size, sequence_length, embedding_dim)

    Output Shape:
        (batch_size, sequence_length, embedding_dim)
    """
    def __init__(self, config:GPTConfig):
        super().__init__()

        #First layer Normalization

        self.norm = nn.LayerNorm(
            config.embedding_dim
        )

        #Multi-Head Self Attention

        self.attention= MultiHeadAttention(config)

        #Second Layer Normalization
        self.norm2 = nn.LayerNorm(
            config.embedding_dim
        )

        #Feed Foward Network
        self.feed_forward = FeedForward(
            config
        )

    def forward(
            self,
            x: torch.Tensor
    )-> torch.Tensor:
         """
        Forward pass.

        Processing:

        Input
            │
            ▼
        LayerNorm
            │
            ▼
        Attention
            │
            ▼
        Residual Add
            │
            ▼
        LayerNorm
            │
            ▼
        FeedForward
            │
            ▼
        Residual Add
            │
            ▼
        Output
        """
         
         #Attention Block

         residual= x 
         x =self.norm(x)
         x =self.attention(x)
         x =residual + x

         #Feed Forward Block

         residual = x
         x =self.norm2(x)
         x =self.feed_forward(x)
         x =residual + x 

         return x