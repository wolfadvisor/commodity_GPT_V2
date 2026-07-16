"""
transformer_stack.py

Stacks multiple Transformer Blocks.

Author:
    Carlos Ribeiro

Project:
    CommodityGPT v2
"""

import torch
import torch.nn as nn 

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.layers.transformer_block import TransformerBlock

class TransformerStack(nn.Module):
     """
    Stacks multiple Transformer Blocks.

    Input:
        (batch_size, sequence_length, embedding_dim)

    Output:
        (batch_size, sequence_length, embedding_dim)
    """
     
     def __init__(self, config: GPTConfig):
          super().__init__()

          self.blocks = nn.ModuleList(
               [
                    TransformerBlock(config)
                    for _ in range (config.num_layers)
               ]
          )

     def forward(self,x:torch.Tensor)-> torch.Tensor:
          
          for block in self.blocks:
               
               x = block(x)
               
          return x
        
     