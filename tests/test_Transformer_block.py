from rich import print
from rich.traceback import install

import torch

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.layers.transformer_block import TransformerBlock


config = GPTConfig(
    vocab_size=100
)

block = TransformerBlock(config)

x = torch.randn(
    2,
    8,
    config.embedding_dim
)

output = block(x)

print("Input:", x.shape)
print("Output: ", output.shape)
