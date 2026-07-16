from rich import print
from rich.traceback import install

import torch

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.model.gpt import CommodityGPT

install()

config = GPTConfig(
    vocab_size=100,
    embedding_dim =128,
    hidden_dim= 512,
    num_heads=4,
    num_layers=6,
    max_seq_length=64
)

model = CommodityGPT(config)

tokens = torch.randint(
    0,
    config.vocab_size,
    (2,10)
)

output = model(tokens)

print("Input :", tokens.shape)
print("Output:", output.shape)