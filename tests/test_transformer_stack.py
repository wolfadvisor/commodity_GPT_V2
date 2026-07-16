from rich import print
from rich.traceback import install

import torch

from commodity_gpt_v2.config import GPTConfig

from commodity_gpt_v2.layers.transformers_stack import TransformerStack

config = GPTConfig(
    vocab_size=100,
    num_layers=12
)

stack = TransformerStack(config)

x= torch.randn(
    2,8,config.embedding_dim
)

output = stack(x)

print("Input: ", x.shape)
print('Output: ', output.shape)