import torch

from rich import print
from rich.traceback import install

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.attention.attention import SelfAttention

install()

config = GPTConfig(
    vocab_size =20
)

attention = SelfAttention(config)

x = torch.randn(
    2,
    4,
    config.embedding_dim
)

output = attention(x)

print("Input: ", x.shape)
print(f'\nOutput: ',output.shape)
