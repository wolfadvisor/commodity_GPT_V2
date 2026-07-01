import torch 
from rich import print
from rich.traceback import install

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.embeddings.positional_embedding import PositionalEmbbedding

install()

config = GPTConfig(
    vocab_size=100
)

position = PositionalEmbbedding(config)

output = position(
    seq_len= 5,
    device= torch.device("cpu")
)

print(output.shape)
print(output)