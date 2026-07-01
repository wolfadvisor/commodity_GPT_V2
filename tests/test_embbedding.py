import torch

from commodity_gpt_v2.config import GPTConfig
from commodity_gpt_v2.embeddings.token_embedding import TokenEmbedding


config = GPTConfig(
    vocab_size=20
)

embedding = TokenEmbedding(config)

tokens = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

output = embedding(tokens)

print(output.shape)

print(output)