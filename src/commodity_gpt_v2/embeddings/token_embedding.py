import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig

class TokenEmbedding(nn.Module):
     """
    Converts token IDs into dense vectors.

    Input:
        (batch_size, seq_len)

    Output:
        (batch_size, seq_len, embedding_dim)
    """
     
     def __init__(self, config: GPTConfig):
          super().__init__()
          self.embbeding = nn.Embedding(
               num_embeddings= config.vocab_size,
               embedding_dim= config.embedding_dim
          )

     def forward(
        self,
        tokens : torch.Tensor
     )-> torch.Tensor:
          return self.embbeding(tokens)