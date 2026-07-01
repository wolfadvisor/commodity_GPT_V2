import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig

class PositionalEmbbedding(nn.Module):
     """
        Learnable positional embeddings.

        Converts token positions into dense vectors.

        Input:
            seq_len

        Output:
            (1, seq_len, embedding_dim)
    """
     def __init__(self, config :GPTConfig):
          super().__init__()

          self.position_embedding = nn.Embedding(
               num_embeddings= config.max_seq_length,
               embedding_dim= config.embedding_dim
          )

     def forward(
               self,
               seq_len : int,
               device : torch.device
     )-> torch.Tensor:
          
          positions = torch.arange(
               seq_len,
               device=device
          ).unsqueeze(0)

          return self.position_embedding(positions)