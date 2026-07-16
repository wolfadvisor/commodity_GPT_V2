import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Converts token positions into dense vectors.

    Input:
        tokens -> (batch_size, sequence_length)

    Output:
        (1, sequence_length, embedding_dim)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_seq_length,
            embedding_dim=config.embedding_dim
        )

    def forward(
        self,
        tokens: torch.Tensor
    ) -> torch.Tensor:

        # tokens shape:
        # (batch_size, sequence_length)

        seq_len = tokens.size(1)

        positions = torch.arange(
            seq_len,
            device=tokens.device
        )

        # (sequence_length, embedding_dim)
        pos_emb = self.position_embedding(
            positions
        )

        # (1, sequence_length, embedding_dim)
        pos_emb = pos_emb.unsqueeze(0)

        return pos_emb