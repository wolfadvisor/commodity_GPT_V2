import torch
import torch.nn as nn
from commodity_gpt_v2.config import GPTConfig

class CommodityGPT(nn.Module):

    def __init__(
        self,
        config: GPTConfig
    ):
        super().__init__()

        # ==========================
        # Token Embedding
        # ==========================

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )

        # ==========================
        # Position Embedding
        # ==========================

        self.position_embedding = nn.Embedding(
            config.max_seq_length,
            config.embedding_dim
        )

        # ==========================
        # Transformer
        # ==========================

        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=1,
            dim_feedforward=config.hidden_dim,
            batch_first=True
        )

        # ==========================
        # LM Head
        # ==========================

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size
        )

    def forward(self, tokens):

        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device=tokens.device
        )

        pos_emb = self.position_embedding(
            positions
        )

        token_emb = self.embedding(
            tokens
        )

        x = token_emb + pos_emb

        # -------------------------
        # Causal Mask
        # -------------------------

        mask = torch.triu(
            torch.ones(
            seq_len,
            seq_len,
            device= tokens.device
        ), 
        diagonal=1
        ).bool()

        x = self.transformer(x, src_mask = mask)

        logits = self.lm_head(x)

        return logits