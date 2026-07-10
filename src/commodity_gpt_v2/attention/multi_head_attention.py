import torch
import torch.nn as nn

from commodity_gpt_v2.config import GPTConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()

        assert( config.embedding_dim % config.num_heads == 0), "embedding dimension must be divisible by number of heads"

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = (config.embedding_dim // config.num_heads)

        #Linear projections 

        self.query = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.key = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.value = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.output = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

    def forward(self,x):
        batch_size , seq_len, _ = x.shape

        #Projectiion to Q,K,V

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #split into heads

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        #Move heads before sequence, tranpose the matrix value

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # (Attention computation will be added next), to not lose integrity

        #temporary output

        out = V

        #Restore dimensions

        out = out.transpose(1,2)

        out = out.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        return self.output(out)
