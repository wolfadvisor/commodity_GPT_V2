import torch
import torch.nn as nn
from commodity_gpt_v2.config import GPTConfig

"""
feed_forward.py

Implements the Position-wise Feed-Forward Network (FFN) used inside
each Transformer block.

The FeedForward network processes every token independently after the
attention mechanism. It expands the embedding dimension to a larger
hidden space, applies a non-linear activation (GELU), projects the
representation back to the original embedding size, and applies dropout
for regularization.

Architecture:

    Embedding
         │
         ▼
    Linear (Expand)
         │
         ▼
        GELU
         │
         ▼
    Linear (Project Back)
         │
         ▼
      Dropout

The residual connection and Layer Normalization are intentionally
NOT implemented here. They belong to the TransformerBlock,
following the Pre-LayerNorm architecture used by GPT-2 and
most modern decoder-only Transformers.

Author:
    Carlos Ribeiro

Project:
    CommodityGPT v2
"""

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This module applies two fully connected layers with a GELU activation
    in between. The first layer expands the embedding dimension into a
    larger hidden representation, allowing the model to learn richer
    feature interactions. The second layer projects the representation
    back to the original embedding dimension.

    Input Shape:
        (batch_size, sequence_length, embedding_dim)

    Output Shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, config: GPTConfig):
        """
        Initializes the FeedForward network.

        Args:
            config (GPTConfig):
                Model configuration containing:

                - embedding_dim
                - hidden_dim
                - dropout
        """

        super().__init__()

        # ----------------------------------------
        # First Linear Projection
        # Expands feature space.
        #
        # Example:
        # embedding_dim → hidden_dim
        # 128           → 512
        # ----------------------------------------
        self.fc1 = nn.Linear(
            config.embedding_dim,
            config.hidden_dim
        )

        # ----------------------------------------
        # GELU Activation
        #
        # GPT models use GELU instead of ReLU
        # because it provides smoother gradients.
        # ----------------------------------------
        self.activation = nn.GELU()

        # ----------------------------------------
        # Second Linear Projection
        #
        # Projects back to the embedding dimension.
        #
        # Example:
        #
        # 512 → 128
        # ----------------------------------------
        self.fc2 = nn.Linear(
            config.hidden_dim,
            config.embedding_dim
        )

        # ----------------------------------------
        # Dropout
        #
        # Helps reduce overfitting during training.
        # Automatically disabled during evaluation.
        # ----------------------------------------
        self.drop_out = nn.Dropout(
            config.drop_out
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes the FeedForward computation.

        Processing pipeline:

            Input
              │
              ▼
          Linear (Expand)
              │
              ▼
             GELU
              │
              ▼
         Linear (Project)
              │
              ▼
           Dropout

        Args:
            x (torch.Tensor):
                Input tensor.

                Shape:
                    (batch_size, sequence_length, embedding_dim)

        Returns:
            torch.Tensor:
                Processed tensor with the same shape as the input.

                Shape:
                    (batch_size, sequence_length, embedding_dim)
        """

        # Expand representation
        x = self.fc1(x)

        # Apply non-linearity
        x = self.activation(x)

        # Project back to embedding size
        x = self.fc2(x)

        # Regularization
        x = self.drop_out(x)

        return x

    
