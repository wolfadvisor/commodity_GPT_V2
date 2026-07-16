"""
causal_mask.py

Implements masking utilities used by Transformer attention.

Currently supports:

- Causal (autoregressive) masking used by GPT.

Future extensions:

- Bidirectional masking (BERT)
- Padding masks
- Encoder-Decoder masks

Author:
    Carlos Ribeiro

Project:
    CommodityGPT v2
"""

import torch

class CasualMask:
     """
    Creates lower triangular causal masks.

    A causal mask prevents each token from attending
    to future tokens.

    Example:

        Sequence Length = 4

            T F F F
            T T F F
            T T T F
            T T T T
    """
     
     @staticmethod
     def create(
          seq_len : int,
          device: torch.device
     )-> torch.Tensor:
        """
        Creates a lower triangular causal mask.

        Args:
            seq_len:
                Sequence length.

            device:
                CPU or CUDA device.

        Returns
        -------
        Tensor

        Shape:
            (seq_len, seq_len)

        Type:
            bool
        """  

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool
            )
        )
        return mask
