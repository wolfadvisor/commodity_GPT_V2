from dataclasses import dataclass

"""
config.py

Central configuration for CommodityGPT.

Author: Carlos Ribeiro
"""

@dataclass
class GPTConfig:
    """
    Store all model hyperparameters

    Every module receives this configuration object,
    avoiding hardcoded values throughout the project implementation
    
    """

    # ==========================
    # Vocabulary
    # ==========================

    vocab_size : int


    # ==========================
    # Model
    # ==========================

    embedding_dim: int = 128

    hidden_dim: int = 512

    num_heads : int = 4

    num_layers : int = 4

    drop_out : float = 0.1

    # ==========================
    # Context
    # ==========================

    max_seq_length: int = 128

    # ==========================
    # Training
    # ==========================

    learning_rate: float = 3e-4

    batch_size: int = 32

    epochs: int = 10


