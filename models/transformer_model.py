"""Transformer-based language model implementation for advanced sequence modeling."""

from typing import Any

import torch
import torch.nn as nn

from models.base_model import BaseLanguageModel
from models.components.generators import Generators


class TransformerLanguageModel(BaseLanguageModel):
    """A Transformer-based language model for sequence prediction.

    Architecture:
        - Token embedding layer maps input indices to dense vectors
        - Position embedding layer encodes sequence order information
        - Stacked Transformer encoder layers process the sequence
        - Final linear layer projects encoder output to vocabulary size

    Attributes:
        batch_size (int): Batch size for training and inference.
        block_size (int): Length of input sequences.
        lr (float): Learning rate.
        embedding_dim (int): Dimension of embedding vectors.
        max_seq_len (int): Maximum sequence length for position embeddings.
        num_heads (int): Number of attention heads in each encoder layer.
        ff_dim (int): Dimension of feedforward network in encoder layers.
        num_layers (int): Number of Transformer encoder layers.
        token_embedding (nn.Embedding): Embedding layer for tokens.
        position_embedding (nn.Embedding): Embedding layer for positions.
        transformer (nn.TransformerEncoder): Stacked Transformer encoder.
        fc (nn.Linear): Final linear layer for output projection.
        config["hparams"] keys: config.json hparams attributes.
            (type-hinted above __init__)

    Notes:
        Keys in config["hparams"] are attributes set dynamically at initialization.
    """

    batch_size: int
    block_size: int
    lr: float
    embedding_dim: int
    max_seq_len: int
    num_heads: int
    ff_dim: int
    num_layers: int

    def __init__(
        self,
        config: dict[str, Any],
        cfg_path: str,
        vocab_size: int,
        model_options: dict[str, Any],
    ) -> None:
        """Initialize the Transformer model and its parameters.

        Args:
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.
            vocab_size (int): Number of unique tokens in the vocabulary.
            model_options (dict[str, Any]): Model options.

        Raises:
            ValueError: If vocab_size is not set for the model.

        Notes:
            config["hparams"] keys are set as attributes on the model instance.
        """
        super().__init__(
            model_name="transformer",
            config=config,
            cfg_path=cfg_path,
            vocab_size=vocab_size,
            model_options=model_options,
        )

        # Set all hparams config keys as attributes
        for key, value in config.get("hparams", {}).items():
            setattr(self, key, value)

        # Embedding layers for tokens and positions
        self.token_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        if not self.vocab_size:
            raise ValueError("Vocab size is not set for Transformer model")

        # Final linear layer projects to vocabulary size
        self.fc = nn.Linear(self.embedding_dim, vocab_size)

    def __repr__(self) -> str:
        """Displays a string representation of the model.

        Returns:
            str: String representation of the model.
        """
        output = (
            f"TransformerLanguageModel(\n"
            f"\tvocab_size={self.vocab_size},\n"
            f"\tembedding_dim={self.embedding_dim},\n"
            f"\tmax_seq_len={self.max_seq_len},\n"
            f"\tnum_heads={self.num_heads},\n"
            f"\tff_dim={self.ff_dim},\n"
            f"\tnum_layers={self.num_layers}\n"
            f")"
        )
        return output.expandtabs(4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for input indices using the Transformer encoder.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).

        Returns:
            torch.Tensor: Model predictions of shape (B, T, vocab_size)
        """
        # (B, T, embedding_dim): map indices to embeddings
        x = self.token_embedding(idx)
        B, T = x.shape[:2]

        # (T, embedding_dim): position embeddings for each position in the sequence
        positions = self.position_embedding(torch.arange(T, device=x.device))
        # (B, T, embedding_dim): expand position embeddings to batch size
        positions = positions.unsqueeze(0).expand(B, -1, -1)

        # (T, T): attention mask to prevent attending to future positions
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # (B, T, embedding_dim): sum token and position embeddings
        out = self.transformer(x + positions, mask=attn_mask)
        # (B, T, vocab_size): project encoder output to vocabulary size
        logits = self.fc(out)

        return logits

    @torch.no_grad()
    def generate(self, stoi: dict[str, int], itos: dict[int, str]) -> str:
        """Generate new text by sampling from the model's predictions.

        Multinomial sampling is used to add randomness to the output.
        Starts from a seed index and generates max_new_tokens characters.

        Args:
            stoi (dict[str, int]): Mapping from characters to token indices.
            itos (dict[int, str]): Mapping from indices to characters.

        Returns:
            str: The decoded string generated by the model.
        """
        generator = Generators.Text.Random(stoi, itos, self.sampler)
        output = generator.output(self)
        return output
