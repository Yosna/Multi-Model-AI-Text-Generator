"""Bigram language model implementation for character-level prediction."""

from typing import Any, cast

import torch
import torch.nn as nn

from models.base_model import BaseLanguageModel


class BigramLanguageModel(BaseLanguageModel):
    """A bigram language model that predicts the next character in a sequence.

    Architecture:
        - Single embedding layer that directly maps characters to their
          next character probabilities
        - No memory of previous context beyond the current character

    This is a simple baseline model that only considers the previous character
    when making predictions.

    Attributes:
        batch_size (int): Batch size for training and inference.
        block_size (int): Length of input sequences.
        lr (float): Learning rate.
        embedding (nn.Embedding): Embedding layer for character tokens.
        config["hparams"] keys: config.json hparams attributes.
            (type-hinted above __init__)

    Notes:
        Keys in config["hparams"] are attributes set dynamically at initialization.
    """

    batch_size: int
    block_size: int
    lr: float

    def __init__(self, config: dict[str, Any], cfg_path: str) -> None:
        """Initialize the bigram model and its parameters.

        Args:
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.

        Raises:
            ValueError: If vocab_size is not set or is too large for the model.

        Notes:
            config["hparams"] keys are set as attributes on the model instance.
        """
        name = "bigram"
        super().__init__(model_name=name, config=config, cfg_path=cfg_path)

        vocab_size = cast(int, self.vocab_size)

        if not self.vocab_size:
            raise ValueError("Vocab size is not set for Bigram model")
        elif vocab_size > 10000:
            raise ValueError(
                f"Attempted to set vocab_size to {vocab_size}.\n"
                "Bigram model is not suitable for large vocab sizes.\n"
                "If you're using word-level tokenization, consider using a\n"
                "different model or switching to character-level tokenization."
            )

        # Each character gets a vector of size vocab_size
        # Character predictions are learned via probability distribution
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def __repr__(self) -> str:
        """Displays a string representation of the model.

        Returns:
            str: String representation of the model.
        """
        output = f"BigramLanguageModel(\n\tvocab_size={self.vocab_size}\n)"
        return output.expandtabs(4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for input indices.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).

        Returns:
            torch.Tensor: Model predictions of shape (B, T, vocab_size)
        """
        # (B, T, vocab_size): map indices to logits for next character prediction
        logits = self.embedding(idx)
        return logits
