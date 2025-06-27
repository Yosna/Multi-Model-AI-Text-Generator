"""LSTM-based language model implementation for sequence modeling."""

from typing import Any, cast

import torch
import torch.nn as nn

from models.base_model import BaseLanguageModel


class LSTMLanguageModel(BaseLanguageModel):
    """An LSTM-based language model that predicts the next character in a sequence.

    Architecture:
        - Embedding layer converts character indices to dense vectors
        - LSTM layers process sequences to capture long-range dependencies
        - Linear layer projects LSTM output back to vocabulary size

    The model can maintain state between predictions, allowing it to learn
    longer-term patterns in the text compared to simpler models.

    Attributes:
        batch_size (int): Batch size for training and inference.
        block_size (int): Length of input sequences.
        lr (float): Learning rate.
        embedding_dim (int): Dimension of embedding vectors.
        hidden_size (int): Number of features in LSTM hidden state.
        num_layers (int): Number of LSTM layers.
        embedding (nn.Embedding): Embedding layer for character tokens.
        lstm (nn.LSTM): LSTM layer(s) for sequence modeling.
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
    hidden_size: int
    num_layers: int

    def __init__(self, config: dict[str, Any], cfg_path: str) -> None:
        """Initialize the LSTM model and its parameters.

        Args:
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.

        Raises:
            ValueError: If vocab_size is not set for the model.

        Notes:
            config["hparams"] keys are set as attributes on the model instance.
        """
        name = "lstm"
        super().__init__(model_name=name, config=config, cfg_path=cfg_path)

        vocab_size = cast(int, self.vocab_size)

        # Initialize the hidden state
        self.hidden = None

        # Each character gets a vector of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        # LSTM layers process the embedded sequences
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        if not self.vocab_size:
            raise ValueError("Vocab size is not set for LSTM model")

        # Final layer projects LSTM output back to vocabulary size
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def __repr__(self) -> str:
        """Displays a string representation of the model.

        Returns:
            str: String representation of the model.
        """
        output = (
            f"LSTMLanguageModel(\n"
            f"\tvocab_size={self.vocab_size},\n"
            f"\tembedding_dim={self.embedding_dim},\n"
            f"\thidden_size={self.hidden_size},\n"
            f"\tnum_layers={self.num_layers}\n"
            f")"
        )
        return output.expandtabs(4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for input indices using LSTM.

        Updates the internal hidden state for use in generation.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).

        Returns:
            torch.Tensor: Model predictions of shape (B, T, vocab_size)
        """
        # (B, T, embedding_dim): map indices to embeddings
        x = self.embedding(idx)
        # (B, T, hidden_size): process sequence with LSTM
        out, self.hidden = self.lstm(x, self.hidden)
        # (B, T, vocab_size): project to vocabulary size
        logits = self.fc(out)

        return logits

    def train_step(self, *args, **kwargs) -> float:
        """Reset hidden state at the start of each training step.

        Please see the base class for more information.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: The loss value for the current training step.
        """
        self.hidden = None
        return super().train_step(*args, **kwargs)
