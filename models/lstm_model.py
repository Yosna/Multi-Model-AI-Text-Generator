import torch
import torch.nn as nn
from models.base_model import BaseLanguageModel
from utils import decode_data
from typing import Any


class LSTMLanguageModel(BaseLanguageModel):
    """
    An LSTM-based language model that predicts the next character in a sequence.

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

    def __init__(self, config: dict[str, Any], cfg_path: str, vocab_size: int) -> None:
        """
        Initialize the LSTM model and its parameters.

        Args:
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.
            vocab_size (int): Number of unique tokens in the vocabulary.
        """
        super().__init__(
            model_name="lstm",
            config=config,
            cfg_path=cfg_path,
            vocab_size=vocab_size,
        )

        # Set all hparams config keys as attributes
        for key, value in config.get("hparams", {}).items():
            setattr(self, key, value)

        # Each character gets a vector of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        # LSTM layers process the embedded sequences
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # Final layer projects LSTM output back to vocabulary size
        if not self.vocab_size:
            raise ValueError("Vocab size is not set for LSTM model")
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def __repr__(self) -> str:
        """
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

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """
        Compute logits and loss for input indices and targets.
        Also returns the hidden state for use in generation.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
            targets (torch.Tensor, optional): Target token indices of shape (B, T).
            hidden (tuple[torch.Tensor, torch.Tensor], optional):
                LSTM hidden state tuple (h_n, c_n).

        Returns:
            tuple: (logits, loss, hidden) where:
                - logits: Model predictions of shape (B, T, vocab_size)
                - loss: Cross entropy loss if targets provided, None otherwise
                - hidden: Updated LSTM hidden state
        """
        # (B, T, embedding_dim): map indices to embeddings
        x = self.embedding(idx)
        # (B, T, hidden_size): process sequence with LSTM
        out, hidden = self.lstm(x, hidden)
        # (B, T, vocab_size): project to vocabulary size
        logits = self.fc(out)

        logits, loss = self.compute_loss(idx, logits, targets)

        return logits, loss, hidden

    @torch.no_grad()
    def generate(
        self,
        start_idx: int,
        itos: dict[int, str],
    ) -> str:
        """
        Generate new text by sampling from the model's predictions.
        Uses multinomial sampling to add randomness to the output.
        Starts from a seed index and generates max_new_tokens characters.
        Maintains the LSTM hidden state to capture context across generated tokens.
        Returns the decoded string generated by the model.

        Args:
            start_idx (int): Index of the initial character to start generation.
            itos (dict[int, str]): Mapping from indices to characters.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            str: The generated text string.
        """
        self.eval()
        idx = torch.tensor([[start_idx]], dtype=torch.long, device=self.device)
        generated = torch.tensor([start_idx], dtype=torch.long, device=self.device)
        hidden = None

        for _ in range(self.max_new_tokens):
            # Get predictions and update hidden state for next step:
            logits, _, hidden = self(idx, hidden=hidden)
            next_idx = self.new_token(logits)
            # Prepare input for next iteration
            idx = next_idx
            generated = torch.cat((generated, next_idx.flatten()), dim=0)

        return decode_data(generated, itos)
