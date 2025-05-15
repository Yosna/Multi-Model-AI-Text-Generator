import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import decode_data


class LSTMLanguageModel(nn.Module):
    """
    An LSTM-based language model that predicts the next character in a sequence.

    Architecture:
        - Embedding layer converts character indices to dense vectors
        - LSTM layers process sequences to capture long-range dependencies
        - Linear layer projects LSTM output back to vocabulary size

    The model can maintain state between predictions, allowing it to learn
    longer-term patterns in the text compared to simpler models.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int
    ) -> None:
        """Initialize the LSTM model and its parameters."""
        super().__init__()
        self.name = "lstm"
        self.dir_path = os.path.join("checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Each character gets a vector of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layers process the embedded sequences
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Final layer projects LSTM output back to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return (
            f"LSTMLanguageModel(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers})"
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Compute logits and loss for input indices and targets.
        Also returns the hidden state for use in generation.
        """
        # idx and targets are both (B,T) tensor of integers
        # B = batch size, T = sequence length
        B, T = idx.shape
        # (B, T, embedding_dim): map indices to embeddings
        x = self.embedding(idx)
        # (B, T, hidden_size): process sequence with LSTM
        out, hidden = self.lstm(x, hidden)
        # (B, T, vocab_size): project to vocabulary size
        logits = self.fc(out)
        loss = None

        if targets is not None:
            # Reshape for cross entropy: (B,T,C) -> (B*T,C)
            # This flattens the batch and sequence dimensions
            # A single prediction per token is received
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, hidden

    @torch.no_grad()
    def generate(
        self,
        start_idx: int,
        itos: dict[int, str],
        max_new_tokens: int,
    ) -> str:
        """
        Generate new text by sampling from the model's predictions.
        Uses multinomial sampling to add randomness to the output.
        Starts from a seed index and generates max_new_tokens characters.
        Maintains the LSTM hidden state to capture context across generated tokens.
        """
        self.eval()
        # (1, 1): batch size 1, sequence length 1
        idx = torch.tensor([[start_idx]], dtype=torch.long, device=self.device)
        generated = [start_idx]
        hidden = None

        for _ in range(max_new_tokens):
            # Get predictions and update hidden state for next step
            logits, _, hidden = self(idx, hidden=hidden)
            # Focus on the last time step
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution
            next_idx = torch.multinomial(probs, num_samples=1)
            # Prepare input for next iteration
            idx = next_idx
            generated.append(next_idx.item())

        return decode_data(generated, itos)
