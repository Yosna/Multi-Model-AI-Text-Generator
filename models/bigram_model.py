import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import decode_data


class BigramLanguageModel(nn.Module):
    """
    A bigram language model that predicts the next character in a sequence.

    Architecture:
        - Single embedding layer that directly maps characters to their
          next character probabilities
        - No memory of previous context beyond the current character

    This is a simple baseline model that only considers the previous
    character when making predictions.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the bigram model and its parameters."""
        super().__init__()

        self.name = "bigram"
        self.dir_path = os.path.join("checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")
        self.cfg_path = "config.json"

        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Each character gets a vector of size vocab_size
        # Character predictions are learned via probability distribution
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"BigramLanguageModel(vocab_size={self.vocab_size})"

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and loss for input indices and targets."""
        # idx and targets are both (B,T) tensor of integers
        # B = batch size, T = sequence length
        B, T = idx.shape
        # (B, T, vocab_size): map indices to logits for next character prediction
        logits = self.embedding(idx)
        loss = None

        if targets is not None:
            # Reshape for cross entropy: (B,T,C) -> (B*T,C)
            # This flattens the batch and sequence dimensions
            # A single prediction per token is received
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

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
        """
        self.eval()
        idx = torch.tensor([[start_idx]], dtype=torch.long, device=self.device)
        generated = [start_idx]

        for _ in range(max_new_tokens):
            # Get predictions for next character
            logits, _ = self(idx)
            # Focus on the last time step
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution
            next_idx = torch.multinomial(probs, num_samples=1)
            generated.append(next_idx.item())

        return decode_data(generated, itos)
