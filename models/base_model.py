import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Any


class BaseLanguageModel(nn.Module):
    """
    Base class for all language models in the project.

    This abstract base class provides common functionality for language models:
    - Device management (CPU/GPU)
    - Checkpoint handling
    - Loss computation
    - Token generation

    All specific language model implementations should inherit from this class.
    """

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any],
        cfg_path: str = "config.json",
        vocab_size: int | None = None,
    ) -> None:
        """
        Initialize the base language model.

        Args:
            model_name: Name of the model, used for checkpoint paths
            vocab_size: Number of unique tokens, or none for transformer models
        """
        super().__init__()
        self.name = model_name
        self.training = config.get("training", False)
        self.batch_size = config.get("batch_size", 0)
        self.block_size = config.get("block_size", 0)
        self.steps = config.get("steps", 0)
        self.interval = config.get("interval", 0)
        self.lr = config.get("lr", 0)
        self.patience = config.get("patience", 0)
        self.max_new_tokens = config.get("max_new_tokens", 0)
        self.max_checkpoints = config.get("max_checkpoints", 0)
        self.cfg_path = cfg_path
        self.dir_path = os.path.join("checkpoints", model_name)
        self.plot_dir = os.path.join("plots", model_name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")

        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = vocab_size

    def train_step(
        self, xb: torch.Tensor, yb: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Perform a single training step for the model.

        Args:
            xb (torch.Tensor): Input batch tensor.
            yb (torch.Tensor): Target batch tensor.
            optimizer (torch.optim.Optimizer): Optimizer to update model parameters.

        Returns:
            torch.Tensor: The loss value for the current training step.
        """
        _, loss, *_ = self(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def compute_loss(
        self,
        idx: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss: torch.Tensor | None = None,
    ):
        """
        Compute the cross entropy loss between model predictions and targets.

        Args:
            B: batch size, T: sequence length, C: vocabulary size
            idx: Input token indices of shape (B, T)
            logits: Model predictions of shape (B, T, C)
            targets: Target token indices of shape (B, T), optional
            loss: Pre-computed loss tensor, optional

        Returns:
            tuple: (logits, loss) where loss is None if no targets are provided
        """
        B, T = idx.shape

        if targets is not None:
            # Reshape for cross entropy: (B,T,C) -> (B*T,C)
            # This flattens the batch and sequence dimensions
            # A single prediction per token is received
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def new_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Generate the next token in the sequence using the model's predictions.

        Args:
            logits: Model predictions of shape (B, T, C) where:
                B: batch size
                T: sequence length
                C: vocabulary size

        Returns:
            torch.Tensor: Next token index of shape (B, 1)
        """
        # Focus on the last time step
        logits = logits[:, -1, :]
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample from the probability distribution
        next_idx = torch.multinomial(probs, num_samples=1)

        return next_idx

    def generate(self, *_, **__):
        raise NotImplementedError("Method implemented in subclasses")

    def run(self, *_, **__):
        raise NotImplementedError("Method implemented in subclasses")
