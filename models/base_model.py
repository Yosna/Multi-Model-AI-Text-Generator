"""Base class and shared utilities for all language models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Any


class BaseLanguageModel(nn.Module):
    """Base class for all language models in the project.

    This abstract base class provides common functionality for language models:
    - Device management (CPU/GPU)
    - Checkpoint handling
    - Loss computation
    - Token generation

    Attributes:
        name (str): Model name.
        cfg_path (str): Path to config file.
        vocab_size (int | None): Vocabulary size.
        dir_path (str): Checkpoint directory.
        plot_dir (str): Plot directory.
        ckpt_dir (str): Current checkpoint directory.
        ckpt_path (str): Path to checkpoint file.
        meta_path (str): Path to metadata file.
        device (torch.device): Device used for computation.
        config["runtime"] keys: config.json runtime attributes
            (type-hinted above __init__)

    Notes:
        All specific language model implementations should inherit from this class.
        Keys in config["runtime"] are attributes set dynamically at initialization.
    """

    training: bool
    steps: int
    interval: int
    patience: int
    max_new_tokens: int
    max_checkpoints: int

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any],
        cfg_path: str = "config.json",
        vocab_size: int | None = None,
        token_level: str = "char",
    ) -> None:
        """Initialize the base language model.

        Args:
            model_name (str): Name of the model, used for checkpoint paths.
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.
            vocab_size (int | None): Number of unique tokens.
            token_level (str): Token level to use for vocabulary building.
                Options: "char" (default), or "word"

        Notes:
            config["runtime"] keys are set as attributes on the model instance.
        """
        super().__init__()
        self.name: str = model_name
        self.cfg_path: str = cfg_path
        self.vocab_size: int | None = vocab_size
        self.token_level: str = token_level
        self.dir_path: str = os.path.join("checkpoints", model_name)
        self.plot_dir: str = os.path.join("plots", model_name)
        self.ckpt_dir: str = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path: str = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path: str = os.path.join(self.ckpt_dir, "metadata.json")

        # Set all runtime config keys as attributes
        for key, value in config.get("runtime", {}).items():
            setattr(self, key, value)

        # Automatically use GPU if available, otherwise CPU
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def train_step(
        self, xb: torch.Tensor, yb: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform a single training step for the model.

        Args:
            xb (torch.Tensor): Input batch tensor.
            yb (torch.Tensor): Target batch tensor.
            optimizer (torch.optim.Optimizer): Optimizer to update model parameters.

        Returns:
            float: The loss value for the current training step.
        """
        logits = self(xb)
        loss = self.compute_loss(logits, xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def compute_loss(
        self,
        logits: torch.Tensor,
        idx: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cross entropy loss between model predictions and targets.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C)
            idx (torch.Tensor): Input token indices of shape (B, T)
            targets (torch.Tensor): Target token indices of shape (B, T)

        Returns:
            torch.Tensor: Cross entropy loss between predictions and targets
        """
        B, T = idx.shape

        # Reshape for cross entropy: (B,T,C) -> (B*T,C)
        # This flattens the batch and sequence dimensions
        # A single prediction per token is received
        logits = logits.view(B * T, -1)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return loss

    def check_patience(
        self, best_loss: float, val_loss: float, wait: int
    ) -> tuple[bool, float, int]:
        """Check if model has stopped improving and should stop training.

        Monitors validation loss improvement and implements early stopping based on
        the model's patience threshold. If validation loss doesn't improve for
        patience number of steps, training should stop.

        Args:
            best_loss: Best validation loss seen so far
            val_loss: Current validation loss
            wait: Number of steps without improvement

        Returns:
            Tuple containing:
            - overfit (bool): Whether training should stop due to overfitting
            - best_loss (float): Updated best loss value
            - wait (int): Updated wait counter
        """
        overfit = False
        loss_improved = val_loss < best_loss
        if loss_improved:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= self.patience:
                overfit = True
                print("Stopping due to overfitting.")
                print(f"Best Loss this training session: {best_loss}")
        return overfit, best_loss, wait

    def new_token(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate the next token in the sequence using the model's predictions.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C)
                - B: batch size
                - T: sequence length
                - C: vocabulary size
            temperature (float): Temperature for sampling (default: 1.0)

        Returns:
            torch.Tensor: Next token index of shape (B, 1)
        """
        # Focus on the last time step
        logits = logits[:, -1, :]
        # Convert logits to probabilities
        probs = F.softmax(logits / temperature, dim=-1)
        # Sample from the probability distribution
        next_idx = torch.multinomial(probs, num_samples=1)
        return next_idx

    def generate(self, *_, **__):
        """Generate text from the model.

        This method must be implemented by subclasses.

        Args:
            *_: Unused arguments
            **__: Unused keyword arguments

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Method implemented in subclasses")

    def run(self, *_, **__):
        """Run the model on input data.

        This method must be implemented by subclasses.

        Args:
            *_: Unused arguments
            **__: Unused keyword arguments

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Method implemented in subclasses")
