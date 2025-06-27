"""Base class and shared utilities for all language models."""

import json
import logging
import os
import shutil
import time
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.generators import Generators, Sampler, Samplers
from utils.io_utils import get_config

logger = logging.getLogger(__name__)


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
        self, model_name: str, config: dict[str, Any], cfg_path: str = "config.json"
    ) -> None:
        """Initialize the base language model.

        Args:
            model_name (str): Name of the model, used for checkpoint paths.
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.

        Notes:
            config["runtime"] keys are set as attributes on the model instance.
        """
        super().__init__()
        vocab = config.get("vocab", {})
        generator_options = get_config(cfg_path, "generator_options")
        model_options = get_config(cfg_path, "model_options")
        runtime = config.get("runtime", {})
        hparams = config.get("hparams", {})

        vocab_size = vocab.get("vocab_size", None)
        logger.debug(f"Initializing {model_name} model with vocab_size={vocab_size}")

        self.name: str = model_name
        self.cfg_path: str = cfg_path
        self.dir_path: str = os.path.join("checkpoints", model_name)
        self.plot_dir: str = os.path.join("plots", model_name)
        self.ckpt_dir: str = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path: str = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path: str = os.path.join(self.ckpt_dir, "metadata.json")

        configs = [vocab, generator_options, model_options, runtime, hparams]
        ignored_keys = ["generator", "sampler"]
        # Set all config keys as attributes
        for config in configs:
            for key, value in config.items():
                if key not in ignored_keys:
                    setattr(self, key, value)
                    logger.debug(f"Set attribute: {key} = {value}")

        # Set generator options as attributes
        sampler = generator_options.get("sampler", "multinomial")
        self.sampler = self._get_sampler(sampler)
        generator = generator_options.get("generator", "random")
        self.generator = self._get_generator(generator)

        logger.debug(
            f"Model options: sampler={self.sampler}, save_model={self.save_model}, "
            f"temperature={self.temperature}, token_level={self.token_level}"
        )

        # Automatically use GPU if available, otherwise CPU
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Model will use device: {self.device}")

    def _get_sampler(self, name: str) -> Sampler:
        """Get the sampler class based on the name.

        Args:
            name (str): Name of the sampler.

        Returns:
            Sampler: The sampler class.

        Raises:
            ValueError: If the sampler is not found.
        """
        if name.lower() == "multinomial":
            temperature = cast(float, self.temperature)
            sampler = Samplers.Multinomial(temperature=temperature)
        elif name.lower() == "argmax":
            sampler = Samplers.Argmax()
        else:
            raise ValueError(f"Sampler {name} not found")
        return sampler

    def _get_generator(self, name: str):
        """Get the generator class based on the name.

        Args:
            name (str): Name of the generator.

        Returns:
            Generator: The generator class.

        Raises:
            ValueError: If the generator is not found.
        """
        stoi = cast(dict[str, int], self.stoi)
        itos = cast(dict[int, str], self.itos)
        context_length = cast(int, self.context_length)

        if name.lower() == "random":
            generator = Generators.Text.Random(stoi, itos, self.sampler)
        elif name.lower() == "prompt":
            generator = Generators.Text.Prompt(context_length, stoi, itos, self.sampler)
        else:
            raise ValueError(f"Generator {name} not found")
        return generator

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
        self, logits: torch.Tensor, idx: torch.Tensor, targets: torch.Tensor
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
            logger.debug(f"Loss improved: {best_loss:.6f} -> {val_loss:.6f}")
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            logger.warning(f"Loss did not improve. Waiting... ({wait}/{self.patience})")
            if wait >= self.patience:
                overfit = True
                logger.warning(
                    "Stopping due to overfitting. "
                    f"Best Loss this training session: {best_loss}",
                )
        return overfit, best_loss, wait

    def save_checkpoint(self, step: int, val_loss: float) -> None:
        """Save a model checkpoint and rotate older checkpoints.

        Saves the current model state and metadata, rotating out older checkpoints
        based on the model's max_checkpoints attribute. Checkpoints are stored in
        numbered directories (checkpoint_1, checkpoint_2, etc.).

        Args:
            step (int): Current training step.
            val_loss (float): Validation loss at this step.
        """
        logger.debug(f"Saving checkpoint at step {step} with val_loss {val_loss:.6f}")

        os.makedirs(self.dir_path, exist_ok=True)

        # Shift existing checkpoints up by 1 (e.g. checkpoint_1 -> checkpoint_2)
        for i in reversed(range(1, self.max_checkpoints)):
            prev_dir = os.path.join(self.dir_path, f"checkpoint_{i}")
            next_dir = os.path.join(self.dir_path, f"checkpoint_{i + 1}")

            if os.path.exists(next_dir):
                logger.debug(f"Removing old checkpoint: {next_dir}")
                shutil.rmtree(next_dir)

            if os.path.exists(prev_dir):
                logger.debug(f"Moving checkpoint: {prev_dir} -> {next_dir}")
                shutil.move(prev_dir, next_dir)

        # Save current model state and metadata
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.state_dict(), self.ckpt_path)

        metadata = {
            "step": step,
            "val_loss": val_loss,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": get_config(self.cfg_path, "models").get(self.name, {}),
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved step {step} checkpoint to {self.ckpt_path}")

    @torch.no_grad()
    def generate(self) -> str:
        """Generate new text by sampling from the model's predictions.

        Returns:
            str: The decoded string generated by the model.
        """
        return self.generator.output(self)

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
