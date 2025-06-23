"""Training utilities for model optimization, validation, and early stopping.

Includes:
- train: Main training loop with early stopping and checkpointing.
- validate_data: Validation and early stopping logic.
"""

from typing import cast

import torch
from optuna import Trial, TrialPruned

from models.registry import ModelRegistry as Model
from utils.data_utils import split_data
from utils.io_utils import get_config, get_metadata
from utils.model_utils import get_batch
from visualizer import plot_losses


def train(
    model: Model.BaseLM,
    data: torch.Tensor,
    trial: Trial | None = None,
    step_divisor: int = 1,
) -> tuple[list[float], list[float]]:
    """Train the model.

    Uses Adam optimization with early stopping.
    Saves model checkpoints after validation loss improves.
    Supports hyperparameter optimization via the trial argument.
    Optimization uses Optuna and is optional.

    Args:
        model (Model.BaseLM): The model to train.
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.
        trial (Trial | None): Optuna trial for pruning.
        step_divisor (int): Trial training step divisor.

    Raises:
        TrialPruned: If the trial should be pruned.

    Returns:
        tuple[list[float], list[float]]:
            - training losses
            - validation losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(model.lr))
    train_data, val_data = split_data(data)
    best_loss = get_metadata(model.meta_path, "val_loss", float("inf"))
    visualization = get_config(model.cfg_path, "visualization")
    losses = []
    val_losses = []
    wait = 0

    for step in range(model.steps // step_divisor):
        block_size = cast(int, model.block_size)
        batch_size = cast(int, model.batch_size)
        xb, yb = get_batch(block_size, batch_size, train_data, model.device)
        loss = model.train_step(xb, yb, optimizer)
        losses.append(loss)

        overfit, best_loss, wait = validate_data(
            model, val_data, step, step_divisor, loss, val_losses, best_loss, wait
        )

        if trial is not None:
            trial.report(val_losses[-1], step)
            if trial.should_prune():
                raise TrialPruned()

        if overfit:
            break

    plot_losses(model, losses, val_losses, step_divisor, visualization)
    return losses, val_losses


def validate_data(
    model: Model.BaseLM,
    data: torch.Tensor,
    step: int,
    step_divisor: int,
    loss: float,
    val_losses: list[float],
    best_loss: float,
    wait: int,
) -> tuple[bool, float, int]:
    """Validate the model on the validation dataset.

    Args:
        model (Model.BaseLM): Model to validate
        data (torch.Tensor): Validation data
        step (int): Current training step
        step_divisor (int): Divisor for step numbers
        loss (float): Current training loss
        val_losses (list[float]): List to append validation losses to
        best_loss (float): Best validation loss so far
        wait (int): Number of steps without improvement

    Returns:
        tuple[bool, float, int]:
            - overfit: Whether the model is overfitting
            - best_loss: Best validation loss so far
            - wait: Number of steps without improvement
    """
    overfit = False
    if step % model.interval == 0:
        block_size = cast(int, model.block_size)
        batch_size = cast(int, model.batch_size)
        xb, yb = get_batch(block_size, batch_size, data, model.device)

        with torch.no_grad():
            logits = model(xb)
            val_loss = model.compute_loss(logits, xb, yb).item()

        print(f"Step: {step:<10} loss: {loss:<20.10f} val_loss: {val_loss:<20.10f}")

        val_losses.append(val_loss)
        loss_improved = val_loss < best_loss
        full_training_run = step_divisor == 1

        if loss_improved and full_training_run and model.save_model:
            # Save model if validation loss improves during a full training run
            model.save_checkpoint(step, val_loss)

        # Check if training should stop due to overfitting
        overfit, best_loss, wait = model.check_patience(best_loss, val_loss, wait)

    return overfit, best_loss, wait
