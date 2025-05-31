"""
Training utilities for model optimization, validation, and early stopping.

Includes:
- train: Main training loop with early stopping and checkpointing.
- validate_data: Validation and early stopping logic.
"""

from models.registry import ModelRegistry as Model
import torch
from optuna import Trial, TrialPruned
from utils import get_batch, save_checkpoint, get_metadata, split_data, get_config
from visualizer import plot_losses


def train(
    model: Model.BaseLM,
    data: torch.Tensor,
    trial: Trial | None = None,
    step_divisor: int = 1,
) -> tuple[list[float], list[float]]:
    """
    Train the model using Adam optimization with early stopping.
    Saves model checkpoints after validation loss improves.
    Supports hyperparameter optimization via the trial argument.
    Optimization is through Optuna and optional.

    Args:
        model (Model.BaseLM): The model to train.
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.
        trial (Trial | None, optional): Optuna trial for pruning. Defaults to None.
        step_divisor (int, optional): Trial training step divisor. Defaults to 1.

    Returns:
        tuple[list[float], list[float]]: (training losses, validation losses)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(model.lr))
    train_data, val_data = split_data(data)
    best_loss = get_metadata(model.meta_path, "val_loss", float("inf"))
    visualization = get_config(model.cfg_path, "visualization")
    losses = []
    val_losses = []
    wait = 0

    for step in range(model.steps // step_divisor):
        xb, yb = get_batch(model, train_data)
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

    plot_losses(model, losses, val_losses, step_divisor, **visualization)
    return losses, val_losses


def validate_data(
    model: Model.BaseLM,
    data: torch.Tensor,
    step: int,
    step_divisor: int,
    loss: torch.Tensor,
    val_losses: list[float],
    best_loss: float,
    wait: int,
) -> tuple[bool, float, int]:
    """
    Validate model performance and handle early stopping.
    Saves the best model and restores it if overfitting is detected.
    Returns overfit status, best loss, and wait count.

    Args:
        model (Model.BaseLM): The model being trained.
        data (torch.Tensor): Validation data as a 1D tensor of encoded characters.
        step (int): Current training step.
        step_divisor (int): Step divisor for validation frequency.
        loss (torch.Tensor): Current training loss.
        val_losses (list[float]): List to append validation losses to.
        best_loss (float): Best validation loss so far.
        wait (int): Number of validation steps since last improvement.

    Returns:
        tuple[bool, float, int]: (overfit, best_loss, wait)
            overfit (bool): True if early stopping triggered.
            best_loss (float): Updated best validation loss.
            wait (int): Updated wait counter.
    """
    overfit = False
    if step % model.interval == 0:
        xb, yb = get_batch(model, data)

        with torch.no_grad():
            _, val_loss, *_ = model(xb, yb)
            val_losses.append(val_loss.item())

            print(
                f"Step: {step:<10}"
                f"loss: {loss:<20.10f}"
                f"val_loss: {val_loss.item():<20.10f}"
            )

            save_model = get_config(model.cfg_path, "save_model")
            full_training_run = step_divisor == 1
            if save_model and full_training_run and (val_loss < best_loss):
                # Save model if validation loss improves
                best_loss = val_loss
                wait = 0
                save_checkpoint(model, step, val_loss.item(), model.max_checkpoints)
            elif full_training_run:
                wait += 1
                if wait >= model.patience:
                    # Try to restore best model before stopping
                    try:
                        model.load_state_dict(torch.load(model.ckpt_path))
                    except Exception as e:
                        print(f"Error restoring model: {e}")
                    overfit = True
                    print(f"Stopping due to overfitting.")
                    print(f"Step: {step}, Best Loss: {best_loss}")

    return overfit, best_loss, wait
