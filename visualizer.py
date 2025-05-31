"""
Visualization utilities for plotting training and validation loss curves.

Includes:
- plot_losses: Plot and optionally save/smooth loss curves.
- smooth: Exponential smoothing for loss values.
- save_plot: Save matplotlib plots with timestamped filenames.
"""

from models.registry import ModelRegistry as Model
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Any


def plot_losses(
    model: Model.BaseLM,
    losses: list[float],
    val_losses: list[float],
    step_divisor: int,
    show_plot: bool,
    smooth_loss: bool,
    smooth_val_loss: bool,
    weight: float,
    save_data: bool,
) -> None:
    """
    Plot and training/validation loss curves for a model.
    Options to smooth the loss curves and save the plot are configurable.

    Args:
        model (nn.Module): The model instance.
            (must have a 'name' and 'plot_dir' attribute)
        losses (list[float]): Training loss values.
        val_losses (list[float]): Validation loss values.
        step_divisor (int): Step divisor for validation loss.
        show_plot (bool): Option to display the plot interactively.
        smooth_loss (bool): Option to smooth the training loss curve.
        smooth_val_loss (bool): Option to smooth the validation loss curve.
        weight (float): Smoothing weight (0-1, higher is smoother).
        save_data (bool): Option to save the plot as an image file.
    """
    steps = range(len(losses))
    val_steps = [i * model.interval for i in range(len(val_losses))]
    losses = smooth(losses, weight) if smooth_loss else losses
    val_losses = smooth(val_losses, weight) if smooth_val_loss else val_losses

    plt.plot(steps, losses, label="Training Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss over Steps for {model.name}_model.py")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    full_training_run = step_divisor == 1
    if save_data and full_training_run:
        save_plot(model, plt, "losses")

    if show_plot and full_training_run:
        plt.show()

    if not full_training_run:
        plt.close("all")


def smooth(values: list[float], weight: float) -> list[float]:
    """
    Apply exponential smoothing to a list of values.

    Args:
        values (list[float]): The input values to smooth.
        weight (float): The smoothing factor.
            (0-1, higher means smoother)

    Returns:
        list[float]: The smoothed values.
    """
    smoothed_values = []
    last_value = values[0]

    for value in values:
        next_value = last_value * weight + value * (1 - weight)
        smoothed_values.append(next_value)
        last_value = next_value

    return smoothed_values


def save_plot(model: Model.BaseLM, plt: Any, plot_name: str) -> None:
    """
    Save the current matplotlib plot to the model's plot directory.
    Plots are timestamped for unique naming.

    Args:
        model (nn.Module): The model instance.
            (must have a 'plot_dir' and 'name' attribute)
        plt (matplotlib.pyplot): The matplotlib pyplot module.
        plot_name (str): A label for the plot (used in the filename).
    """
    os.makedirs(model.plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model.name}_{plot_name}_{timestamp}.png"
    plt.savefig(os.path.join(model.plot_dir, filename))
