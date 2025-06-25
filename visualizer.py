"""Visualization utilities for plotting training and validation loss curves.

Includes:
- plot_losses: Plot and optionally save/smooth loss curves.
- smooth: Exponential smoothing for loss values.
- save_plot: Save matplotlib plots with timestamped filenames.
"""

import logging
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt

from models.registry import ModelRegistry as Model

logger = logging.getLogger(__name__)


def plot_losses(
    model: Model.BaseLM,
    losses: list[float],
    val_losses: list[float],
    step_divisor: int,
    visualization: dict[str, bool | float],
) -> None:
    """Plot and optionally save and smooth training and validation loss curves.

    Creates a plot of training and validation losses over time, with optional
    smoothing and saving capabilities. The visualization dictionary controls
    various plotting options.

    Args:
        model (BaseLM): Model instance being trained
        losses (list[float]): List of training losses
        val_losses (list[float]): List of validation losses
        step_divisor (int): Divisor for step numbers (1 for full training)
        visualization (dict[str, bool | float]): Dictionary containing:
            - show_plot (bool): Whether to display the plot
            - smooth_loss (bool): Whether to smooth training loss
            - smooth_val_loss (bool): Whether to smooth validation loss
            - weight (float): Smoothing weight (0-1)
            - save_plot (bool): Whether to save the plot
    """
    logger.debug(f"Creating loss plot for {model.name} model")
    logger.debug(
        f"Loss data: {len(losses)} training steps, {len(val_losses)} validation points"
    )

    steps = range(len(losses))
    val_steps = [i * model.interval for i in range(len(val_losses))]

    if visualization.get("smooth_loss", False):
        logger.debug("Applying smoothing to training loss")
        losses = smooth(losses, visualization.get("weight", 0.9))
    if visualization.get("smooth_val_loss", False):
        logger.debug("Applying smoothing to validation loss")
        val_losses = smooth(val_losses, visualization.get("weight", 0.9))

    plt.plot(steps, losses, label="Training Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss over Steps for {model.name}_model.py")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    full_training_run = step_divisor == 1
    if visualization.get("save_plot", False) and full_training_run:
        logger.debug("Saving loss plot")
        save_plot(model, plt, "losses")
    if visualization.get("show_plot", False) and full_training_run:
        logger.debug("Displaying loss plot")
        plt.show()
    if not full_training_run:
        logger.debug("Closing plot (trial run)")
        plt.close("all")


def smooth(values: list[float], weight: float) -> list[float]:
    """Apply exponential smoothing to a list of values.

    Args:
        values (list[float]): The input values to smooth.
        weight (float): The smoothing factor.
            (0-1, higher means smoother)

    Returns:
        list[float]: The smoothed values.
    """
    logger.debug(f"Applying exponential smoothing with weight: {weight}")

    smoothed_values = []
    last_value = values[0]

    for value in values:
        next_value = last_value * weight + value * (1 - weight)
        smoothed_values.append(next_value)
        last_value = next_value

    logger.debug(f"Smoothed {len(values)} values")
    return smoothed_values


def save_plot(model: Model.BaseLM, plt: Any, plot_name: str) -> None:
    """Save the current matplotlib plot to the model's plot directory.

    Plots are timestamped for unique naming.

    Args:
        model (nn.Module): The model instance.
            (must have a 'plot_dir' and 'name' attribute)
        plt (Any): The matplotlib pyplot module.
        plot_name (str): A label for the plot (used in the filename).
    """
    logger.debug(f"Saving plot '{plot_name}' for model {model.name}")

    os.makedirs(model.plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model.name}_{plot_name}_{timestamp}.png"
    filepath = os.path.join(model.plot_dir, filename)

    plt.savefig(filepath)
    logger.info(f"Plot saved to: {filepath}")
