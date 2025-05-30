from models.registry import ModelRegistry as Model
import torch
from utils import get_batch, save_checkpoint, get_metadata, split_data, get_config
from visualizer import plot_losses


def train(model: Model.BaseLM, data: torch.Tensor) -> tuple[list[float], list[float]]:
    """
    Train the model using Adam optimization with early stopping.
    Saves model checkpoints after validation loss improves.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    train_data, val_data = split_data(data)
    best_loss = get_metadata(model.meta_path, "val_loss", float("inf"))
    visualization = get_config(model.cfg_path, "visualization")
    losses = []
    val_losses = []
    wait = 0

    for step in range(model.steps):
        xb, yb = get_batch(model, train_data, model.batch_size, model.block_size)
        loss = model.train_step(xb, yb, optimizer)
        losses.append(loss)

        overfit, best_loss, wait = validate_data(
            model, val_data, step, loss, val_losses, best_loss, wait
        )

        if overfit:
            break

    plot_losses(model, losses, val_losses, model.interval, **visualization)
    return losses, val_losses


def validate_data(
    model: Model.BaseLM,
    data: torch.Tensor,
    step: int,
    loss: torch.Tensor,
    val_losses: list[float],
    best_loss: float,
    wait: int,
) -> tuple[bool, float, int]:
    """
    Validate model performance and handle early stopping.
    Saves the best model and restores it if overfitting is detected.
    Returns overfit status, best loss, and wait count.
    """
    overfit = False
    if step % model.interval == 0:
        xb, yb = get_batch(model, data, model.batch_size, model.block_size)

        with torch.no_grad():
            _, val_loss, *_ = model(xb, yb)
            val_losses.append(val_loss.item())

            print(
                f"Step: {step:<10}"
                f"loss: {loss:<20.10f}"
                f"val_loss: {val_loss.item():<20.10f}"
            )

            if (val_loss < best_loss) and get_config(model.cfg_path, "save_model"):
                # Save model if validation loss improves
                best_loss = val_loss
                wait = 0
                save_checkpoint(model, step, val_loss.item(), model.max_checkpoints)
            else:
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
