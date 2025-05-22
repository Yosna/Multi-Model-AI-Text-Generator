import torch
import torch.nn as nn
import random
import os
import argparse
import utils
from models.registry import ModelRegistry
from typing import TypeVar

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a language model")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["bigram", "lstm", "transformer"],
        metavar="[bigram|lstm|transformer]",
        help="Model name to use from config.json",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Prepare data, initialize model, and run training or generation."""
    model_name = args.model.lower()
    text = utils.load_full_directory("dataset", "txt")
    chars, vocab_size = utils.build_vocab(text)
    stoi, itos = utils.create_mappings(chars)
    data = utils.encode_data(text, stoi)
    config = utils.get_config("config.json", model_name)
    model = utils.get_model(ModelRegistry, model_name, vocab_size, **config["model"])

    validate_model(model, data, stoi, itos, **config["runtime"])


def validate_model(
    model: nn.Module,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    **config: T,
) -> None:
    """Validate the type of model to determine the appropriate run method."""
    if model.name == "transformer":
        generated_text = model.run(data, itos, **config)
        print(generated_text)
    else:
        run_model(model, data, stoi, itos, **config)


def run_model(
    model: nn.Module,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    training: bool,
    batch_size: int,
    block_size: int,
    steps: int,
    interval: int,
    lr: float,
    patience: int,
    max_new_tokens: int,
    max_checkpoints: int,
) -> None:
    """
    Run training or text generation for the model.
    Loads from checkpoint if available. Randomizes seed character for generation.
    """
    if os.path.exists(model.ckpt_path):
        model.load_state_dict(torch.load(model.ckpt_path))

    if training:
        train(
            model,
            data,
            batch_size,
            block_size,
            steps,
            interval,
            lr,
            patience,
            max_checkpoints,
        )
    else:
        seed_char = random.choice(list(stoi.keys()))
        start_idx = stoi[seed_char]
        generated_text = model.generate(start_idx, itos, max_new_tokens)
        print(generated_text)


def train(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    steps: int,
    interval: int,
    lr: float,
    patience: int,
    max_checkpoints: int,
) -> None:
    """
    Train the model using Adam optimization with early stopping.
    Saves model checkpoints after validation loss improves.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data, val_data = utils.split_data(data)
    best_loss = utils.get_metadata(model.meta_path, "val_loss", float("inf"))
    wait = 0

    for step in range(steps):
        xb, yb = utils.get_batch(train_data, batch_size, block_size)
        _, loss, *_ = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overfit, best_loss, wait = validate_data(
            model,
            val_data,
            batch_size,
            block_size,
            step,
            loss,
            best_loss,
            wait,
            interval,
            patience,
            max_checkpoints,
        )

        if overfit:
            break


def validate_data(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    step: int,
    loss: torch.Tensor,
    best_loss: float,
    wait: int,
    interval: int,
    patience: int,
    max_checkpoints: int,
) -> tuple[bool, float, int]:
    """
    Validate model performance and handle early stopping.
    Saves the best model and restores it if overfitting is detected.
    Returns overfit status, best loss, and wait count.
    """
    overfit = False
    if step % interval == 0:
        xb, yb = utils.get_batch(data, batch_size, block_size)

        with torch.no_grad():
            _, val_loss, *_ = model(xb, yb)

            print(
                f"Step: {step:<10}"
                f"loss: {loss.item():<20.10f}"
                f"val_loss: {val_loss.item():<20.10f}"
            )

            if val_loss < best_loss:
                # Save model if validation loss improves
                best_loss = val_loss
                wait = 0
                utils.save_checkpoint(model, step, val_loss.item(), max_checkpoints)
            else:
                wait += 1
                if wait >= patience:
                    # Restore best model before stopping
                    model.load_state_dict(torch.load(model.ckpt_path))
                    overfit = True
                    print(f"Stopping due to overfitting.")
                    print(f"Step: {step}, Best Loss: {best_loss}")

    return overfit, best_loss, wait


if __name__ == "__main__":
    main(parse_args())
