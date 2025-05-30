from models.base_model import BaseLanguageModel as BaseLM
import torch
import os
import shutil
import json
import time
from typing import TypeVar, Any

T = TypeVar("T")


def build_vocab(text: str) -> tuple[list[str], int]:
    """Build a sorted character vocabulary from text and return it with its size."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    return chars, vocab_size


def create_mappings(chars: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create character-to-index and index-to-character mappings."""
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_data(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Encode text into a tensor of integer indices using the provided mapping."""
    encoded = [stoi[c] for c in text]
    data = torch.tensor(encoded, dtype=torch.long)
    return data


def decode_data(data: torch.Tensor, itos: dict[int, str]) -> str:
    """Decode a tensor of integer indices back into a string using the mapping."""
    decoded = "".join([itos[i] for i in data.tolist()])
    return decoded


def split_data(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data tensor into training and validation sets (90/10 split)."""
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


def get_batch(
    model: BaseLM, data: torch.Tensor, batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of data for training.
    Returns input (x) and target (y) sequences of length block_size.
    Sequences are stacked and targets are shifted by 1 position.
    Starting indices are randomized for each sequence in the batch.

    Shapes:
        data: (N,) - 1D tensor of encoded characters
        x: (batch_size, block_size) - input sequences
        y: (batch_size, block_size) - target sequences (shifted by 1)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(model.device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(model.device)
    return x, y


def get_metadata(path: str, key: str, default: T) -> T:
    """Retrieve a value from metadata.json; returns a default if not found."""
    data = default
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            data = metadata.get(key, default)
    return data


def get_config(path: str, config_name: str) -> dict[str, Any]:
    """Load and return the configuration dictionary for the given model."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)[config_name]
    if config is None:
        raise ValueError(f"No config found for: {config_name}")
    return config


def get_model(
    models: type,
    model_name: str,
    config: dict[str, Any],
    cfg_path: str,
    vocab_size: int,
) -> BaseLM:
    """Create and return a language model based on the specified model type."""
    if model_name == "bigram":
        model = models.BigramLM(config[model_name], cfg_path, vocab_size)
    elif model_name == "lstm":
        model = models.LSTMLM(config[model_name], cfg_path, vocab_size)
    elif model_name == "transformer":
        model = models.TransformerLM(config[model_name], cfg_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    model.to(model.device)
    return model


def save_checkpoint(
    model: BaseLM, step: int, val_loss: float, max_checkpoints: int
) -> None:
    """
    Saves a model checkpoint and rotates older checkpoints.
    The oldest checkpoint is removed and the rest are shifted up by 1.
    """
    os.makedirs(model.dir_path, exist_ok=True)

    # Shift existing checkpoints up by 1 (e.g. checkpoint_1 -> checkpoint_2)
    for i in reversed(range(1, max_checkpoints)):
        prev_dir = os.path.join(model.dir_path, f"checkpoint_{i}")
        next_dir = os.path.join(model.dir_path, f"checkpoint_{i + 1}")

        if os.path.exists(next_dir):
            shutil.rmtree(next_dir)

        if os.path.exists(prev_dir):
            shutil.move(prev_dir, next_dir)

    # Save current model state and metadata
    os.makedirs(model.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), model.ckpt_path)

    metadata = {
        "step": step,
        "val_loss": val_loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config": get_config(model.cfg_path, model.name),
    }

    with open(model.meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved checkpoint to {model.ckpt_path}")
