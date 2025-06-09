"""
Utility functions for data processing, batching, configuration, and checkpointing.

Includes:
- Vocabulary and mapping creation
- Data encoding/decoding
- Data splitting and batching
- Model configuration and instantiation
- Checkpoint saving and rotation
"""

from models.base_model import BaseLanguageModel as BaseLM
import torch
import os
import shutil
import json
import time
import re
from typing import TypeVar, Any, cast

T = TypeVar("T")


def build_vocab(text: str) -> tuple[list[str], int]:
    """
    Build a sorted character vocabulary from text and return it with its size.

    Args:
        text (str): Input text.

    Returns:
        tuple[list[str], int]: Sorted list of unique characters and vocabulary size.
    """
    chars = sorted(set(text))
    vocab_size = len(chars)
    return chars, vocab_size


def create_mappings(chars: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Create character-to-index and index-to-character mappings.

    Args:
        chars (list[str]): List of unique characters.

    Returns:
        tuple[dict[str, int], dict[int, str]]: stoi and itos mappings.
    """
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_data(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """
    Encode text into a tensor of integer indices using the provided mapping.

    Args:
        text (str): Input text.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        torch.Tensor: Encoded tensor of indices (dtype=torch.long).
    """
    encoded = [stoi[c] for c in text]
    data = torch.tensor(encoded, dtype=torch.long)
    return data


def decode_data(data: torch.Tensor, itos: dict[int, str]) -> str:
    """
    Decode a tensor of integer indices back into a string using the mapping.

    Args:
        data (torch.Tensor): Tensor of indices.
        itos (dict[int, str]): Index-to-character mapping.

    Returns:
        str: Decoded string.
    """
    decoded = "".join([itos[i] for i in data.tolist()])
    return decoded


def split_data(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split data tensor into training and validation sets (90/10 split).

    Args:
        data (torch.Tensor): Input data tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (train_data, val_data)
    """
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


def get_batch(model: BaseLM, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data for training.
    Returns input (x) and target (y) sequences of length block_size.
    Sequences are stacked and targets are shifted by 1 position.
    Starting indices are randomized for each sequence in the batch.

    Args:
        model (BaseLM): Model with block_size, batch_size, and device attributes.
        data (torch.Tensor): 1D tensor of encoded characters.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            x: (batch_size, block_size) - input sequences
            y: (batch_size, block_size) - target sequences (shifted by 1)
    """
    block_size = cast(int, model.block_size)
    batch_size = cast(int, model.batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(model.device), y.to(model.device)


def get_metadata(path: str, key: str, default: T) -> T:
    """
    Retrieve a value from metadata.json; returns a default if not found.

    Args:
        path (str): Path to metadata.json.
        key (str): Key to retrieve from the metadata.
        default (T): Default value if key is not found.

    Returns:
        T: The value from metadata or the default.
    """
    data = default
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            data = metadata.get(key, default)
    return data


def save_config(config: dict[str, Any], cfg_path: str) -> None:
    """
    Save a configuration dictionary to a JSON file.
    Formats arrays to remove line breaks
    Adds line breaks between top-level sections that end with '},'.

    Args:
        config (dict[str, Any]): The configuration dictionary to save
        cfg_path (str): Path where the configuration file should be saved
    """
    cfg_str = json.dumps(config, indent=2)

    def fix_arrays(match):
        # Extract only the numbers from the array content
        numbers = re.findall(r"\d+", match.group(1))
        return "[" + ", ".join(numbers) + "]"

    # Collapse arrays to a single line
    cfg_str = re.sub(r"\[(.*?)\]", fix_arrays, cfg_str, flags=re.DOTALL)

    cfg_lines = cfg_str.split("\n")
    cfg_format = []

    for i, line in enumerate(cfg_lines):
        cfg_format.append(line)
        # Add a line break after top-level sections that end with '},'
        if line == "  }," and i < len(cfg_lines) - 1:
            cfg_format.append("")

    cfg_str = "\n".join(cfg_format) + "\n"

    with open(cfg_path, "w") as f:
        f.write(cfg_str)


def load_config(path: str) -> dict[str, Any]:
    """
    Load the entire config.json as a dict.

    Args:
        path (str): Path to config.json.

    Returns:
        dict[str, Any]: The loaded configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config(path: str, config_name: str) -> dict[str, Any]:
    """
    Load and return the configuration dictionary for the given model.

    Args:
        path (str): Path to config.json.
        config_name (str): Name of the model config to retrieve.

    Returns:
        dict[str, Any]: The configuration dictionary for the model.

    Raises:
        KeyError: If no config is found for the given name.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)[config_name]
    except KeyError:
        raise KeyError(f"No config found for: {config_name}")
    if config is None:
        raise ValueError(f"No value found for config: {config_name}")
    return config


def get_model(
    models: type,
    model_name: str,
    config: dict[str, Any],
    cfg_path: str,
    vocab_size: int,
) -> BaseLM:
    """
    Create and return a language model based on the specified model type.

    Args:
        models (type): Model registry with BigramLM, LSTMLM, TransformerLM, etc.
        model_name (str): Name of the model type ("bigram", "lstm", "transformer").
        config (dict[str, Any]): Configuration dictionary for all models.
        cfg_path (str): Path to the config file.
        vocab_size (int): Vocabulary size (not used for transformer).

    Returns:
        BaseLM: Instantiated language model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name == "bigram":
        model = models.BigramLM(config[model_name], cfg_path, vocab_size)
    elif model_name == "lstm":
        model = models.LSTMLM(config[model_name], cfg_path, vocab_size)
    elif model_name == "gru":
        model = models.GRULM(config[model_name], cfg_path, vocab_size)
    elif model_name == "transformer":
        model = models.TransformerLM(config[model_name], cfg_path, vocab_size)
    elif model_name == "distilgpt2":
        model = models.DistilGPT2LM(config[model_name], cfg_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    model.to(model.device)
    return model


def save_checkpoint(
    model: BaseLM, step: int, val_loss: float, max_checkpoints: int
) -> None:
    """
    Save a model checkpoint and rotate older checkpoints.

    The oldest checkpoint is removed and the rest are shifted up by 1. The
    current model state and metadata are saved as checkpoint_1.

    Args:
        model (BaseLM): Model instance with dir_path, ckpt_dir, ckpt_path, meta_path.
        step (int): Current training step.
        val_loss (float): Validation loss at this step.
        max_checkpoints (int): Maximum number of checkpoints to keep.
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
        "config": get_config(model.cfg_path, "models")[model.name],
    }

    with open(model.meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved checkpoint to {model.ckpt_path}")
