"""Utilities for vocabulary creation, mappings, batching, and model instantiation.

Includes:
- Vocabulary and mapping creation
- Batch generation for training
- Model instantiation for language models
"""

import logging
from typing import Any, TypeVar

import torch
from torch import Tensor
from torch import device as Device

from models.registry import ModelRegistry as Model

logger = logging.getLogger(__name__)

T = TypeVar("T")


def build_vocab(
    text: str, token_level: str = "char"
) -> tuple[list[str], list[str], int]:
    """Build a sorted character vocabulary from text and return it with its size.

    Args:
        text (str): Input text.
        token_level (str): Token level to use for vocabulary building.
            Options: "char" (default), or "word"

    Returns:
        tuple[list[str], list[str], int]:
            - Sorted list of unique tokens
            - Sorted list of unique characters
            - Vocabulary size

    Raises:
        ValueError: If token_level is not "char" or "word"
    """
    logger.debug(
        f"Building vocabulary from text of length {len(text)} "
        f"with token_level: {token_level}"
    )

    if token_level == "char":
        tokens = list(text)
    elif token_level == "word":
        tokens = text.split()
    else:
        raise ValueError(f"Invalid token level: {token_level}")

    vocab = sorted(set(tokens))
    vocab_size = len(vocab)

    logger.info(
        f"Built vocabulary: {vocab_size} unique {token_level}s "
        f"from {len(tokens)} total tokens"
    )
    logger.debug(
        f"Vocabulary size: {vocab_size}, "
        f"unique tokens: {vocab[:10]}{'...' if vocab_size > 10 else ''}"
    )

    return tokens, vocab, vocab_size


def create_mappings(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create character-to-index and index-to-character mappings.

    Args:
        tokens (list[str]): List of unique tokens.

    Returns:
        tuple[dict[str, int], dict[int, str]]:
            - stoi: character-to-index mapping
            - itos: index-to-character mapping
    """
    logger.debug(f"Creating mappings for {len(tokens)} unique tokens")

    stoi = {token: i for i, token in enumerate(tokens)}
    itos = {i: token for i, token in enumerate(tokens)}

    logger.debug(f"Created mappings: stoi size {len(stoi)}, itos size {len(itos)}")
    return stoi, itos


def get_batch(
    block_size: int, batch_size: int, data: Tensor, device: Device | None = None
) -> tuple[Tensor, Tensor]:
    """Generate a batch of data for training.

    Sequences are stacked and targets are shifted by 1 position.
    Starting indices are randomized for each sequence in the batch.

    Args:
        block_size (int): Size of each input sequence.
        batch_size (int): Number of sequences in each batch.
        data (Tensor): 1D tensor of encoded characters.
        device (Device | None): Device to use for tensor operations.

    Returns:
        tuple[Tensor, Tensor]:
            - x: (batch_size, block_size) - input sequences
            - y: (batch_size, block_size) - target sequences (shifted by 1)
    """
    logger.debug(
        f"Generating batch: block_size={block_size}, "
        f"batch_size={batch_size}, device={device}"
    )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    result = x.to(device or "cpu"), y.to(device or "cpu")
    logger.debug(f"Generated batch shapes: x={result[0].shape}, y={result[1].shape}")

    return result


def get_model(
    model_name: str,
    config: dict[str, Any],
    cfg_path: str,
    vocab_size: int,
    token_level: str,
) -> Model.BaseLM:
    """Create and return a language model based on the specified model type.

    Args:
        model_name (str): Name of the model type ("bigram", "lstm", "transformer").
        config (dict[str, Any]): Configuration dictionary for all models.
        cfg_path (str): Path to the config file.
        vocab_size (int): Vocabulary size (not used for transformer).
        token_level (str): Token level to use for vocabulary building.
            Options: "char" (default), or "word"

    Returns:
        Model.BaseLM: Instantiated language model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    logger.info(
        f"Creating {model_name} model with vocab_size={vocab_size}, "
        f"token_level={token_level}"
    )
    logger.debug(f"Model config keys: {list(config.keys())}")

    if model_name == "bigram":
        model = Model.BigramLM(config[model_name], cfg_path, vocab_size, token_level)
    elif model_name == "lstm":
        model = Model.LSTMLM(config[model_name], cfg_path, vocab_size, token_level)
    elif model_name == "gru":
        model = Model.GRULM(config[model_name], cfg_path, vocab_size, token_level)
    elif model_name == "transformer":
        model = Model.TransformerLM(
            config[model_name], cfg_path, vocab_size, token_level
        )
    elif model_name == "distilgpt2":
        model = Model.DistilGPT2LM(config[model_name], cfg_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.to(model.device)
    logger.info(f"Model created and moved to device: {model.device}")
    logger.debug(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total"
    )

    return model
