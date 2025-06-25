"""Data encoding, decoding, and splitting utilities for language modeling workflows.

Includes:
- Encoding tokens to tensors
- Decoding tensors to text
- Splitting data into training and validation sets
"""

import logging

import torch

logger = logging.getLogger(__name__)


def encode_data(tokens: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """Encode text into a tensor of integer indices using the provided mapping.

    Args:
        tokens (list[str]): Input tokens.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        torch.Tensor: Encoded tensor of indices (dtype=torch.long).
    """
    logger.debug(f"Encoding {len(tokens)} tokens using vocabulary of size {len(stoi)}")

    encoded = [stoi[t] for t in tokens]
    data = torch.tensor(encoded, dtype=torch.long)

    logger.debug(f"Encoded data shape: {data.shape}, dtype: {data.dtype}")
    return data


def decode_data(data: torch.Tensor, itos: dict[int, str], token_level: str) -> str:
    """Decode a tensor of integer indices back into a string using the mapping.

    Args:
        data (torch.Tensor): Tensor of indices.
        itos (dict[int, str]): Index-to-character mapping.
        token_level (str): Token level to use for vocabulary building.
            Options: "char" or "word"

    Returns:
        str: Decoded string.
    """
    logger.debug(
        f"Decoding tensor of shape {data.shape} with token_level: {token_level}"
    )

    separator = " " if token_level == "word" else ""
    decoded = separator.join([itos[i] for i in data.tolist()])

    logger.debug(f"Decoded text length: {len(decoded)} characters")
    return decoded


def split_data(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data tensor into training and validation sets (90/10 split).

    Args:
        data (torch.Tensor): Input data tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - train_data: Training data tensor
            - val_data: Validation data tensor
    """
    total_size = len(data)
    split_idx = int(0.9 * total_size)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(
        f"Split data: {total_size} total tokens -> "
        f"{len(train_data)} train ({len(train_data)/total_size:.1%}), "
        f"{len(val_data)} validation ({len(val_data)/total_size:.1%})"
    )

    return train_data, val_data
