"""Data encoding, decoding, and splitting utilities for language modeling workflows.

Includes:
- Encoding tokens to tensors
- Decoding tensors to text
- Splitting data into training and validation sets
"""

import torch


def encode_data(tokens: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """Encode text into a tensor of integer indices using the provided mapping.

    Args:
        tokens (list[str]): Input tokens.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        torch.Tensor: Encoded tensor of indices (dtype=torch.long).
    """
    encoded = [stoi[t] for t in tokens]
    data = torch.tensor(encoded, dtype=torch.long)
    return data


def decode_data(data: torch.Tensor, itos: dict[int, str], token_level: str) -> str:
    """Decode a tensor of integer indices back into a string using the mapping.

    Args:
        data (torch.Tensor): Tensor of indices.
        itos (dict[int, str]): Index-to-character mapping.
        token_level (str): Token level to use for vocabulary building.
            Options: "char" (default), or "word"

    Returns:
        str: Decoded string.
    """
    separator = " " if token_level == "word" else ""
    decoded = separator.join([itos[i] for i in data.tolist()])
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
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data
