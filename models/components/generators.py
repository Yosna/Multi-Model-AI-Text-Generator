"""Text generation strategies for language models."""

import random
from typing import Protocol

import torch
import torch.nn.functional as F

from utils.data_utils import decode_data


class Sampler(Protocol):
    """Protocol for token sampling strategies."""

    def get_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the next token index given model logits.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C).
        """
        ...


class MultinomialSampler:
    """Samples the next token using multinomial sampling with temperature."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize the multinomial sampler.

        Args:
            temperature (float): The temperature for sampling.
        """
        self.temperature = temperature

    def get_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample the next token index from the model's logits.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C).

        Returns:
            torch.Tensor: The index of the next token.
        """
        logits = logits[:, -1, :]
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class ArgmaxSampler:
    """Selects the most likely token (argmax)."""

    def __init__(self) -> None:
        """Initialize the argmax sampler."""
        pass

    def get_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the most likely token index from the model's logits.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C).

        Returns:
            torch.Tensor: The index of the next token.
        """
        logits = logits[:, -1, :]
        return torch.argmax(logits, dim=-1, keepdim=True)


class RandomTextGenerator:
    """Generates text from a model using a specified sampling strategy.

    This generator uses a provided sampler to select the next token at each step.
    The starting index is chosen randomly from the vocabulary.

    Attributes:
        start_idx (int): The randomly chosen starting token index.
        sampler (Sampler): The token sampling strategy to use.
    """

    def __init__(
        self, stoi: dict[str, int], itos: dict[int, str], sampler: Sampler | None = None
    ) -> None:
        """Initialize the text generator.

        Args:
            stoi (dict[str, int]): The mapping from characters to token indices.
            itos (dict[int, str]): The mapping from token indices to characters.
            sampler (Sampler | None): The token sampling strategy to use.
        """
        self.start_idx = stoi[random.choice(list(stoi.keys()))]
        self.sampler = sampler or MultinomialSampler()
        self.stoi = stoi
        self.itos = itos

    def tokens(self, model) -> torch.Tensor:
        """Generate tokens from the model.

        Args:
            model: The language model instance to use for generation.

        Returns:
            torch.Tensor: The generated sequence of token indices.
        """
        model.eval()
        tokens = torch.tensor([self.start_idx], dtype=torch.long, device=model.device)
        idx = torch.tensor([[self.start_idx]], dtype=torch.long, device=model.device)

        for _ in range(model.max_new_tokens):
            logits = model(idx)
            next_idx = self.sampler.get_next_token(logits)
            idx = next_idx
            tokens = torch.cat((tokens, next_idx.flatten()), dim=0)

        return tokens

    def output(self, model) -> str:
        """Generate text from the model.

        Args:
            model: The language model instance to use for generation.

        Returns:
            str: The generated text.
        """
        return decode_data(self.tokens(model), self.itos, model.token_level)


class Generators:
    """Registry for generator classes."""

    class Text:
        """Registry for text classes."""

        Random = RandomTextGenerator


class Samplers:
    """Registry for sampler classes."""

    Multinomial = MultinomialSampler
    Argmax = ArgmaxSampler
