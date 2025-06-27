"""Text generation strategies for language models."""

import logging
import random
from typing import Protocol

import torch
import torch.nn.functional as F

from utils.data_utils import decode_data

logger = logging.getLogger(__name__)


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
        logger.debug(f"Initialized MultinomialSampler with temperature: {temperature}")

    def get_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample the next token index from the model's logits.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C).

        Returns:
            torch.Tensor: The index of the next token.
        """
        logits = logits[:, -1, :]
        probs = F.softmax(logits / self.temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        logger.debug(f"Sampled token index: {next_token.item()}")
        return next_token


class ArgmaxSampler:
    """Selects the most likely token (argmax)."""

    def __init__(self) -> None:
        """Initialize the argmax sampler."""
        logger.debug("Initialized ArgmaxSampler")

    def get_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the most likely token index from the model's logits.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, T, C).

        Returns:
            torch.Tensor: The index of the next token.
        """
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        logger.debug(f"Selected token index (argmax): {next_token.item()}")
        return next_token


class TextGenerator:
    """Base class for text generators."""

    def __init__(
        self, stoi: dict[str, int], itos: dict[int, str], sampler: Sampler | None = None
    ) -> None:
        """Initialize the text generator.

        Args:
            stoi (dict[str, int]): The mapping from characters to token indices.
            itos (dict[int, str]): The mapping from token indices to characters.
            sampler (Sampler | None): The token sampling strategy to use.
        """
        self.stoi = stoi
        self.itos = itos
        self.sampler = sampler or MultinomialSampler()

    def tokens(self, *_) -> torch.Tensor:
        """Generate tokens from the model.

        Args:
            *_: Unused arguments

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Method implemented in subclasses")

    def output(self, model) -> str:
        """Generate text from the model using the prompt.

        Args:
            model: The language model instance to use for generation.

        Returns:
            str: The generated text.
        """
        tokens = self.tokens(model)
        text = decode_data(tokens, self.itos, model.token_level)

        logger.info(f"Generated text length: {len(text)} characters")
        logger.debug(
            f"Generated text preview: {text[:100]}{'...' if len(text) > 100 else ''}"
        )

        return text


class RandomTextGenerator(TextGenerator):
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
        super().__init__(stoi, itos, sampler)
        self.start_idx = stoi[random.choice(list(stoi.keys()))]

        logger.debug(
            f"Initialized RandomTextGenerator with start_idx: {self.start_idx}"
        )
        logger.debug(f"Vocabulary size: {len(stoi)} tokens")

    def tokens(self, model) -> torch.Tensor:
        """Generate tokens from the model.

        Args:
            model: The language model instance to use for generation.

        Returns:
            torch.Tensor: The generated sequence of token indices.
        """
        max_tokens = model.max_new_tokens
        logger.info(f"Starting random text generation for {max_tokens} tokens")

        model.eval()
        tokens = torch.tensor([self.start_idx], dtype=torch.long, device=model.device)
        idx = torch.tensor([[self.start_idx]], dtype=torch.long, device=model.device)

        for i in range(model.max_new_tokens):
            logits = model(idx)
            next_idx = self.sampler.get_next_token(logits)
            idx = next_idx
            tokens = torch.cat((tokens, next_idx.flatten()), dim=0)

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{model.max_new_tokens} tokens")

        logger.info(f"Text generation completed: {len(tokens)} tokens")
        return tokens


class PromptTextGenerator(TextGenerator):
    """Generates text from a model using a prompt.

    This generator uses a provided prompt and context length to generate text.
    The starting index is chosen randomly from the vocabulary.

    Attributes:
        context_length (int): The context length to use for generation.
    """

    def __init__(
        self,
        context_length: int,
        stoi: dict[str, int],
        itos: dict[int, str],
        sampler: Sampler | None = None,
    ) -> None:
        """Initialize the prompt text generator.

        Args:
            context_length (int): The context length to use for generation.
            stoi (dict[str, int]): The mapping from characters to token indices.
            itos (dict[int, str]): The mapping from token indices to characters.
            sampler (Sampler | None): The token sampling strategy to use.
        """
        super().__init__(stoi, itos, sampler)
        self.context_length = context_length

    def tokens(self, model) -> torch.Tensor:
        """Generate tokens from the model using the prompt.

        Args:
            model: The language model instance to use for generation.

        Returns:
            torch.Tensor: The generated sequence of token indices.
        """
        max_tokens = model.max_new_tokens
        logger.info(f"Starting prompt text generation for {max_tokens} tokens")

        model.eval()
        prompt = f"{input('Enter a prompt: ')} "
        prompt_tokens = torch.tensor([self.stoi[c] for c in prompt], dtype=torch.long)
        tokens = prompt_tokens.clone().to(model.device)
        idx = tokens.unsqueeze(0)

        for i in range(max_tokens):
            idx_max = idx[:, -self.context_length :]
            logits = model(idx_max)
            next_idx = self.sampler.get_next_token(logits)
            idx = torch.cat((idx, next_idx), dim=1)
            tokens = torch.cat((tokens, next_idx.flatten()), dim=0)

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{max_tokens} tokens")

        logger.info(f"Text generation completed: {len(tokens)} tokens")
        return tokens


class Generators:
    """Registry for generator classes."""

    class Text:
        """Registry for text classes."""

        Random = RandomTextGenerator
        Prompt = PromptTextGenerator


class Samplers:
    """Registry for sampler classes."""

    Multinomial = MultinomialSampler
    Argmax = ArgmaxSampler
