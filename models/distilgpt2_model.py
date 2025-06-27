"""DistilGPT-2 wrapper for language modeling and text generation."""

import random
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_model import BaseLanguageModel


class DistilGPT2LanguageModel(BaseLanguageModel):
    """A language model that integrates a transformer (DistilGPT2) for text generation.

    Architecture:
        - Uses Hugging Face's tokenizer and model for causal language modeling.
        - Generates text by sampling from the model's predictions given a prompt.
        - The prompt is a random sequence of tokens from the dataset.

    Attributes:
        block_size (int): Length of input sequences for the transformer.
        tokenizer (AutoTokenizer): Hugging Face tokenizer for input text.
        model (AutoModelForCausalLM): Hugging Face transformer model.
        config["hparams"] keys: config.json hparams attributes.
            (type-hinted above __init__)

    Notes:
        Keys in config["hparams"] are attributes set dynamically at initialization.
    """

    block_size: int

    def __init__(self, config: dict[str, Any], cfg_path: str) -> None:
        """Initialize the transformer language model and load pre-trained weights.

        Args:
            config (dict): Configuration dictionary for the model.
            cfg_path (str): Path to the config file.

        Notes:
            config["hparams"] keys are set as attributes on the model instance.
        """
        super().__init__(
            model_name="distilgpt2",
            config=config,
            cfg_path=cfg_path,
        )

        # These are unnecessary since DistilGPT2 uses its own vocabulary
        for key in config.get("vocab", {}).keys():
            delattr(self, key)

        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        # Set all hparams config keys as attributes
        for key, value in config.get("hparams", {}).items():
            setattr(self, key, value)

        self.block_size = int(self.block_size)

    def run(self, text: str) -> str:
        """Generate text using the pre-trained transformer model.

        Selects a random prompt from the input text, encodes it, and generates new
        text using the transformer model. Returns the generated text as a string.

        Args:
            text (str): Input text to use as the source for prompts.

        Returns:
            str: The generated text string.
        """
        # Select a random prompt from the dataset
        start_idx = random.randint(0, len(text) - self.block_size)
        prompt = text[start_idx : start_idx + self.block_size]
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # Generate new text using the transformer model
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
