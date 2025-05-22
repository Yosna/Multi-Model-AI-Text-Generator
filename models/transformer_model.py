import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base_model import BaseLanguageModel
from utils import decode_data


class TransformerLanguageModel(BaseLanguageModel):
    """
    A language model that integrates a pre-trained transformer (DistilGPT-2) for text generation.

    Architecture:
        - Uses Hugging Face's DistilGPT-2 tokenizer and model for causal language modeling.
        - Generates text by sampling from the model's predictions given a prompt.
        - The prompt is a random sequence of tokens from the dataset.
    """

    def __init__(self, vocab_size: int = 0) -> None:
        """Initialize the transformer language model and load pre-trained weights."""
        super().__init__(vocab_size=vocab_size, model_name="transformer")
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def run(
        self, data: torch.Tensor, itos: dict, block_size: int, max_new_tokens: int
    ) -> str:
        """Generate text using the pre-trained transformer model."""
        # Select a random prompt from the dataset
        start_idx = random.randint(0, data.size(0) - block_size)
        prompt_ids = data[start_idx : start_idx + block_size]
        prompt = decode_data(prompt_ids.tolist(), itos)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)

        # Generate new text using the transformer model
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
