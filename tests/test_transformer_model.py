import torch
from models.transformer_model import TransformerLanguageModel


def test_transformer_model():
    model = TransformerLanguageModel()
    assert model is not None


def test_transformer_model_init():
    model = TransformerLanguageModel()
    assert model.name == "transformer"
    assert model.tokenizer is not None
    assert model.model is not None


def test_transformer_model_tokenizer():
    model = TransformerLanguageModel()
    assert model.tokenizer.name_or_path == "distilgpt2"
    assert model.tokenizer.model_max_length == 1024
    assert model.tokenizer.vocab_size == 50257


def test_transformer_model_model():
    model = TransformerLanguageModel()
    assert model.model.name_or_path == "distilgpt2"
    assert model.model.config.architectures == ["GPT2LMHeadModel"]
    assert model.model.config.model_type == "gpt2"
    assert model.model.config.vocab_size == 50257


def test_transformer_model_run():
    model = TransformerLanguageModel()
    text = "Hello!"
    block_size = 5
    max_new_tokens = 5
    result = model.run(text, block_size, max_new_tokens)
    assert type(result) == str
    assert len(result) >= (block_size + max_new_tokens)
    assert result[:block_size] in text
