from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_bigram_config():
    return {
        "runtime": {
            "training": True,
            "batch_size": 2,
            "block_size": 4,
            "steps": 1,
            "interval": 1,
            "lr": 0.0015,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 1,
        },
        "model": {},
    }


def get_bigram_model(
    config: dict[str, Any] = get_bigram_config(),
    cfg_path: str = "test_config.json",
    vocab_size: int = 10,
    token_level: str = "char",
):
    return Model.BigramLM(config, cfg_path, vocab_size, token_level)


def test_bigram_model():
    model = get_bigram_model()
    assert model is not None


def test_bigram_model_vocab_errors():
    with pytest.raises(ValueError):
        Model.BigramLM(
            config=get_bigram_config(),
            cfg_path="test_config.json",
            vocab_size=0,
            token_level="char",
        )
    with pytest.raises(ValueError):
        Model.BigramLM(
            config=get_bigram_config(),
            cfg_path="test_config.json",
            vocab_size=100000,
            token_level="char",
        )


def test_bigram_model_init():
    model = get_bigram_model()
    assert model.name == "bigram"
    assert model.vocab_size == 10


def test_bigram_model_embedding_layer():
    model = get_bigram_model()
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 10


def test_bigram_model_repr():
    model = get_bigram_model()
    assert str(model) == (
        f"BigramLanguageModel(\n" f"\tvocab_size={model.vocab_size}\n)"
    ).expandtabs(4)


def test_bigram_model_forward():
    model = get_bigram_model()
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 10])


def test_bigram_model_generate():
    model = get_bigram_model(vocab_size=5)
    model.device = torch.device("cpu")
    start_idx = 1
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(start_idx, itos)
    assert isinstance(generated, str)
    assert generated[0] == "H"
    assert len(generated) == model.max_new_tokens + 1
