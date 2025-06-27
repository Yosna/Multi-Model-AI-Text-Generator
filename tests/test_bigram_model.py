import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_bigram_config(vocab_size=10):
    return {
        "vocab": {
            "vocab_size": vocab_size,
            "stoi": {str(i): i for i in range(vocab_size or 1)},
            "itos": {i: str(i) for i in range(vocab_size or 1)},
        },
        "generator_options": {
            "generator": "random",
            "context_length": 128,
            "sampler": "multinomial",
            "temperature": 1.0,
        },
        "model_options": {
            "save_model": True,
            "token_level": "char",
            "patience": 10,
            "max_checkpoints": 1,
        },
        "runtime": {
            "training": True,
            "batch_size": 2,
            "block_size": 4,
            "steps": 1,
            "interval": 1,
            "lr": 0.0015,
            "max_new_tokens": 10,
        },
        "model": {},
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_bigram_model(tmp_path, config: dict[str, Any] | None = None):
    config = config or get_bigram_config()
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    return Model.BigramLM(config, cfg_path)


def test_bigram_model(tmp_path):
    model = get_bigram_model(tmp_path)
    assert model is not None


@pytest.mark.parametrize("vocab_size", [0, 100000])
def test_bigram_model_vocab_errors(tmp_path, vocab_size):
    config = get_bigram_config(vocab_size=vocab_size)
    cfg_path = build_file(tmp_path, "config.json", json.dumps(config))
    with pytest.raises(ValueError):
        Model.BigramLM(config, cfg_path)


def test_bigram_model_init(tmp_path):
    model = get_bigram_model(tmp_path)
    assert model.name == "bigram"
    assert model.vocab_size == 10


def test_bigram_model_embedding_layer(tmp_path):
    model = get_bigram_model(tmp_path)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 10


def test_bigram_model_repr(tmp_path):
    model = get_bigram_model(tmp_path)
    assert str(model) == (
        f"BigramLanguageModel(\n" f"\tvocab_size={model.vocab_size}\n)"
    ).expandtabs(4)


def test_bigram_model_forward(tmp_path):
    model = get_bigram_model(tmp_path)
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 10])
