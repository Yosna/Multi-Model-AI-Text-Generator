import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_bigram_config():
    return {
        "model_options": {},
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


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_bigram_model(
    tmp_path,
    config: dict[str, Any] | None = None,
    cfg_path: str | None = None,
    vocab_size: int = 10,
    token_level: str = "char",
):
    if config is None:
        config = get_bigram_config()
    if cfg_path is None:
        cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))

    return Model.BigramLM(config, cfg_path, vocab_size, token_level)


def test_bigram_model(tmp_path):
    model = get_bigram_model(tmp_path)
    assert model is not None


def test_bigram_model_vocab_errors(tmp_path):
    with pytest.raises(ValueError):
        Model.BigramLM(
            config=get_bigram_config(),
            cfg_path=build_file(
                tmp_path, "config.json", json.dumps(get_bigram_config())
            ),
            vocab_size=0,
            token_level="char",
        )
    with pytest.raises(ValueError):
        Model.BigramLM(
            config=get_bigram_config(),
            cfg_path=build_file(
                tmp_path, "config.json", json.dumps(get_bigram_config())
            ),
            vocab_size=100000,
            token_level="char",
        )


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


def test_bigram_model_generate(tmp_path):
    model = get_bigram_model(tmp_path, vocab_size=5)
    model.device = torch.device("cpu")
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(stoi, itos)
    assert isinstance(generated, str)
    assert all(char in stoi.keys() for char in generated)
    assert len(generated) == model.max_new_tokens + 1
