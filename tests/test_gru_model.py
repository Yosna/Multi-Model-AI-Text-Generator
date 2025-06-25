import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_gru_config():
    return {
        "model_options": {},
        "runtime": {
            "training": True,
            "steps": 1,
            "interval": 1,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 1,
        },
        "hparams": {
            "batch_size": 2,
            "block_size": 4,
            "lr": 0.0015,
            "embedding_dim": 4,
            "hidden_size": 8,
            "num_layers": 1,
        },
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_gru_model(
    tmp_path,
    config: dict[str, Any] | None = None,
    cfg_path: str | None = None,
    vocab_size: int = 5,
    token_level: str = "char",
):
    if config is None:
        config = get_gru_config()
    if cfg_path is None:
        cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))

    return Model.GRULM(config, cfg_path, vocab_size, token_level)


def test_gru_model(tmp_path):
    model = get_gru_model(tmp_path)
    assert model is not None


def test_gru_model_no_vocab_size(tmp_path):
    with pytest.raises(ValueError):
        Model.GRULM(
            config=get_gru_config(),
            cfg_path=str(
                build_file(tmp_path, "config.json", json.dumps(get_gru_config()))
            ),
            vocab_size=0,
            token_level="char",
        )


def test_gru_model_init(tmp_path):
    model = get_gru_model(tmp_path)
    assert model.name == "gru"
    assert model.vocab_size == 5
    assert model.embedding_dim == 4
    assert model.hidden_size == 8
    assert model.num_layers == 1


def test_gru_model_embedding_layer(tmp_path):
    model = get_gru_model(tmp_path)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 5
    assert model.embedding.embedding_dim == 4


def test_gru_model_gru_layer(tmp_path):
    model = get_gru_model(tmp_path)
    assert isinstance(model.gru, nn.GRU)
    assert model.gru.input_size == 4
    assert model.gru.hidden_size == 8
    assert model.gru.num_layers == 1
    assert model.gru.batch_first


def test_gru_model_fc_layer(tmp_path):
    model = get_gru_model(tmp_path)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 8
    assert model.fc.out_features == 5
    assert model.fc.bias is not None


def test_gru_model_repr(tmp_path):
    model = get_gru_model(tmp_path)
    assert str(model) == (
        f"GRULanguageModel(\n"
        f"\tvocab_size={model.vocab_size},\n"
        f"\tembedding_dim={model.embedding_dim},\n"
        f"\thidden_size={model.hidden_size},\n"
        f"\tnum_layers={model.num_layers}\n"
        f")"
    ).expandtabs(4)


def test_gru_model_forward(tmp_path):
    model = get_gru_model(tmp_path)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 5])
    assert isinstance(model.hidden, torch.Tensor)
    assert len(model.hidden) == 1
    assert model.hidden.shape == torch.Size([1, 1, 8])


def test_gru_model_generate(tmp_path):
    model = get_gru_model(tmp_path)
    model.device = torch.device("cpu")
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(stoi, itos)
    assert isinstance(generated, str)
    assert all(char in stoi.keys() for char in generated)
    assert len(generated) == model.max_new_tokens + 1
