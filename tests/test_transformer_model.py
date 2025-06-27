import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_transformer_config(vocab_size=10):
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
            "steps": 1,
            "interval": 1,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 1,
        },
        "hparams": {
            "batch_size": 2,
            "block_size": 4,
            "lr": 0.001,
            "embedding_dim": 4,
            "max_seq_len": 8,
            "num_heads": 2,
            "ff_dim": 8,
            "num_layers": 1,
        },
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_transformer_model(tmp_path, config=None):
    if config is None:
        config = get_transformer_config()
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    return Model.TransformerLM(config, cfg_path)


def test_transformer_model(tmp_path):
    model = get_transformer_model(tmp_path)
    assert model is not None


def test_transformer_model_no_vocab_size(tmp_path):
    config = get_transformer_config(vocab_size=0)
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    with pytest.raises(ValueError):
        Model.TransformerLM(config, cfg_path)


def test_transformer_model_init(tmp_path):
    model = get_transformer_model(tmp_path)
    assert model.name == "transformer"
    assert model.vocab_size == 10
    assert model.embedding_dim == 4
    assert model.max_seq_len == 8
    assert model.num_heads == 2
    assert model.ff_dim == 8
    assert model.num_layers == 1


def test_transformer_model_token_embedding_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.token_embedding, nn.Embedding)
    assert model.token_embedding.num_embeddings == 10
    assert model.token_embedding.embedding_dim == 4


def test_transformer_model_position_embedding_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.position_embedding, nn.Embedding)
    assert model.position_embedding.num_embeddings == 8
    assert model.position_embedding.embedding_dim == 4


def test_transformer_model_encoder(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.transformer, nn.TransformerEncoder)
    assert len(model.transformer.layers) == 1


def test_transformer_model_fc_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 4
    assert model.fc.out_features == 10
    assert model.fc.bias is not None


def test_transformer_model_repr(tmp_path):
    model = get_transformer_model(tmp_path)
    assert str(model) == (
        f"TransformerLanguageModel(\n"
        f"\tvocab_size={model.vocab_size},\n"
        f"\tembedding_dim={model.embedding_dim},\n"
        f"\tmax_seq_len={model.max_seq_len},\n"
        f"\tnum_heads={model.num_heads},\n"
        f"\tff_dim={model.ff_dim},\n"
        f"\tnum_layers={model.num_layers}\n"
        f")"
    ).expandtabs(4)


def test_transformer_model_forward(tmp_path):
    model = get_transformer_model(tmp_path)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 10])
