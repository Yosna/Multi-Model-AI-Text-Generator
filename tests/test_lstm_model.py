import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_lstm_config(vocab_size=10):
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


def get_lstm_model(tmp_path, config=None):
    if config is None:
        config = get_lstm_config()
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    return Model.LSTMLM(config, cfg_path)


def test_lstm_model(tmp_path):
    model = get_lstm_model(tmp_path)
    assert model is not None


def test_lstm_model_no_vocab_size(tmp_path):
    config = get_lstm_config(vocab_size=0)
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    with pytest.raises(ValueError):
        Model.LSTMLM(config, cfg_path)


def test_lstm_model_init(tmp_path):
    model = get_lstm_model(tmp_path)
    assert model.name == "lstm"
    assert model.vocab_size == 10
    assert model.embedding_dim == 4
    assert model.hidden_size == 8
    assert model.num_layers == 1


def test_lstm_model_embedding_layer(tmp_path):
    model = get_lstm_model(tmp_path)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 4


def test_lstm_model_lstm_layer(tmp_path):
    model = get_lstm_model(tmp_path)
    assert isinstance(model.lstm, nn.LSTM)
    assert model.lstm.input_size == 4
    assert model.lstm.hidden_size == 8
    assert model.lstm.num_layers == 1
    assert model.lstm.batch_first


def test_lstm_model_fc_layer(tmp_path):
    model = get_lstm_model(tmp_path)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 8
    assert model.fc.out_features == 10
    assert model.fc.bias is not None


def test_lstm_model_repr(tmp_path):
    model = get_lstm_model(tmp_path)
    assert str(model) == (
        f"LSTMLanguageModel(\n"
        f"\tvocab_size={model.vocab_size},\n"
        f"\tembedding_dim={model.embedding_dim},\n"
        f"\thidden_size={model.hidden_size},\n"
        f"\tnum_layers={model.num_layers}\n"
        f")"
    ).expandtabs(4)


def test_lstm_model_forward(tmp_path):
    model = get_lstm_model(tmp_path)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 10])
    assert isinstance(model.hidden, tuple)
    assert len(model.hidden) == 2
    assert model.hidden[0].shape == torch.Size([1, 1, 8])
    assert model.hidden[1].shape == torch.Size([1, 1, 8])
