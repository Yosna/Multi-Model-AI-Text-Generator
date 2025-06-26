import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_lstm_config():
    return {
        "model_options": {
            "sampler": "multinomial",
            "save_model": True,
            "token_level": "char",
            "temperature": 1.0,
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


def get_lstm_model(
    tmp_path,
    config: dict[str, Any] = get_lstm_config(),
    vocab_size: int = 5,
):
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    model_options = config["model_options"]
    return Model.LSTMLM(config, cfg_path, vocab_size, model_options)


def test_lstm_model(tmp_path):
    model = get_lstm_model(tmp_path)
    assert model is not None


def test_lstm_model_no_vocab_size(tmp_path):
    config = get_lstm_config()
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    with pytest.raises(ValueError):
        Model.LSTMLM(
            config=config,
            cfg_path=cfg_path,
            vocab_size=0,
            model_options=config["model_options"],
        )


def test_lstm_model_init(tmp_path):
    model = get_lstm_model(tmp_path)
    assert model.name == "lstm"
    assert model.vocab_size == 5
    assert model.embedding_dim == 4
    assert model.hidden_size == 8
    assert model.num_layers == 1


def test_lstm_model_embedding_layer(tmp_path):
    model = get_lstm_model(tmp_path)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 5
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
    assert model.fc.out_features == 5
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
    assert logits.shape == torch.Size([1, 5, 5])
    assert isinstance(model.hidden, tuple)
    assert len(model.hidden) == 2
    assert model.hidden[0].shape == torch.Size([1, 1, 8])
    assert model.hidden[1].shape == torch.Size([1, 1, 8])


def test_lstm_model_generate(tmp_path):
    model = get_lstm_model(tmp_path)
    model.device = torch.device("cpu")
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(stoi, itos)
    assert isinstance(generated, str)
    assert all(char in stoi.keys() for char in generated)
    assert len(generated) == model.max_new_tokens + 1
