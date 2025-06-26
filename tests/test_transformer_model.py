import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


def get_transformer_config():
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
            "max_seq_len": 16,
            "num_heads": 2,
            "ff_dim": 8,
            "num_layers": 1,
        },
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_transformer_model(
    tmp_path,
    config: dict[str, Any] = get_transformer_config(),
    vocab_size: int = 5,
):
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    model_options = config["model_options"]

    return Model.TransformerLM(config, cfg_path, vocab_size, model_options)


def test_transformer_model(tmp_path):
    model = get_transformer_model(tmp_path)
    assert model is not None


def test_transformer_model_no_vocab_size(tmp_path):
    config = get_transformer_config()
    cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))
    with pytest.raises(ValueError):
        Model.TransformerLM(
            config=config,
            cfg_path=cfg_path,
            vocab_size=0,
            model_options=config["model_options"],
        )


def test_transformer_model_init(tmp_path):
    model = get_transformer_model(tmp_path)
    assert model.name == "transformer"
    assert model.vocab_size == 5
    assert model.embedding_dim == 4
    assert model.max_seq_len == 16
    assert model.num_heads == 2
    assert model.ff_dim == 8
    assert model.num_layers == 1


def test_transformer_model_token_embedding_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.token_embedding, nn.Embedding)
    assert model.token_embedding.num_embeddings == 5
    assert model.token_embedding.embedding_dim == 4


def test_transformer_model_position_embedding_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.position_embedding, nn.Embedding)
    assert model.position_embedding.num_embeddings == 16
    assert model.position_embedding.embedding_dim == 4


def test_transformer_model_encoder(tmp_path):
    model = get_transformer_model(tmp_path)
    encoder_layer = model.transformer.layers[0]
    out = encoder_layer(torch.randn(2, 4, 4))
    assert isinstance(model.transformer, nn.TransformerEncoder)
    assert len(model.transformer.layers) == model.num_layers
    assert isinstance(encoder_layer, nn.TransformerEncoderLayer)
    assert encoder_layer.self_attn.num_heads == 2
    assert encoder_layer.self_attn.embed_dim == 4
    assert encoder_layer.linear1.out_features == 8
    assert out.shape == (2, 4, 4)


def test_transformer_model_fc_layer(tmp_path):
    model = get_transformer_model(tmp_path)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 4
    assert model.fc.out_features == 5
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
    assert logits.shape == torch.Size([1, 5, 5])


def test_transformer_model_generate(tmp_path):
    model = get_transformer_model(tmp_path)
    model.device = torch.device("cpu")
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(stoi, itos)
    assert isinstance(generated, str)
    assert all(char in stoi.keys() for char in generated)
    assert len(generated) == model.max_new_tokens + 1
