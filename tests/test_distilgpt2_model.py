import json
import os
from typing import Any

from models.registry import ModelRegistry as Model


def get_distilgpt2_config():
    return {
        "vocab": {
            "vocab_size": 50257,
            "stoi": {str(i): i for i in range(100)},
            "itos": {i: str(i) for i in range(100)},
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
        },
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_distilgpt2_model(
    tmp_path,
    config: dict[str, Any] | None = None,
    cfg_path: str | None = None,
):
    if config is None:
        config = get_distilgpt2_config()
    if cfg_path is None:
        cfg_path = str(build_file(tmp_path, "config.json", json.dumps(config)))

    return Model.DistilGPT2LM(config, cfg_path)


def test_distilgpt2_model(tmp_path):
    model = get_distilgpt2_model(tmp_path)
    assert model is not None


def test_distilgpt2_model_init(tmp_path):
    model = get_distilgpt2_model(tmp_path)
    assert model.name == "distilgpt2"
    assert model.tokenizer is not None
    assert model.model is not None


def test_distilgpt2_model_tokenizer(tmp_path):
    model = get_distilgpt2_model(tmp_path)
    assert model.tokenizer.name_or_path == "distilgpt2"
    assert model.tokenizer.model_max_length == 1024
    assert model.tokenizer.vocab_size == 50257


def test_distilgpt2_model_model(tmp_path):
    model = get_distilgpt2_model(tmp_path)
    assert model.model.name_or_path == "distilgpt2"
    assert model.model.config.architectures == ["GPT2LMHeadModel"]
    assert model.model.config.model_type == "gpt2"
    assert model.model.config.vocab_size == 50257


def test_distilgpt2_model_run(tmp_path):
    model = get_distilgpt2_model(tmp_path)
    text = "Hello!"
    result = model.run(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result[: model.block_size] in text
