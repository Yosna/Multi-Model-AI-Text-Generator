import json
import os
from unittest.mock import patch
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.components.generators import Samplers
from models.registry import ModelRegistry as Model


class BaseLanguageModel(Model.BaseLM):
    def __init__(self, model_name, config, cfg_path):
        super().__init__(model_name, config, cfg_path)
        vocab_size = config["vocab"]["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, vocab_size).to(self.device)
        self.vocab_size = vocab_size

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits


def get_test_config(vocab_size=10):
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
            "max_checkpoints": 10,
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
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_base_model(tmp_path, config=None):
    if config is None:
        config = get_test_config()
    path = build_file(tmp_path, "config.json", json.dumps(config))
    model = BaseLanguageModel(model_name="mock", config=config, cfg_path=path)
    model.dir_path = os.path.join(tmp_path, "checkpoints", "mock")
    model.plot_dir = os.path.join(tmp_path, "plots", "mock")
    model.ckpt_dir = os.path.join(model.dir_path, "checkpoint_1")
    model.ckpt_path = os.path.join(model.ckpt_dir, "checkpoint.pt")
    model.meta_path = os.path.join(model.ckpt_dir, "metadata.json")

    return model


def test_base_model(tmp_path):
    model = get_base_model(tmp_path)
    assert model is not None


def test_base_model_init(tmp_path):
    model = get_base_model(tmp_path)
    assert model.name == "mock"
    assert model.dir_path == os.path.join(tmp_path, "checkpoints", "mock")
    assert model.plot_dir == os.path.join(tmp_path, "plots", "mock")
    assert model.ckpt_dir == os.path.join(model.dir_path, "checkpoint_1")
    assert model.ckpt_path == os.path.join(model.ckpt_dir, "checkpoint.pt")
    assert model.meta_path == os.path.join(model.ckpt_dir, "metadata.json")
    assert model.cfg_path == tmp_path / "config.json"
    assert model.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model.vocab_size == 10


@pytest.mark.parametrize(
    "sampler, expected",
    [("multinomial", Samplers.Multinomial), ("argmax", Samplers.Argmax)],
)
def test_base_model_init_model_options(tmp_path, sampler, expected):
    config = get_test_config()
    config["generator_options"]["sampler"] = sampler
    model = get_base_model(tmp_path, config)
    assert isinstance(model.sampler, expected)
    assert model.temperature == 1.0
    assert model.save_model


@pytest.mark.parametrize("option", ["sampler", "generator"])
def test_base_model_invalid_generator_options(tmp_path, option):
    config = get_test_config()
    config["generator_options"][option] = "invalid"
    with pytest.raises(ValueError):
        get_base_model(tmp_path, config)


def test_base_model_train_step(tmp_path):
    model = get_base_model(tmp_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    xb = torch.tensor([[0, 1, 2, 3, 4]]).to(model.device)
    yb = torch.tensor([[1, 2, 3, 4, 0]]).to(model.device)
    loss = model.train_step(xb, yb, optimizer)
    assert loss > 0


def test_base_model_compute_loss(tmp_path):
    model = get_base_model(tmp_path)
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).repeat(1, 10, 1)
    targets = torch.tensor([[2, 3, 4, 5, 6]])
    loss = model.compute_loss(logits, idx, targets)
    assert loss is not None
    assert loss > 0


def test_base_model_check_patience_loss_improvement(tmp_path):
    model = get_base_model(tmp_path)
    overfit, best_loss, wait = model.check_patience(2, 1, 1)
    assert not overfit
    assert best_loss == 1
    assert wait == 0


def test_base_model_check_patience_no_loss_improvement(tmp_path):
    model = get_base_model(tmp_path)
    overfit, best_loss, wait = model.check_patience(1, 2, 9)
    assert overfit
    assert best_loss == 1
    assert wait == 10


def test_save_checkpoint(tmp_path):
    model = get_base_model(tmp_path)
    build_file(tmp_path, "config.json", '{"models": {"mock": {"test": true}}}')
    checkpoints = 0

    for i in range(11):
        model.save_checkpoint(step=i, val_loss=1.33)

    with open(model.meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for i in range(1, 11):
        checkpoint = os.path.join(model.dir_path, f"checkpoint_{i}")
        checkpoints += 1 if os.path.exists(checkpoint) else 0

    assert checkpoints == 10
    assert os.path.exists(model.ckpt_path)
    assert os.path.exists(model.meta_path)
    assert "timestamp" in metadata
    assert metadata["step"] == 10
    assert metadata["val_loss"] == 1.33
    assert metadata["config"] == {"test": True}


@pytest.mark.parametrize("generator", ["random", "prompt"])
def test_base_model_generators(tmp_path, generator):
    config = get_test_config()
    config["generator_options"]["generator"] = generator
    model = get_base_model(tmp_path, config)
    with patch("builtins.input", return_value="0"):
        generated = model.generate()
    assert isinstance(generated, str)
    assert all(char in [str(i) for i in range(10)] for char in generated)
    assert len(generated) == model.max_new_tokens + 1


def test_not_implemented_methods(tmp_path):
    model = get_base_model(tmp_path)
    with pytest.raises(NotImplementedError):
        model.run()
