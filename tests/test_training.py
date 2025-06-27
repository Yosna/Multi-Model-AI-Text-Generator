import json
import os

import pytest
import torch
import torch.nn as nn
from optuna import TrialPruned

from models.registry import ModelRegistry as Model
from training import train, validate_data


def get_test_config():
    return {
        "vocab": {
            "vocab_size": 100,
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
        "models": {
            "mock": {
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
                },
            }
        },
        "visualization": {
            "show_plot": False,
            "smooth_loss": False,
            "smooth_val_loss": False,
            "weight": 1,
            "save_data": False,
        },
    }


class MockModel(Model.BaseLM):
    def __init__(self, base_dir):
        config = get_test_config()
        config = {**config["models"]["mock"], "vocab": config["vocab"]}
        cfg_path = os.path.join(base_dir, "config.json")
        super().__init__(model_name="mock", config=config, cfg_path=cfg_path)
        self.dir_path = os.path.join(base_dir, "checkpoints", "mock")
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.dir_path, "meta.json")
        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(100, 100)

    def forward(self, idx):
        return self.embedding(idx)

    def generate(self):
        return "test"

    def train_step(self, *_, **__):
        return torch.tensor(1)


class MockTrial:
    def should_prune(self):
        return True

    def report(self, *_, **__):
        pass


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_train(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    losses, val_losses = train(model=model, data=torch.tensor([i for i in range(100)]))
    assert len(losses) == 1
    assert len(val_losses) == 1
    assert losses[0] > 0
    assert val_losses[0] > 0


def test_validate_data(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    overfit, best_loss, wait = validate_data(
        model=model,
        data=torch.tensor([i for i in range(100)]),
        step=1,
        step_divisor=1,
        loss=0.0,
        best_loss=float("inf"),
        wait=0,
        val_losses=[],
    )
    assert overfit == False
    assert best_loss > 0
    assert wait == 0


def test_trial_pruning(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    data = torch.tensor([i for i in range(100)])
    trial = MockTrial()
    with pytest.raises(TrialPruned):
        train(model, data, trial)  # type: ignore
