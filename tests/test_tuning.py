import json
import os

import optuna
import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model
from tuning import create_pruner, make_objective, optimize_and_train


def get_test_config():
    return {
        "model_options": {
            "sampler": "multinomial",
            "save_model": False,
            "token_level": "char",
            "temperature": 1.0,
        },
        "models": {
            "bigram": {
                "runtime": {
                    "training": True,
                    "steps": 1000,
                    "interval": 10,
                    "patience": 10,
                    "max_new_tokens": 10,
                    "max_checkpoints": 1,
                },
                "hparams": {
                    "batch_size": 2,
                    "block_size": 4,
                    "lr": 0.0015,
                    "num_layers": 2,
                },
            }
        },
        "pruners": {
            "median": {
                "n_startup_trials": 10,
                "n_warmup_steps": 10,
            },
            "halving": {
                "min_resource": 10,
                "reduction_factor": 10,
                "min_early_stopping_rate": 10,
            },
            "hyperband": {
                "min_resource": 10,
                "reduction_factor": 10,
            },
        },
        "tuning_options": {
            "auto_tuning": True,
            "save_tuning": True,
            "save_study": False,
            "n_trials": 1,
            "pruner": "",
            "step_divisor": 10,
        },
        "tuning_ranges": {
            "batch_size": {
                "type": "int",
                "min": 4,
                "max": 12,
                "step": 1,
            },
            "block_size": {
                "type": "int",
                "min": 8,
                "max": 24,
                "step": 1,
            },
            "lr": {"type": "float", "min": 0.001, "max": 0.002, "log": False},
            "num_layers": {"type": "categorical", "values": [1, 2]},
        },
        "visualization": {
            "save_plot": False,
            "show_plot": False,
            "smooth_loss": False,
            "smooth_val_loss": False,
            "weight": 1,
        },
    }


class MockModel(Model.BaseLM):
    def __init__(self, base_dir):
        config = get_test_config()
        bigram_config = config["models"]["bigram"]
        super().__init__(
            model_name="bigram",
            config=bigram_config,
            cfg_path=os.path.join(base_dir, "config.json"),
            vocab_size=100,
            model_options=config["model_options"],
        )

        for key, value in bigram_config.get("hparams", {}).items():
            setattr(self, key, value)

        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(100, 100)

    def forward(self, idx):
        return self.embedding(idx)

    def train_step(self, *_, **__):
        return torch.tensor(1)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_optimize_and_train(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    data = torch.tensor([i for i in range(100)])
    losses, val_losses = optimize_and_train(model, data)
    assert len(losses) > 0
    assert len(val_losses) > 0
    assert losses[0] > 0
    assert val_losses[0] > 0


def test_make_objective(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    data = torch.tensor([i for i in range(100)])
    objective = make_objective(model, data)
    assert objective is not None
    assert callable(objective)


def test_make_objective_error(tmp_path):
    build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = MockModel(str(tmp_path))
    data = torch.tensor([i for i in range(100)])
    with pytest.raises(ValueError):
        model.vocab_size = 0
        make_objective(model, data)
    with pytest.raises(ValueError):
        model.vocab_size = None
        make_objective(model, data)


@pytest.mark.parametrize("pruner", ["median", "halving", "hyperband"])
def test_create_pruner_no_config(tmp_path, pruner):
    config = get_test_config()
    config["pruners"] = {}
    config["tuning_options"]["pruner"] = pruner
    build_file(tmp_path, "config.json", json.dumps(config))
    model = MockModel(str(tmp_path))
    pruner = create_pruner(model)
    assert pruner is not None
    assert isinstance(pruner, optuna.pruners.BasePruner)


def test_create_pruner_median(tmp_path):
    config = get_test_config()
    config["tuning_options"]["pruner"] = "median"
    build_file(tmp_path, "config.json", json.dumps(config))
    model = MockModel(str(tmp_path))
    pruner = create_pruner(model)
    assert isinstance(pruner, optuna.pruners.MedianPruner)
    assert pruner._n_startup_trials == 10
    assert pruner._n_warmup_steps == 10


def test_create_pruner_halving(tmp_path):
    config = get_test_config()
    config["tuning_options"]["pruner"] = "halving"
    build_file(tmp_path, "config.json", json.dumps(config))
    model = MockModel(str(tmp_path))
    pruner = create_pruner(model)
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert pruner._min_resource == 10
    assert pruner._reduction_factor == 10
    assert pruner._min_early_stopping_rate == 10


def test_create_pruner_hyperband(tmp_path):
    config = get_test_config()
    config["tuning_options"]["pruner"] = "hyperband"
    build_file(tmp_path, "config.json", json.dumps(config))
    model = MockModel(str(tmp_path))
    pruner = create_pruner(model)
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)
    assert pruner._min_resource == 10
    assert pruner._max_resource == 100
    assert pruner._reduction_factor == 10


def test_create_pruner_none(tmp_path):
    config = get_test_config()
    config["tuning_options"]["pruner"] = "invalid"
    build_file(tmp_path, "config.json", json.dumps(config))
    model = MockModel(str(tmp_path))
    pruner = create_pruner(model)
    assert pruner is None
