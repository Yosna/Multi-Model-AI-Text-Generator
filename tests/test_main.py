import json
import os
import sys
from pickle import UnpicklingError
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from main import main, parse_args, run_model, validate_model
from models.registry import ModelRegistry as Model


def get_test_config(tmp_path, training=True):
    return {
        "vocab": {
            "vocab_size": 100,
            "stoi": {str(i): i for i in range(100)},
            "itos": {i: str(i) for i in range(100)},
        },
        "datasets": {
            "source": "local",
            "locations": {
                "local": {
                    "directory": build_test_dataset(tmp_path),
                    "extension": "txt",
                },
            },
        },
        "generator_options": {
            "generator": "random",
            "context_length": 128,
            "sampler": "multinomial",
            "temperature": 1.0,
        },
        "model_options": {
            "save_model": False,
            "token_level": "char",
            "patience": 1,
            "max_checkpoints": 1,
        },
        "models": {
            "bigram": get_test_model(training),
            "lstm": get_test_model(training),
            "gru": get_test_model(training),
            "transformer": get_test_model(training),
            "distilgpt2": get_test_model(training=False),
        },
        "visualization": {
            "show_plot": False,
            "smooth_loss": False,
            "smooth_val_loss": False,
            "weight": 1,
            "save_data": False,
        },
    }


def get_test_model(training):
    return {
        "runtime": {
            "training": training,
            "steps": 1,
            "interval": 1,
            "max_new_tokens": 1,
        },
        "hparams": {
            "batch_size": 2,
            "block_size": 4,
            "lr": 0.0015,
            "embedding_dim": 4,
            "hidden_size": 8,
            "num_layers": 1,
            "max_seq_len": 16,
            "num_heads": 2,
            "ff_dim": 8,
        },
    }


class MockModel(Model.BaseLM):
    def __init__(self, base_dir, model, training=True):
        config = get_test_config(base_dir, training)
        config = {**config["models"][model], "vocab": config["vocab"]}
        cfg_path = os.path.join(base_dir, "config.json")
        super().__init__(model_name=model, config=config, cfg_path=cfg_path)
        self.dir_path = os.path.join(base_dir, "checkpoints", model)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.dir_path, "meta.json")
        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(100, 100)

    def forward(self, idx):
        return self.embedding(idx)

    def generate(self):
        return "test"


def build_test_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    for i in range(100):
        content = "".join([str(i) for i in range(100)])
        build_file(dataset_dir, f"input_{i}.txt", content)
    return str(dataset_dir)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


@pytest.mark.parametrize(
    "model", ["bigram", "lstm", "gru", "transformer", "distilgpt2"]
)
def test_main(tmp_path, model):
    main_ran_successfully = False
    cli_args = ["main.py", "--model", model]
    config = json.dumps(get_test_config(tmp_path))
    with patch.object(sys, "argv", cli_args):
        directory = os.getcwd()
        os.chdir(tmp_path)

        try:
            main(
                parse_args(),
                build_file(tmp_path, "config.json", config),
            )
            main_ran_successfully = True
        except Exception:
            pass
        os.chdir(directory)
    assert main_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_validate_model(tmp_path, model):
    validate_model_ran_successfully = False
    config = json.dumps(get_test_config(tmp_path))
    build_file(tmp_path, "config.json", config)
    try:
        model = MockModel(tmp_path, model)
        text = "".join([str(i) for i in range(100)])
        data = torch.tensor([i for i in range(100)])
        validate_model(model=model, text=text, data=data)
        validate_model_ran_successfully = True
    except Exception:
        pass
    assert validate_model_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_run_model_training(tmp_path, model):
    run_model_ran_successfully = False
    config = json.dumps(get_test_config(tmp_path))
    build_file(tmp_path, "config.json", config)
    try:
        model = MockModel(tmp_path, model, training=True)
        data = torch.tensor([i for i in range(100)])
        run_model(model=model, data=data)
        run_model_ran_successfully = True
    except Exception:
        pass
    assert run_model_ran_successfully


@pytest.mark.parametrize(
    "model", ["bigram", "lstm", "gru", "transformer", "distilgpt2"]
)
def test_run_model_generation(tmp_path, model):
    run_model_ran_successfully = False
    config = json.dumps(get_test_config(tmp_path, training=False))
    build_file(tmp_path, "config.json", config)
    try:
        model = MockModel(tmp_path, model, training=False)
        data = torch.tensor([i for i in range(100)])
        run_model(model=model, data=data)
        run_model_ran_successfully = True
    except Exception:
        pass
    assert run_model_ran_successfully


def test_run_model_load_error(tmp_path):
    config = json.dumps(get_test_config(tmp_path))
    build_file(tmp_path, "config.json", config)
    model = MockModel(tmp_path, "bigram", training=True)
    data = torch.tensor([i for i in range(100)])

    os.makedirs(model.dir_path, exist_ok=True)
    os.makedirs(model.ckpt_dir, exist_ok=True)
    with open(model.ckpt_path, "w") as f:
        f.write("Invalid state dict")

    error_loading_model = False
    try:
        run_model(model=model, data=data)
    except UnpicklingError:
        error_loading_model = True
    assert error_loading_model
