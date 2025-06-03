from models.registry import ModelRegistry as Model
import torch
import torch.nn as nn
import os
import sys
import json
import pytest
from unittest.mock import patch
from main import parse_args, main, validate_model, run_model


def get_test_config(tmp_path, training=True):
    return json.dumps(
        {
            "datasets": {
                "source": "local",
                "locations": {
                    "local": {
                        "directory": build_test_dataset(tmp_path),
                        "extension": "txt",
                    },
                },
            },
            "save_model": False,
            "models": {
                "bigram": get_test_model(training),
                "lstm": get_test_model(training),
                "gru": get_test_model(training),
                "transformer": get_test_model(training=False),
            },
            "auto_tuning": False,
            "visualization": {
                "show_plot": False,
                "smooth_loss": False,
                "smooth_val_loss": False,
                "weight": 1,
                "save_data": False,
            },
        }
    )


def get_test_model(training):
    return {
        "runtime": {
            "training": training,
            "steps": 1,
            "interval": 1,
            "patience": 1,
            "max_new_tokens": 1,
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


class MockModel(Model.BaseLM):
    def __init__(self, base_dir, model_name, vocab_size=5, training=True):
        config = json.loads(get_test_config(base_dir, training))["models"][model_name]
        super().__init__(
            model_name=model_name,
            config=config,
            cfg_path="config.json",
            vocab_size=vocab_size,
        )
        self.dir_path = os.path.join(base_dir, "checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")
        self.cfg_path = os.path.join(base_dir, "config.json")

        for key, value in config.get("hparams", {}).items():
            setattr(self, key, value)

        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(vocab_size, 1)

    def forward(self, *_, **__):
        return None, torch.tensor(1)

    def train_step(self, *_, **__):
        return torch.tensor(1)

    def run(self, *_, **__):
        return "Transformer output"

    def generate(self, *_, **__):
        return "Bigram & LSTM output"


def build_test_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    for i in range(100):
        build_file(dataset_dir, f"input_{i}.txt", f"Hello, World!")
    return str(dataset_dir)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_main(tmp_path, model):
    main_ran_successfully = False
    cli_args = ["main.py", "--model", model]
    with patch.object(sys, "argv", cli_args):
        try:
            main(
                parse_args(),
                build_file(tmp_path, "config.json", get_test_config(tmp_path)),
            )
            main_ran_successfully = True
        except Exception as e:
            print(e)
    assert main_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_validate_model(tmp_path, model):
    validate_model_ran_successfully = False
    build_file(tmp_path, "config.json", get_test_config(tmp_path))
    try:
        validate_model(
            model=MockModel(tmp_path, model),
            text="Hello!",
            data=torch.tensor([i for i in range(100)]),
            stoi={"!": 0, "H": 1, "e": 2, "l": 3, "o": 4},
            itos={0: "!", 1: "H", 2: "e", 3: "l", 4: "o"},
        )
        validate_model_ran_successfully = True
    except Exception as e:
        print(e)
    assert validate_model_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_run_model_training(tmp_path, model):
    run_model_ran_successfully = False
    build_file(tmp_path, "config.json", get_test_config(tmp_path))
    try:
        run_model(
            model=MockModel(tmp_path, model, training=True),
            data=torch.tensor([i for i in range(100)]),
            stoi={"!": 0, "H": 1, "e": 2, "l": 3, "o": 4},
            itos={0: "!", 1: "H", 2: "e", 3: "l", 4: "o"},
        )
        run_model_ran_successfully = True
    except Exception as e:
        print(e)
    assert run_model_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
def test_run_model_generation(tmp_path, model):
    run_model_ran_successfully = False
    try:
        run_model(
            model=MockModel(tmp_path, model, training=False),
            data=torch.tensor([i for i in range(100)]),
            stoi={"!": 0, "H": 1, "e": 2, "l": 3, "o": 4},
            itos={0: "!", 1: "H", 2: "e", 3: "l", 4: "o"},
        )
        run_model_ran_successfully = True
    except Exception as e:
        print(e)
    assert run_model_ran_successfully
