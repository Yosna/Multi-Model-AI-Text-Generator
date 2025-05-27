import torch
import torch.nn as nn
import os
import sys
import json
import pytest
from unittest.mock import patch
from main import parse_args, main, validate_model, run_model


class MockModel(nn.Module):
    def __init__(self, base_dir, model_name, vocab_size=5):
        super().__init__()
        self.name = model_name
        self.dir_path = os.path.join(base_dir, "checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")
        self.cfg_path = os.path.join(base_dir, "config.json")
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 1)

    def forward(self, idx, targets):
        return None, torch.tensor(1)

    def train_step(self, xb, yb, optimizer):
        return torch.tensor(1)

    def run(self, text, **config):
        return "Transformer output"

    def generate(self, start_idx, itos, max_new_tokens):
        return "Bigram & LSTM output"


def get_test_config(tmp_path):
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
            "bigram": {
                "runtime": {
                    "training": True,
                    "batch_size": 1,
                    "block_size": 1,
                    "steps": 1,
                    "interval": 1,
                    "lr": 1,
                    "patience": 1,
                    "max_new_tokens": 1,
                    "max_checkpoints": 1,
                },
                "model": {},
            },
            "lstm": {
                "runtime": {
                    "training": True,
                    "batch_size": 1,
                    "block_size": 1,
                    "steps": 1,
                    "interval": 1,
                    "lr": 1,
                    "patience": 1,
                    "max_new_tokens": 1,
                    "max_checkpoints": 1,
                },
                "model": {
                    "embedding_dim": 1,
                    "hidden_size": 1,
                    "num_layers": 1,
                },
            },
            "transformer": {
                "runtime": {
                    "block_size": 1,
                    "max_new_tokens": 1,
                },
                "model": {},
            },
            "visualization": {
                "show_plot": False,
                "smooth_loss": False,
                "smooth_val_loss": False,
                "weight": 1,
                "save_data": False,
            },
        }
    )


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


def test_parse_args():
    cli_args = ["main.py"]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args is not None


def test_parse_args_default():
    cli_args = ["main.py"]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.model == "transformer"


@pytest.mark.parametrize("model", ["bigram", "lstm", "transformer"])
def test_parse_args_model(model):
    cli_args = ["main.py", "--model", model]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.model == model


@pytest.mark.parametrize("invalid", ["test", "", None])
def test_parse_args_invalid(invalid):
    cli_args = ["main.py", "--model", invalid]
    with patch.object(sys, "argv", cli_args):
        try:
            parse_args()
        except SystemExit as e:
            assert e.code == 2


@pytest.mark.parametrize("model", ["bigram", "lstm", "transformer"])
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


@pytest.mark.parametrize("model", ["bigram", "lstm", "transformer"])
def test_validate_model(tmp_path, model):
    validate_model_ran_successfully = False
    config_path = build_file(tmp_path, "config.json", get_test_config(tmp_path))
    config = json.loads(config_path.read_text())
    try:
        validate_model(
            model=MockModel(str(tmp_path), model),
            text="Hello!",
            data=torch.tensor([i for i in range(100)]),
            stoi={"!": 0, "H": 1, "e": 2, "l": 3, "o": 4},
            itos={0: "!", 1: "H", 2: "e", 3: "l", 4: "o"},
            **config[model]["runtime"],
        )
        validate_model_ran_successfully = True
    except Exception as e:
        print(e)
    assert validate_model_ran_successfully


@pytest.mark.parametrize("model", ["bigram", "lstm"])
def test_run_model(tmp_path, model):
    run_model_ran_successfully = False
    config_path = build_file(tmp_path, "config.json", get_test_config(tmp_path))
    config = json.loads(config_path.read_text())
    try:
        run_model(
            model=MockModel(str(tmp_path), model),
            data=torch.tensor([i for i in range(100)]),
            stoi={"!": 0, "H": 1, "e": 2, "l": 3, "o": 4},
            itos={0: "!", 1: "H", 2: "e", 3: "l", 4: "o"},
            **config[model]["runtime"],
        )
        run_model_ran_successfully = True
    except Exception as e:
        print(e)
    assert run_model_ran_successfully
