import argparse
import json
import sys
from unittest.mock import patch

import pytest

from cli import add_arg, parse_args, parse_config, set_arg_bool


def get_test_config():
    return {
        "model_options": {"model_options_passed": False},
        "models": {
            "mock": {
                "runtime": {"runtime_passed": False},
                "hparams": {"hparams_passed": False},
            }
        },
        "tuning_options": {"tuning_options_passed": False},
        "visualization": {"visualization_passed": False},
    }


def get_test_args():
    args = argparse.Namespace()
    args.model = "mock"
    args.model_options_passed = "true"
    args.runtime_passed = "true"
    args.hparams_passed = "true"
    args.tuning_options_passed = "true"
    args.visualization_passed = "true"
    return args


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_set_arg_bool():
    assert set_arg_bool("true") == True
    assert set_arg_bool("false") == False


def test_parse_config(tmp_path):
    args = get_test_args()
    config = get_test_config()
    cfg_path = build_file(tmp_path, "config.json", json.dumps(config))
    parse_config(args, cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        parsed_config = json.load(f)
    assert parsed_config["model_options"]["model_options_passed"]
    assert parsed_config["models"]["mock"]["runtime"]["runtime_passed"]
    assert parsed_config["models"]["mock"]["hparams"]["hparams_passed"]
    assert parsed_config["tuning_options"]["tuning_options_passed"]
    assert parsed_config["visualization"]["visualization_passed"]


def test_add_arg():
    parser = argparse.ArgumentParser()
    add_arg(parser, "--test", type=str, metavar="[str]", help="test help")
    args = parser.parse_args(["--test", "test_passed"])
    assert args.test == "test_passed"


def test_parse_args():
    cli_args = ["main.py"]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args is not None


def test_parse_args_default_model():
    cli_args = ["main.py"]
    default_model = "transformer"
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.model == default_model


@pytest.mark.parametrize(
    "model", ["bigram", "lstm", "gru", "transformer", "distilgpt2"]
)
def test_parse_args_model(model):
    cli_args = ["main.py", "--model", model]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.model == model


@pytest.mark.parametrize("invalid", ["test", "", None])
def test_parse_args_invalid(invalid):
    cli_args = ["main.py", "--model", invalid]
    with patch.object(sys, "argv", cli_args):
        with pytest.raises(SystemExit):
            parse_args()


def test_parse_args_model_options():
    cli_args = [
        "main.py",
        *["--sampler", "multinomial"],
        *["--save-model", "true"],
        *["--token-level", "char"],
        *["--temperature", "1.0"],
    ]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.sampler == "multinomial"
        assert args.save_model
        assert args.token_level == "char"
        assert args.temperature == 1.0


def test_parse_args_runtime():
    cli_args = [
        "main.py",
        *["--training", "true"],
        *["--steps", "100"],
        *["--interval", "10"],
        *["--patience", "10"],
        *["--max-new-tokens", "10"],
        *["--max-checkpoints", "10"],
    ]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.training
        assert args.steps == 100
        assert args.interval == 10
        assert args.patience == 10
        assert args.max_new_tokens == 10
        assert args.max_checkpoints == 10


def test_parse_args_hparams():
    cli_args = [
        "main.py",
        *["--batch-size", "10"],
        *["--block-size", "10"],
        *["--lr", "0.001"],
        *["--embedding-dim", "10"],
        *["--hidden-size", "10"],
        *["--num-layers", "10"],
        *["--max-seq-len", "10"],
        *["--num-heads", "10"],
        *["--ff-dim", "10"],
    ]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.batch_size == 10
        assert args.block_size == 10
        assert args.lr == 0.001
        assert args.embedding_dim == 10
        assert args.hidden_size == 10
        assert args.num_layers == 10
        assert args.max_seq_len == 10
        assert args.num_heads == 10
        assert args.ff_dim == 10


def test_parse_args_tuning_options():
    cli_args = [
        "main.py",
        *["--auto-tuning", "true"],
        *["--save-tuning", "true"],
        *["--save-study", "true"],
        *["--n-trials", "10"],
        *["--pruner", "hyperband"],
        *["--step-divisor", "10"],
    ]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.auto_tuning
        assert args.save_tuning
        assert args.save_study
        assert args.n_trials == 10
        assert args.pruner == "hyperband"
        assert args.step_divisor == 10


def test_parse_args_visualization():
    cli_args = [
        "main.py",
        *["--save-plot", "true"],
        *["--show-plot", "true"],
        *["--smooth-loss", "true"],
        *["--smooth-val-loss", "true"],
        *["--weight", "0.5"],
    ]
    with patch.object(sys, "argv", cli_args):
        args = parse_args()
        assert args.save_plot
        assert args.show_plot
        assert args.smooth_loss
        assert args.smooth_val_loss
        assert args.weight == 0.5
