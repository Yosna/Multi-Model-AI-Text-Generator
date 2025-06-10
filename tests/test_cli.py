from unittest.mock import patch
import pytest
import sys
import argparse
import pytest
import json
from cli import set_arg_bool, add_arg, parse_config, parse_args


def get_test_config():
    return {
        "models": {
            "mock": {
                "runtime": {"runtime_passed": False},
                "hparams": {"hparams_passed": False},
            }
        }
    }


def get_test_args():
    args = argparse.Namespace()
    args.model = "mock"
    args.runtime_passed = "true"
    args.hparams_passed = "true"
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
    assert parsed_config["models"]["mock"]["runtime"]["runtime_passed"]
    assert parsed_config["models"]["mock"]["hparams"]["hparams_passed"]


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
