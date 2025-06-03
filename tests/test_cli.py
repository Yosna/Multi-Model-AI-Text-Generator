from unittest.mock import patch
import sys
import argparse
import pytest
import json
from cli import set_arg_bool, parse_config, parse_args


def test_set_arg_bool():
    assert set_arg_bool("true") == True
    assert set_arg_bool("false") == False


def get_test_config():
    return {"models": {"mock": {"runtime": {"test_passed": False}}}}


def get_test_args():
    args = argparse.Namespace()
    args.model = "mock"
    args.test_passed = "true"
    return args


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_parse_config(tmp_path):
    args = get_test_args()
    config = get_test_config()
    cfg_path = build_file(tmp_path, "config.json", json.dumps(config))
    parse_config(args, cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        test_passed = json.load(f)["models"]["mock"]["runtime"]["test_passed"]
    assert test_passed


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


@pytest.mark.parametrize("model", ["bigram", "lstm", "gru", "transformer"])
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
