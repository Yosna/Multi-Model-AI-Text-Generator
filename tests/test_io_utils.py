import json

import pytest

from utils.io_utils import get_config, get_metadata, load_config, save_config


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_get_metadata(tmp_path):
    meta_path = build_file(tmp_path, "metadata.json", '{"val_loss": 1.5}')
    assert get_metadata(meta_path, "val_loss", float("inf")) == 1.5


def test_save_config(tmp_path):
    cfg_path = build_file(tmp_path, "config.json", '{"test_passed": false}')
    config = {"test_passed": True}
    save_config(config, cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        test_passed = json.load(f)["test_passed"]
    assert test_passed


def test_load_config(tmp_path):
    cfg_path = build_file(tmp_path, "config.json", '{"bigram": {"val_loss": 1.5}}')
    assert load_config(cfg_path) == {"bigram": {"val_loss": 1.5}}


def test_get_config(tmp_path):
    cfg_path = build_file(tmp_path, "config.json", '{"bigram": {"val_loss": 1.5}}')
    assert get_config(cfg_path, "bigram") == {"val_loss": 1.5}


def test_get_config_errors(tmp_path):
    cfg_path = build_file(tmp_path, "config.json", '{"invalid_value": null}')
    with pytest.raises(ValueError):
        get_config(cfg_path, "invalid_value")
    with pytest.raises(KeyError):
        get_config(cfg_path, "invalid_key")
