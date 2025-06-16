from models.registry import ModelRegistry as Model
import pytest
import torch
import os
from typing import cast
from utils.model_utils import build_vocab, create_mappings, get_batch, get_model


def get_test_config():
    return {
        "model_options": {
            "save_model": True,
            "token_level": "char",
            "auto_tuning": False,
            "save_tuning": False,
        },
        "models": {
            "bigram": get_models_config(),
            "lstm": get_models_config(),
            "gru": get_models_config(),
            "distilgpt2": get_models_config(),
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


def get_models_config():
    return {
        "runtime": {
            "training": True,
            "steps": 1,
            "interval": 1,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 10,
        },
        "hparams": {
            "batch_size": 2,
            "block_size": 3,
            "lr": 0.0015,
            "embedding_dim": 4,
            "hidden_size": 8,
            "num_layers": 1,
        },
    }


class MockModel(Model.BaseLM):
    def __init__(self, base_dir):
        config = get_test_config()["models"]["bigram"]
        super().__init__(
            model_name="bigram",
            config=config,
            cfg_path=os.path.join(base_dir, "config.json"),
            vocab_size=10,
        )
        self.dir_path = os.path.join(base_dir, "checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")

        for key, value in config.get("hparams", {}).items():
            setattr(self, key, value)

        self.device = torch.device("cpu")


def test_build_vocab_char():
    tokens, vocab, vocab_size = build_vocab("Hello, World!", token_level="char")
    assert tokens == ["H", "e", "l", "l", "o", ",", " ", "W", "o", "r", "l", "d", "!"]
    assert vocab == [" ", "!", ",", "H", "W", "d", "e", "l", "o", "r"]
    assert vocab_size == 10


def test_build_vocab_word():
    tokens, vocab, vocab_size = build_vocab("Hello, World!", token_level="word")
    assert tokens == ["Hello,", "World!"]
    assert vocab == ["Hello,", "World!"]
    assert vocab_size == 2


def test_build_vocab_errors():
    with pytest.raises(ValueError):
        build_vocab("Hello, World!", token_level="invalid")


def test_create_mappings():
    stoi, itos = create_mappings(["!", "H", "e", "l", "o"])
    assert stoi == {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    assert itos == {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}


def test_get_batch(tmp_path):
    torch.manual_seed(42)
    model = MockModel(str(tmp_path))
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    block_size = cast(int, model.block_size)
    batch_size = cast(int, model.batch_size)
    x, y = get_batch(block_size, batch_size, data, model.device)
    assert x.tolist() == [[2, 3, 4], [3, 4, 5]]
    assert y.tolist() == [[3, 4, 5], [4, 5, 6]]


def test_get_model():
    config = get_test_config()["models"]
    bigram = get_model("bigram", config, "config.json", 10, "char")
    lstm = get_model("lstm", config, "config.json", 10, "char")
    gru = get_model("gru", config, "config.json", 10, "char")
    distilgpt2 = get_model("distilgpt2", config, "config.json", 10, "char")
    assert bigram.__class__.__name__ == "BigramLanguageModel"
    assert lstm.__class__.__name__ == "LSTMLanguageModel"
    assert gru.__class__.__name__ == "GRULanguageModel"
    assert distilgpt2.__class__.__name__ == "DistilGPT2LanguageModel"


def test_get_model_error():
    config = get_test_config()["models"]
    with pytest.raises(ValueError):
        get_model("test", config, "config.json", 10, "char")
