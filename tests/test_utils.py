from models.registry import ModelRegistry as Model
import torch
import os
import json
from utils import (
    build_vocab,
    create_mappings,
    encode_data,
    decode_data,
    split_data,
    get_batch,
    get_metadata,
    load_config,
    get_config,
    get_model,
    save_checkpoint,
)


def get_models_config():
    return {
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
            "block_size": 3,
            "lr": 0.0015,
            "embedding_dim": 4,
            "hidden_size": 8,
            "num_layers": 1,
        },
    }


def get_test_config():
    return {
        "save_model": True,
        "models": {
            "bigram": get_models_config(),
            "lstm": get_models_config(),
            "transformer": get_models_config(),
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


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_build_vocab():
    chars, vocab_size = build_vocab("Hello, World!")
    assert chars == [" ", "!", ",", "H", "W", "d", "e", "l", "o", "r"]
    assert vocab_size == 10


def test_create_mappings():
    stoi, itos = create_mappings(["!", "H", "e", "l", "o"])
    assert stoi == {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    assert itos == {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}


def test_encode_data():
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    data = encode_data("Hello!", stoi)
    assert data.tolist() == [1, 2, 3, 3, 4, 0]
    assert data.dtype == torch.long


def test_decode_data():
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    data = torch.tensor([1, 2, 3, 3, 4, 0])
    decoded = decode_data(data, itos)
    assert decoded == "Hello!"


def test_split_data():
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_data, val_data = split_data(data)
    assert train_data.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert val_data.tolist() == [10]


def test_get_batch(tmp_path):
    torch.manual_seed(42)
    model = MockModel(str(tmp_path))
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x, y = get_batch(model, data)
    assert x.tolist() == [[2, 3, 4], [3, 4, 5]]
    assert y.tolist() == [[3, 4, 5], [4, 5, 6]]


def test_get_metadata(tmp_path):
    build_file(tmp_path, "metadata.json", '{"val_loss": 1.5}')
    path = tmp_path / "metadata.json"
    assert get_metadata(path, "val_loss", float("inf")) == 1.5


def test_load_config(tmp_path):
    build_file(tmp_path, "config.json", '{"bigram": {"val_loss": 1.5}}')
    assert load_config(tmp_path / "config.json") == {"bigram": {"val_loss": 1.5}}


def test_get_config(tmp_path):
    build_file(tmp_path, "config.json", '{"bigram": {"val_loss": 1.5}}')
    assert get_config(tmp_path / "config.json", "bigram") == {"val_loss": 1.5}


def test_get_model():
    config = get_test_config()["models"]
    bigram = get_model(Model, "bigram", config, "config.json", 10)
    lstm = get_model(Model, "lstm", config, "config.json", 10)
    transformer = get_model(Model, "transformer", config, "config.json", 10)
    assert bigram.__class__.__name__ == "BigramLanguageModel"
    assert lstm.__class__.__name__ == "LSTMLanguageModel"
    assert transformer.__class__.__name__ == "TransformerLanguageModel"


def test_save_checkpoint(tmp_path):
    model = MockModel(str(tmp_path))
    build_file(tmp_path, "config.json", '{"models": {"bigram": {"test": true}}}')
    save_checkpoint(model, step=10, val_loss=1.33, max_checkpoints=5)

    with open(model.meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert os.path.exists(model.ckpt_path)
    assert os.path.exists(model.meta_path)
    assert "timestamp" in metadata
    assert metadata["step"] == 10
    assert metadata["val_loss"] == 1.33
    assert metadata["config"] == {"test": True}
