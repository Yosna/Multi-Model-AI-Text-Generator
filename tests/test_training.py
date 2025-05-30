from models.registry import ModelRegistry as Model
import torch
import torch.nn as nn
import os
from training import train, validate_data


def get_runtime_config():
    return {
        "training": True,
        "batch_size": 2,
        "block_size": 4,
        "steps": 1,
        "interval": 1,
        "lr": 0.0015,
        "patience": 10,
        "max_new_tokens": 10,
        "max_checkpoints": 1,
    }


class MockModel(Model.BaseLM):
    def __init__(self, base_dir):
        super().__init__(
            model_name="mock",
            config=get_runtime_config(),
            cfg_path="config.json",
            vocab_size=10,
        )
        self.dir_path = os.path.join(base_dir, "checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")
        self.cfg_path = os.path.join(base_dir, "config.json")
        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(1, 1)

    def forward(self, idx, targets):
        return None, torch.tensor(1)

    def train_step(self, xb, yb, optimizer):
        return torch.tensor(1)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_train(tmp_path):
    test_config = """{
            "save_model": false,
            "mock": {},
            "visualization": {
                "show_plot": false,
                "smooth_loss": false,
                "smooth_val_loss": false,
                "weight": 1,
                "save_data": false
            }
        }"""
    build_file(tmp_path, "config.json", test_config)
    model = MockModel(str(tmp_path))
    losses, val_losses = train(model=model, data=torch.tensor([i for i in range(100)]))
    assert len(losses) == 1
    assert len(val_losses) == 1
    assert losses[0] == 1
    assert val_losses[0] == 1


def test_validate_data(tmp_path):
    build_file(tmp_path, "config.json", '{"save_model": true, "mock": {}}')
    overfit, best_loss, wait = validate_data(
        model=MockModel(str(tmp_path)),
        data=torch.tensor([i for i in range(100)]),
        step=1,
        loss=torch.tensor(0.0),
        best_loss=float("inf"),
        wait=0,
        val_losses=[],
    )
    assert overfit == False
    assert best_loss == torch.tensor(1)
    assert wait == 0
