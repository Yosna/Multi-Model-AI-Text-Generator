from models.registry import ModelRegistry as Model
import torch
import torch.nn as nn
import os
from typing import Any


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


class BaseLanguageModel(Model.BaseLM):
    def __init__(
        self,
        model_name: str = "test",
        config: dict[str, Any] = get_runtime_config(),
        cfg_path: str = "config.json",
        vocab_size: int = 10,
    ):
        super().__init__(model_name, config, cfg_path, vocab_size)
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.embedding(idx)
        logits, loss = self.compute_loss(idx, logits, targets)
        return logits, loss

    def generate(self, start_idx, max_new_tokens):
        self.eval()
        idx = torch.tensor([[start_idx]], dtype=torch.long, device=self.device)
        generated = torch.tensor([start_idx], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            next_idx = self.new_token(logits)
            generated = torch.cat((generated, next_idx), dim=0)
        return generated


def test_base_model():
    model = BaseLanguageModel()
    assert model is not None


def test_base_model_init():
    model = BaseLanguageModel()
    assert model.name == "test"
    assert model.dir_path == os.path.join("checkpoints", "test")
    assert model.plot_dir == os.path.join("plots", "test")
    assert model.ckpt_dir == os.path.join(model.dir_path, "checkpoint_1")
    assert model.ckpt_path == os.path.join(model.ckpt_dir, "checkpoint.pt")
    assert model.meta_path == os.path.join(model.ckpt_dir, "metadata.json")
    assert model.cfg_path == "config.json"
    assert model.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model.vocab_size == 10


def test_base_model_train_step():
    model = BaseLanguageModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    xb = torch.tensor([[1, 2, 3, 4, 5]])
    yb = torch.tensor([[2, 3, 4, 5, 6]])
    loss = model.train_step(xb, yb, optimizer)
    assert loss > 0


def test_base_model_compute_loss():
    model = BaseLanguageModel()
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).repeat(1, 10, 1)
    targets = torch.tensor([[2, 3, 4, 5, 6]])
    logits, loss = model.compute_loss(idx, logits, targets)
    assert logits.shape == torch.Size([5, 10])
    assert loss is not None
    assert loss > 0


def test_base_model_new_token():
    model = BaseLanguageModel()
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).repeat(1, 10, 1)
    next_idx = model.new_token(logits)
    assert isinstance(next_idx, torch.Tensor)
    assert next_idx.shape == torch.Size([1, 1])
    assert model.vocab_size is not None
    assert next_idx.item() in range(model.vocab_size)
