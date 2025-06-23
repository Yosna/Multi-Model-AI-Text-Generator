import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.registry import ModelRegistry as Model


class BaseLanguageModel(Model.BaseLM):
    def __init__(self, model_name, config, cfg_path, vocab_size, token_level):
        super().__init__(model_name, config, cfg_path, vocab_size, token_level)
        self.embedding = nn.Embedding(vocab_size, vocab_size)  # type: ignore

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits

    def generate(self, start_idx, max_new_tokens):
        self.eval()
        idx = torch.tensor([[start_idx]], dtype=torch.long, device=self.device)
        generated = torch.tensor([start_idx], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            logits = self(idx)
            next_idx = self.new_token(logits)
            generated = torch.cat((generated, next_idx), dim=0)
        return generated


def get_base_model(tmp_path):
    cfg_path = build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    model = BaseLanguageModel(
        model_name="mock",
        config=get_test_config(),
        cfg_path=cfg_path,
        vocab_size=10,
        token_level="char",
    )
    model.dir_path = os.path.join(tmp_path, "checkpoints", "mock")
    model.plot_dir = os.path.join(tmp_path, "plots", "mock")
    model.ckpt_dir = os.path.join(model.dir_path, "checkpoint_1")
    model.ckpt_path = os.path.join(model.ckpt_dir, "checkpoint.pt")
    model.meta_path = os.path.join(model.ckpt_dir, "metadata.json")
    return model


def get_test_config():
    return {
        "model_options": {},
        "runtime": {
            "training": True,
            "batch_size": 2,
            "block_size": 4,
            "steps": 1,
            "interval": 1,
            "lr": 0.0015,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 10,
        },
    }


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_base_model(tmp_path):
    model = get_base_model(tmp_path)
    assert model is not None


def test_base_model_init(tmp_path):
    model = get_base_model(tmp_path)
    cfg_path = build_file(tmp_path, "config.json", json.dumps(get_test_config()))
    assert model.name == "mock"
    assert model.dir_path == os.path.join(tmp_path, "checkpoints", "mock")
    assert model.plot_dir == os.path.join(tmp_path, "plots", "mock")
    assert model.ckpt_dir == os.path.join(model.dir_path, "checkpoint_1")
    assert model.ckpt_path == os.path.join(model.ckpt_dir, "checkpoint.pt")
    assert model.meta_path == os.path.join(model.ckpt_dir, "metadata.json")
    assert model.cfg_path == cfg_path
    assert model.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model.vocab_size == 10


def test_base_model_train_step(tmp_path):
    model = get_base_model(tmp_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    xb = torch.tensor([[1, 2, 3, 4, 5]])
    yb = torch.tensor([[2, 3, 4, 5, 6]])
    loss = model.train_step(xb, yb, optimizer)
    assert loss > 0


def test_base_model_compute_loss(tmp_path):
    model = get_base_model(tmp_path)
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).repeat(1, 10, 1)
    targets = torch.tensor([[2, 3, 4, 5, 6]])
    loss = model.compute_loss(logits, idx, targets)
    assert loss is not None
    assert loss > 0


def test_base_model_check_patience_loss_improvement(tmp_path):
    model = get_base_model(tmp_path)
    overfit, best_loss, wait = model.check_patience(2, 1, 1)
    assert not overfit
    assert best_loss == 1
    assert wait == 0


def test_base_model_check_patience_no_loss_improvement(tmp_path):
    model = get_base_model(tmp_path)
    overfit, best_loss, wait = model.check_patience(1, 2, 9)
    assert overfit
    assert best_loss == 1
    assert wait == 10


def test_base_model_new_token(tmp_path):
    model = get_base_model(tmp_path)
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).repeat(1, 10, 1)
    next_idx = model.new_token(logits)
    assert isinstance(next_idx, torch.Tensor)
    assert next_idx.shape == torch.Size([1, 1])
    assert model.vocab_size is not None
    assert next_idx.item() in range(model.vocab_size)


def test_save_checkpoint(tmp_path):
    model = get_base_model(tmp_path)
    build_file(tmp_path, "config.json", '{"models": {"mock": {"test": true}}}')
    checkpoints = 0

    for i in range(11):
        model.save_checkpoint(step=i, val_loss=1.33)

    with open(model.meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for i in range(1, 11):
        checkpoint = os.path.join(model.dir_path, f"checkpoint_{i}")
        checkpoints += 1 if os.path.exists(checkpoint) else 0

    assert checkpoints == 10
    assert os.path.exists(model.ckpt_path)
    assert os.path.exists(model.meta_path)
    assert "timestamp" in metadata
    assert metadata["step"] == 10
    assert metadata["val_loss"] == 1.33
    assert metadata["config"] == {"test": True}


def test_not_implemented_methods(tmp_path):
    model = Model.BaseLM(model_name="mock", config={})
    with pytest.raises(NotImplementedError):
        model.generate()
    with pytest.raises(NotImplementedError):
        model.run()
