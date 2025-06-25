import json
import os
from typing import cast

import pytest
import torch
import torch.nn as nn

from models.components.generators import (
    ArgmaxSampler,
    Generators,
    MultinomialSampler,
    RandomTextGenerator,
    Samplers,
)
from models.registry import ModelRegistry as Model


class MockModel(Model.BaseLM):
    def __init__(self, tmp_path):
        config, cfg_path = get_test_config(tmp_path)
        model_config = config["models"]["mock"]
        super().__init__(
            model_name="mock",
            config=model_config,
            cfg_path=cfg_path,
            vocab_size=5,
        )
        self.dir_path = os.path.join(tmp_path, "checkpoints", self.name)
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.ckpt_dir, "metadata.json")

        for key, value in model_config.get("hparams", {}).items():
            setattr(self, key, value)

        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(5, 5)

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits


def get_test_config(tmp_path):
    config = {
        "model_options": {
            "save_model": True,
            "token_level": "char",
            "auto_tuning": False,
            "save_tuning": False,
        },
        "models": {
            "mock": {
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
                    "block_size": 4,
                    "lr": 0.0015,
                },
            }
        },
    }
    cfg_path = build_file(tmp_path, "config.json", json.dumps(config))
    return config, cfg_path


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_test_mappings():
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    return stoi, itos


def test_registries():
    assert Generators.Text.Random == RandomTextGenerator
    assert Samplers.Multinomial == MultinomialSampler
    assert Samplers.Argmax == ArgmaxSampler


@pytest.mark.parametrize("sampler", [Samplers.Multinomial, Samplers.Argmax])
def test_samplers(sampler):
    sampler = sampler()
    assert sampler is not None
    if isinstance(sampler, MultinomialSampler):
        assert sampler.temperature == 1.0
    assert hasattr(sampler, "get_next_token")
    assert callable(sampler.get_next_token)


@pytest.mark.parametrize("sampler", [Samplers.Multinomial, Samplers.Argmax])
def test_samplers_get_next_token(tmp_path, sampler):
    sampler = sampler()
    model = MockModel(tmp_path)
    tokens = [i for i in range(5)]
    idx = torch.tensor([tokens])
    logits = model(idx)
    token = sampler.get_next_token(logits)
    assert token.item() in tokens


def test_random_text_generator_init():
    stoi, itos = get_test_mappings()
    generator = Generators.Text.Random(stoi, itos)
    assert generator is not None
    assert generator.start_idx in itos.keys()
    assert generator.stoi == stoi
    assert generator.itos == itos


def test_random_text_generator_tokens(tmp_path):
    stoi, itos = get_test_mappings()
    generator = Generators.Text.Random(stoi, itos)
    model = MockModel(tmp_path)
    token_range = range(cast(int, model.vocab_size))
    tokens = generator.tokens(model)
    assert tokens is not None
    assert len(tokens) == model.max_new_tokens + 1
    assert all(token in token_range for token in tokens)


def test_random_text_generator_output(tmp_path):
    stoi, itos = get_test_mappings()
    generator = Generators.Text.Random(stoi, itos)
    model = MockModel(tmp_path)
    output = generator.output(model)
    assert output is not None
    assert len(output) == model.max_new_tokens + 1
    assert all(char in stoi.keys() for char in output)
