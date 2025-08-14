import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from models.components.generators import (
    ArgmaxSampler,
    Generators,
    MultinomialSampler,
    PromptTextGenerator,
    RandomTextGenerator,
    Samplers,
    TextGenerator,
)
from models.registry import ModelRegistry as Model


def get_test_config(tmp_path):
    config = {
        "vocab": {
            "vocab_size": 10,
            "stoi": {str(i): i for i in range(10)},
            "itos": {i: str(i) for i in range(10)},
        },
        "generator_options": {
            "generator": "random",
            "context_length": 128,
            "sampler": "multinomial",
            "temperature": 1.0,
        },
        "model_options": {
            "save_model": True,
            "token_level": "char",
            "patience": 1,
            "max_checkpoints": 1,
        },
        "models": {
            "mock": {
                "runtime": {
                    "training": True,
                    "steps": 1,
                    "interval": 1,
                    "max_new_tokens": 10,
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


class MockModel(Model.BaseLM):
    def __init__(self, base_dir, vocab_size=10):
        config, cfg_path = get_test_config(base_dir)
        config = {**config["models"]["mock"], "vocab": config["vocab"]}
        super().__init__(model_name="mock", config=config, cfg_path=cfg_path)
        self.dir_path = os.path.join(base_dir, "checkpoints", "mock")
        self.ckpt_dir = os.path.join(self.dir_path, "checkpoint_1")
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pt")
        self.meta_path = os.path.join(self.dir_path, "meta.json")
        self.device = torch.device("cpu")
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.embedding(idx)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def get_test_mappings():
    stoi = {**{str(i): i for i in range(10)}, " ": 10}
    itos = {**{i: str(i) for i in range(10)}, 10: " "}
    return stoi, itos


def test_registries():
    assert Generators.Text.Random == RandomTextGenerator
    assert Generators.Text.Prompt == PromptTextGenerator
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
    tokens = [i for i in range(10)]
    idx = torch.tensor([tokens])
    logits = model(idx)
    token = sampler.get_next_token(logits)
    assert token.item() in tokens


def test_text_generator_init():
    stoi, itos = get_test_mappings()
    generator = TextGenerator(stoi, itos)
    assert generator is not None
    assert generator.stoi == stoi
    assert generator.itos == itos
    assert isinstance(generator.sampler, Samplers.Multinomial)


def test_text_generator_error():
    stoi, itos = get_test_mappings()
    generator = TextGenerator(stoi, itos)
    with pytest.raises(NotImplementedError):
        generator.tokens()


def test_random_text_generator_init():
    stoi, itos = get_test_mappings()
    generator = Generators.Text.Random(stoi, itos)
    assert generator is not None
    assert generator.start_idx in itos.keys()


def test_prompt_text_generator_init():
    stoi, itos = get_test_mappings()
    generator = Generators.Text.Prompt(128, stoi, itos)
    assert generator is not None
    assert generator.context_length == 128


@pytest.mark.parametrize(
    "generator, parameter",
    [
        (Generators.Text.Random, []),
        (Generators.Text.Prompt, [128]),
    ],
)
def test_text_generators_tokens(tmp_path, generator, parameter):
    stoi, itos = get_test_mappings()
    vocab_size = max(stoi.values()) + 1
    parameters = [*parameter, stoi, itos]
    generator = generator(*parameters)
    model = MockModel(tmp_path, vocab_size)
    with patch("builtins.input", return_value="0"):
        tokens = generator.tokens(model)
    assert isinstance(tokens, torch.Tensor)
    assert len(tokens) == model.max_new_tokens + 1


@pytest.mark.parametrize(
    "generator, parameter",
    [
        (Generators.Text.Random, []),
        (Generators.Text.Prompt, [128]),
    ],
)
def test_text_generators_output(tmp_path, generator, parameter):
    stoi, itos = get_test_mappings()
    vocab_size = max(stoi.values()) + 1
    parameters = [*parameter, stoi, itos]
    generator = generator(*parameters)
    model = MockModel(tmp_path, vocab_size)
    with patch("builtins.input", return_value="0"):
        output = generator.output(model)
    assert isinstance(output, str)
    assert len(output) == model.max_new_tokens + 1
