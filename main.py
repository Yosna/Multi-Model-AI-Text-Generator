"""
Main entry point for running and training language models.

Handles argument parsing, dataset loading, model initialization, and dispatches
training or text generation based on configuration and model type.
"""

from models.registry import ModelRegistry as Model
import torch
import random
import os
import argparse
from utils import (
    get_config,
    build_vocab,
    create_mappings,
    encode_data,
    get_model,
)
from cli import parse_args, parse_config
from tuning import optimize_and_train
from library import get_dataset
from typing import Any


def main(args: argparse.Namespace, cfg_path: str = "config.json") -> None:
    """
    Prepare data, initialize model, and run training or generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cfg_path (str): Path to the configuration file.
    """
    parse_config(args, cfg_path)

    model_name = args.model.lower()
    token_level = get_config(cfg_path, "model_options").get("token_level", "char")
    datasets = get_config(cfg_path, "datasets")
    text = get_dataset(datasets["source"], datasets["locations"])
    tokens, vocab, vocab_size = build_vocab(text, token_level)
    stoi, itos = create_mappings(vocab)
    data = encode_data(tokens, stoi)
    config = get_config(cfg_path, "models")
    model = get_model(Model, model_name, config, cfg_path, vocab_size, token_level)

    validate_model(model, text, data, stoi, itos)


def validate_model(
    model: Model.BaseLM,
    text: str,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    """
    Validate the type of model to determine the appropriate run method.

    Args:
        model (Model.BaseLM): The model instance.
        text (str): The full dataset text.
        data (torch.Tensor): Encoded dataset tensor.
        stoi (dict[str, int]): Character-to-index mapping.
        itos (dict[int, str]): Index-to-character mapping.
    """
    if model.name == "distilgpt2":
        generated_text = model.run(text)
        print(generated_text)
    else:
        run_model(model, data, stoi, itos)


def run_model(
    model: Model.BaseLM,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    """
    Run training or text generation for the model.
    Loads from checkpoint if available.
    Randomizes seed character for generation.

    Args:
        model (Model.BaseLM): The model instance.
        data (torch.Tensor): Encoded dataset tensor.
        stoi (dict[str, int]): Character-to-index mapping.
        itos (dict[int, str]): Index-to-character mapping.

    Returns:
        None
    """
    if os.path.exists(model.ckpt_path):
        try:
            model.load_state_dict(torch.load(model.ckpt_path))
        except Exception as e:
            print(f"Error loading model: {e}")

    if model.training:
        optimize_and_train(model, data)
    else:
        model_options = get_config(model.cfg_path, "model_options")
        temperature = model_options.get("temperature", 1.0)
        seed_char = random.choice(list(stoi.keys()))
        start_idx = stoi[seed_char]
        generated_text = model.generate(start_idx, itos, temperature)
        print(generated_text)


if __name__ == "__main__":
    main(parse_args())
