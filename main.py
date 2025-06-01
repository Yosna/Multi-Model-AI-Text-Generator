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
from tuning import optimize_and_train
from library import get_dataset
from typing import Any


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model selection.

    Returns:
        argparse.Namespace: Parsed arguments with model selection.
    """
    parser = argparse.ArgumentParser(description="Run a language model")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["bigram", "lstm", "gru", "transformer"],
        metavar="[bigram|lstm|gru|transformer]",
        help="Model name to use from config.json",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, cfg_path: str) -> None:
    """
    Prepare data, initialize model, and run training or generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cfg_path (str): Path to the configuration file.
    """
    model_name = args.model.lower()
    datasets = get_config(cfg_path, "datasets")
    text = get_dataset(datasets["source"], datasets["locations"])
    chars, vocab_size = build_vocab(text)
    stoi, itos = create_mappings(chars)
    data = encode_data(text, stoi)
    config = get_config(cfg_path, "models")
    model = get_model(Model, model_name, config, cfg_path, vocab_size)

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
    if model.name == "transformer":
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
        seed_char = random.choice(list(stoi.keys()))
        start_idx = stoi[seed_char]
        generated_text = model.generate(start_idx, itos)
        print(generated_text)


if __name__ == "__main__":
    main(parse_args(), "config.json")
