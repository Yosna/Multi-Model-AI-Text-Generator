import torch
import torch.nn as nn
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
from models.registry import ModelRegistry
from training import train
from library import get_dataset
from typing import TypeVar

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a language model")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["bigram", "lstm", "transformer"],
        metavar="[bigram|lstm|transformer]",
        help="Model name to use from config.json",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Prepare data, initialize model, and run training or generation."""
    model_name = args.model.lower()
    datasets = get_config("config.json", "datasets")
    text = get_dataset(datasets["source"], datasets["locations"])
    chars, vocab_size = build_vocab(text)
    stoi, itos = create_mappings(chars)
    data = encode_data(text, stoi)
    config = get_config("config.json", model_name)
    model = get_model(ModelRegistry, model_name, vocab_size, **config["model"])

    validate_model(model, data, stoi, itos, **config["runtime"])


def validate_model(
    model: nn.Module,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    **config: T,
) -> None:
    """Validate the type of model to determine the appropriate run method."""
    if model.name == "transformer":
        generated_text = model.run(data, itos, **config)
        print(generated_text)
    else:
        run_model(model, data, stoi, itos, **config)


def run_model(
    model: nn.Module,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    training: bool,
    batch_size: int,
    block_size: int,
    steps: int,
    interval: int,
    lr: float,
    patience: int,
    max_new_tokens: int,
    max_checkpoints: int,
) -> None:
    """
    Run training or text generation for the model.
    Loads from checkpoint if available. Randomizes seed character for generation.
    """
    if os.path.exists(model.ckpt_path):
        model.load_state_dict(torch.load(model.ckpt_path))

    if training:
        train(
            model,
            data,
            batch_size,
            block_size,
            steps,
            interval,
            lr,
            patience,
            max_checkpoints,
        )
    else:
        seed_char = random.choice(list(stoi.keys()))
        start_idx = stoi[seed_char]
        generated_text = model.generate(start_idx, itos, max_new_tokens)
        print(generated_text)


if __name__ == "__main__":
    main(parse_args())
