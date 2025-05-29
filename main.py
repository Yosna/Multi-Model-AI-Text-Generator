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
from training import train
from library import get_dataset
from typing import Any


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


def main(args: argparse.Namespace, cfg_path: str) -> None:
    """Prepare data, initialize model, and run training or generation."""
    model_name = args.model.lower()
    datasets = get_config(cfg_path, "datasets")
    text = get_dataset(datasets["source"], datasets["locations"])
    chars, vocab_size = build_vocab(text)
    stoi, itos = create_mappings(chars)
    data = encode_data(text, stoi)
    config = get_config(cfg_path, model_name)
    model = get_model(Model, model_name, cfg_path, vocab_size, **config["model"])

    validate_model(model, text, data, stoi, itos, **config["runtime"])


def validate_model(
    model: Model.BaseLM,
    text: str,
    data: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    **config: Any,
) -> None:
    """Validate the type of model to determine the appropriate run method."""
    if model.name == "transformer":
        generated_text = model.run(text, **config)
        print(generated_text)
    else:
        run_model(model, data, stoi, itos, **config)


def run_model(
    model: Model.BaseLM,
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
        try:
            model.load_state_dict(torch.load(model.ckpt_path))
        except Exception as e:
            print(f"Error loading model: {e}")

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
    main(parse_args(), "config.json")
