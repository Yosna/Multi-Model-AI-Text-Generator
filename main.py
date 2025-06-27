"""Main entry point for running and training language models.

Handles argument parsing, dataset loading, model initialization, and dispatches
training or text generation based on configuration and model type.
"""

import argparse
import logging
import os
from pickle import UnpicklingError

import torch

from cli import parse_args, parse_config
from library import get_dataset
from models.registry import ModelRegistry as Model
from tuning import optimize_and_train
from utils.data_utils import encode_data
from utils.model_utils import build_vocab, create_mappings, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace, cfg_path: str = "config.json") -> None:
    """Prepare data, initialize model, and run training or generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cfg_path (str): Path to the configuration file.
    """
    logger.info(f"Starting main execution with model: {args.model}")
    logger.debug(f"Configuration file: {cfg_path}")

    config = parse_config(args, cfg_path)
    logger.debug("Configuration parsed and updated")

    model_name = args.model.lower()
    model_options = config.get("model_options", {})
    token_level = model_options.get("token_level", "char")
    datasets = config.get("datasets", {})

    logger.info(f"Loading dataset with token_level: {token_level}")
    text = get_dataset(datasets["source"], datasets["locations"])

    logger.info("Building vocabulary and mappings")
    tokens, vocab, vocab_size = build_vocab(text, token_level)
    stoi, itos = create_mappings(vocab)

    logger.info("Encoding data")
    data = encode_data(tokens, stoi)

    logger.info("Creating model")
    vocab_data = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    model_config = {**config["models"], "vocab": vocab_data}
    model = get_model(model_name, model_config, cfg_path)

    logger.info("Validating model")
    validate_model(model, text, data)


def validate_model(model: Model.BaseLM, text: str, data: torch.Tensor) -> None:
    """Validate the type of model to determine the appropriate run method.

    Args:
        model (Model.BaseLM): The model instance.
        text (str): The full dataset text.
        data (torch.Tensor): Encoded dataset tensor.
    """
    logger.debug(f"Validating model type: {model.name}")

    if model.name == "distilgpt2":
        logger.info("Running DistilGPT2 model for text generation")
        generated_text = model.run(text)
        logger.info(generated_text)
    else:
        logger.info(f"Running {model.name} model")
        run_model(model, data)


def run_model(model: Model.BaseLM, data: torch.Tensor) -> None:
    """Run training or text generation for the model.

    Loads from checkpoint if available.
    Randomizes seed character for generation.

    Args:
        model (Model.BaseLM): The model instance.
        data (torch.Tensor): Encoded dataset tensor.

    Raises:
        UnpicklingError: If there is an error loading the model.
    """
    logger.debug(
        f"Running model in {'training' if model.training else 'generation'} mode"
    )

    if os.path.exists(model.ckpt_path):
        logger.info(f"Loading checkpoint from {model.ckpt_path}")
        try:
            model.load_state_dict(torch.load(model.ckpt_path))
            logger.info("Checkpoint loaded successfully")
        except UnpicklingError as e:
            logger.error(f"Error loading model: {e}")
            raise
    else:
        logger.debug("No checkpoint found, starting with a new model")

    if model.training:
        logger.info("Starting model training")
        optimize_and_train(model, data)
    else:
        logger.info("Starting text generation")
        generated_text = model.generate()
        logger.info(generated_text)


if __name__ == "__main__":
    main(parse_args())
