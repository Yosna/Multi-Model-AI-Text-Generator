"""Command-line interface for settings configuration.

Allows configuring, training, generating, and managing language models.
Handles argument parsing, runtime overrides, and user interaction for all
supported models and features.
"""

import argparse
from typing import Any
from utils.io_utils import load_config, save_config

# Constants for boolean string values
TRUE_STRINGS = ["true", "on", "yes", "1"]
FALSE_STRINGS = ["false", "off", "no", "0"]
# Constant for the boolean argument metavars
BOOL_METAVAR = f"[{', '.join(TRUE_STRINGS)} | {', '.join(FALSE_STRINGS)}]"


def true_string(arg: str) -> bool:
    """Check if a string represents a true boolean value.

    Args:
        arg: The value to check as a string

    Returns:
        bool: True if matching a value in TRUE_STRINGS, False otherwise
    """
    return str(arg).lower() in TRUE_STRINGS


def false_string(arg: str) -> bool:
    """Check if a string represents a false boolean value.

    Args:
        arg: The value to check as a string

    Returns:
        bool: True if matching a value in FALSE_STRINGS, False otherwise
    """
    return str(arg).lower() in FALSE_STRINGS


def set_arg_bool(arg: Any) -> Any:
    """Convert a string to a boolean value if it matches true/false patterns.

    Args:
        arg: The argument to potentially convert to boolean

    Returns:
        Any: Boolean if matching true/false patterns, otherwise the original value
    """
    if true_string(arg):
        arg = True
    elif false_string(arg):
        arg = False
    return arg


def parse_config(args: argparse.Namespace, cfg_path: str) -> None:
    """Parse command line arguments and update the model configuration file.

    - Loads the existing configuration
    - Updates the model's runtime settings based on provided arguments
    - Preserves type consistency with existing settings
    - Saves the updated configuration back to file

    Args:
        args (argparse.Namespace): Parsed command line arguments
        cfg_path (str): Path to the configuration file
    """
    config = load_config(cfg_path)
    models = config.get("models", {})
    model_options = config.get("model_options", {})
    model_config = models.get(args.model, {})
    runtime = model_config.get("runtime", {})
    hparams = model_config.get("hparams", {})
    tuning_options = config.get("tuning_options", {})
    visualization = config.get("visualization", {})

    config_sections = {
        "model_options": model_options,
        "runtime": runtime,
        "hparams": hparams,
        "tuning_options": tuning_options,
        "visualization": visualization,
    }

    for setting in config_sections.values():
        for key in setting:
            if hasattr(args, key):
                arg = getattr(args, key)
                arg = arg if isinstance(arg, int | float) else set_arg_bool(arg)
                matching_type = isinstance(arg, type(setting[key]))
                if arg is not None and matching_type:
                    setting[key] = arg
    save_config(config, cfg_path)


def add_arg(
    parser: argparse.ArgumentParser, name: str, type: type, metavar: str, help: str
) -> None:
    """Add a command-line argument to the parser with consistent defaults.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the argument to
        name (str): The name of the argument (e.g. "--batch-size")
        type (type): The type of the argument (e.g. int, float, str)
        metavar (str): The metavar string to display in help messages
        help (str): The help message describing the argument
    """
    parser.add_argument(name, type=type, default=None, metavar=metavar, help=help)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model selection and runtime configuration.

    Returns:
        argparse.Namespace: Parsed arguments for model selection and runtime settings
    """
    parser = argparse.ArgumentParser(description="Run a language model")

    parse_model(parser)
    parse_model_options(parser)
    parse_runtime(parser)
    parse_hparams(parser)
    parse_tuning_options(parser)
    parse_visualization(parser)

    return parser.parse_args()


def parse_model(parser: argparse.ArgumentParser) -> None:
    """Add model selection argument to the parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser for the model argument
    """
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["bigram", "lstm", "gru", "transformer", "distilgpt2"],
        metavar="[bigram | lstm | gru | transformer | distilgpt2]",
        help="Model name to use from config (section: models)",
    )


def parse_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model option configuration arguments to the parser.

    Parses arguments for:
    - Model checkpoint saving toggle (save_model)
    - Tokenization level (token_level)
    - Temperature (temperature)

    Args:
        parser (argparse.ArgumentParser): The argument parser for model options
    """
    save_model_help = "Override save_model in config (section: model_options)"
    token_level_help = "Override token_level in config (section: model_options)"
    temperature_help = "Override temperature in config (section: model_options)"

    add_arg(parser, "--save-model", str, BOOL_METAVAR, save_model_help)
    add_arg(parser, "--token-level", str, "[char | word]", token_level_help)
    add_arg(parser, "--temperature", float, "[float]", temperature_help)


def parse_runtime(parser: argparse.ArgumentParser) -> None:
    """Add runtime configuration arguments to the parser.

    Parses arguments for:
    - Training mode toggle (training)
    - Number of training steps (steps)
    - Evaluation interval (interval)
    - Early stopping patience (patience)
    - Maximum new tokens for generation (max_new_tokens)
    - Maximum number of checkpoints to keep (max_checkpoints)

    Args:
        parser (argparse.ArgumentParser): The argument parser for runtime values
    """
    training_help = "Override training in config (section: runtime)"
    steps_help = "Override steps in config (section: runtime)"
    interval_help = "Override interval in config (section: runtime)"
    patience_help = "Override patience in config (section: runtime)"
    max_new_tokens_help = "Override max_new_tokens in config (section: runtime)"
    max_checkpoints_help = "Override max_checkpoints in config (section: runtime)"

    add_arg(parser, "--training", str, BOOL_METAVAR, training_help)
    add_arg(parser, "--steps", int, "[int]", steps_help)
    add_arg(parser, "--interval", int, "[int]", interval_help)
    add_arg(parser, "--patience", int, "[int]", patience_help)
    add_arg(parser, "--max-new-tokens", int, "[int]", max_new_tokens_help)
    add_arg(parser, "--max-checkpoints", int, "[int]", max_checkpoints_help)


def parse_hparams(parser: argparse.ArgumentParser) -> None:
    """Add hyperparameter configuration arguments to the parser.

    Parses arguments for:
    - Batch size for training (batch_size)
    - Context window size (block_size)
    - Learning rate (lr)
    - Embedding dimension size (embedding_dim)
    - Hidden layer size (hidden_size)
    - Number of model layers (num_layers)
    - Maximum sequence length (max_seq_len)
    - Number of attention heads (num_heads)
    - Feedforward dimension (ff_dim)

    Args:
        parser (argparse.ArgumentParser): The argument parser for hyperparameters
    """
    batch_size_help = "Override batch_size in config (section: hparams)"
    block_size_help = "Override block_size in config (section: hparams)"
    lr_help = "Override lr in config (section: hparams)"
    embedding_dim_help = "Override embedding_dim in config (section: hparams)"
    hidden_size_help = "Override hidden_size in config (section: hparams)"
    num_layers_help = "Override num_layers in config (section: hparams)"
    max_seq_len_help = "Override max_seq_len in config (section: hparams)"
    num_heads_help = "Override num_heads in config (section: hparams)"
    ff_dim_help = "Override ff_dim in config (section: hparams)"

    add_arg(parser, "--batch-size", int, "[int]", batch_size_help)
    add_arg(parser, "--block-size", int, "[int]", block_size_help)
    add_arg(parser, "--lr", float, "[float]", lr_help)
    add_arg(parser, "--embedding-dim", int, "[int]", embedding_dim_help)
    add_arg(parser, "--hidden-size", int, "[int]", hidden_size_help)
    add_arg(parser, "--num-layers", int, "[int]", num_layers_help)
    add_arg(parser, "--max-seq-len", int, "[int]", max_seq_len_help)
    add_arg(parser, "--num-heads", int, "[int]", num_heads_help)
    add_arg(parser, "--ff-dim", int, "[int]", ff_dim_help)


def parse_tuning_options(parser: argparse.ArgumentParser) -> None:
    """Add tuning option configuration arguments to the parser.

    Parses arguments for:
    - Hyperparameter optimization toggle (auto_tuning)
    - Tuned hyperparameters save toggle (save_tuning)
    - Optuna study save toggle (save_study)
    - Number of trials (n_trials)
    - Type of pruner to use for studies (pruner)
    - Step divisor for trials; divides steps in runtime (step_divisor)

    Args:
        parser (argparse.ArgumentParser): The argument parser for tuning options
    """
    auto_tuning_help = "Override auto_tuning in config (section: tuning_options)"
    save_tuning_help = "Override save_tuning in config (section: tuning_options)"
    save_study_help = "Override save_study in config (section: tuning_options)"
    n_trials_help = "Override n_trials in config (section: tuning_options)"
    pruner_help = "Override pruner in config (section: tuning_options)"
    step_divisor_help = "Override step_divisor in config (section: tuning_options)"

    add_arg(parser, "--auto-tuning", str, BOOL_METAVAR, auto_tuning_help)
    add_arg(parser, "--save-tuning", str, BOOL_METAVAR, save_tuning_help)
    add_arg(parser, "--save-study", str, BOOL_METAVAR, save_study_help)
    add_arg(parser, "--n-trials", int, "[int]", n_trials_help)
    add_arg(parser, "--pruner", str, "[median | halving | hyperband]", pruner_help)
    add_arg(parser, "--step-divisor", int, "[int]", step_divisor_help)


def parse_visualization(parser: argparse.ArgumentParser) -> None:
    """Add visualization configuration arguments to the parser.

    Parses arguments for:
    - Plot saving toggle (save_plot)
    - Plot showing toggle (show_plot)
    - Loss curve smoothing toggle (smooth_loss)
    - Validation loss curve smoothing toggle (smooth_val_loss)
    - Weight for smoothing aggressiveness (weight)

    Args:
        parser (argparse.ArgumentParser): The argument parser for visualization options
    """
    save_plot_help = "Override save_plot in config (section: visualization)"
    show_plot_help = "Override show_plot in config (section: visualization)"
    smooth_loss_help = "Override smooth_loss in config (section: visualization)"
    smooth_val_loss_help = "Override smooth_val_loss in config (section: visualization)"
    weight_help = "Override weight in config (section: visualization)"

    add_arg(parser, "--save-plot", str, BOOL_METAVAR, save_plot_help)
    add_arg(parser, "--show-plot", str, BOOL_METAVAR, show_plot_help)
    add_arg(parser, "--smooth-loss", str, BOOL_METAVAR, smooth_loss_help)
    add_arg(parser, "--smooth-val-loss", str, BOOL_METAVAR, smooth_val_loss_help)
    add_arg(parser, "--weight", float, "[float]", weight_help)
