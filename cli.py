import argparse
from typing import Any
from utils import load_config, save_config

# Constants for boolean string values
TRUE_STRINGS = ["true", "on", "yes", "1"]
FALSE_STRINGS = ["false", "off", "no", "0"]


def true_string(arg):
    """
    Check if a string represents a true boolean value.

    Args:
        arg: The value to check as a string

    Returns:
        bool: True if matching a value in TRUE_STRINGS, False otherwise
    """
    return str(arg).lower() in TRUE_STRINGS


def false_string(arg):
    """
    Check if a string represents a false boolean value.

    Args:
        arg: The value to check as a string

    Returns:
        bool: True if matching a value in FALSE_STRINGS, False otherwise
    """
    return str(arg).lower() in FALSE_STRINGS


def set_arg_bool(arg: Any) -> Any:
    """
    Convert a string to a boolean value if it matches true/false patterns.

    Args:
        arg: The argument to potentially convert to boolean

    Returns:
        Any: Boolean if matching true/false patterns, otherwise no change
    """
    if true_string(arg):
        return True
    elif false_string(arg):
        return False
    return arg


def parse_config(args: argparse.Namespace, cfg_path: str):
    """
    Parse command line arguments and update the model configuration file.

    - Loads the existing configuration
    - Updates the model's runtime settings based on provided arguments
    - Preserves type consistency with existing settings
    - Saves the updated configuration back to file

    Args:
        args (argparse.Namespace): Parsed command line arguments
        cfg_path (str): Path to the configuration file
    """
    config = load_config(cfg_path)
    model_config = config["models"][args.model]

    config_sections = {
        "runtime": model_config["runtime"],
        "hparams": model_config["hparams"],
    }

    for setting in config_sections.values():
        for key in setting:
            if hasattr(args, key):
                arg = getattr(args, key)
                arg = arg if isinstance(arg, int | float) else set_arg_bool(arg)
                matching_type = type(setting[key]) == type(arg)
                if arg is not None and matching_type:
                    setting[key] = arg
    save_config(config, cfg_path)


def add_arg(
    parser: argparse.ArgumentParser, name: str, type: type, metavar: str, help: str
) -> None:
    """
    Add a command-line argument to the parser with consistent defaults.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the argument to
        name (str): The name of the argument (e.g. "--batch-size")
        type (type): The type of the argument (e.g. int, float, str)
        metavar (str): The metavar string to display in help messages
        help (str): The help message describing the argument
    """
    parser.add_argument(name, type=type, default=None, metavar=metavar, help=help)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model selection and runtime configuration.

    Returns:
        argparse.Namespace: Parsed arguments for model selection and runtime settings
    """
    parser = argparse.ArgumentParser(description="Run a language model")

    parse_model(parser)
    parse_runtime(parser)
    parse_hparams(parser)

    return parser.parse_args()


def parse_model(parser: argparse.ArgumentParser) -> None:
    """
    Add model selection argument to the parser.

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


def parse_runtime(parser: argparse.ArgumentParser) -> None:
    """
    Add runtime configuration arguments to the parser.

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
    training_metavar = f"[{', '.join(TRUE_STRINGS)} | {', '.join(FALSE_STRINGS)}]"
    training_help = "Override training in config (section: runtime)"
    steps_help = "Override steps in config (section: runtime)"
    interval_help = "Override interval in config (section: runtime)"
    patience_help = "Override patience in config (section: runtime)"
    max_new_tokens_help = "Override max_new_tokens in config (section: runtime)"
    max_checkpoints_help = "Override max_checkpoints in config (section: runtime)"

    add_arg(parser, "--training", str, training_metavar, training_help)
    add_arg(parser, "--steps", int, "[int]", steps_help)
    add_arg(parser, "--interval", int, "[int]", interval_help)
    add_arg(parser, "--patience", int, "[int]", patience_help)
    add_arg(parser, "--max-new-tokens", int, "[int]", max_new_tokens_help)
    add_arg(parser, "--max-checkpoints", int, "[int]", max_checkpoints_help)


def parse_hparams(parser: argparse.ArgumentParser) -> None:
    """
    Add hyperparameter configuration arguments to the parser.

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
