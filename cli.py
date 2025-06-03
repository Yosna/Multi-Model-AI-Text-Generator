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
        bool: True if the string matches a value in TRUE_STRINGS, False otherwise
    """
    return str(arg).lower() in TRUE_STRINGS


def false_string(arg):
    """
    Check if a string represents a false boolean value.

    Args:
        arg: The value to check as a string

    Returns:
        bool: True if the string matches a value in FALSE_STRINGS, False otherwise
    """
    return str(arg).lower() in FALSE_STRINGS


def set_arg_bool(arg: Any) -> Any:
    """
    Convert a string argument to a boolean value if it matches true/false patterns.

    Args:
        arg: The argument to potentially convert to boolean

    Returns:
        Any: Boolean if the argument matches true/false patterns, otherwise the original argument
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

    for key in dir(args):
        arg = getattr(args, key)
        if not key.startswith("_") and arg is not None:
            arg = arg if isinstance(arg, int | float) else set_arg_bool(arg)
            try:
                setting = model_config["runtime"][key]
                matching_types = type(setting) == type(arg)
                model_config["runtime"][key] = arg if matching_types else setting
            except:
                pass
    save_config(config, cfg_path)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model selection and runtime configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing model selection and runtime settings
    """
    parser = argparse.ArgumentParser(description="Run a language model")

    parse_model(parser)
    parse_runtime(parser)

    return parser.parse_args()


def parse_model(parser: argparse.ArgumentParser):
    """
    Add model selection argument to the parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the model argument to
    """
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["bigram", "lstm", "gru", "transformer"],
        metavar="[bigram | lstm | gru | transformer]",
        help="Model name to use from config",
    )


def parse_runtime(parser: argparse.ArgumentParser):
    """
    Add runtime configuration arguments to the parser.

    Parses arguments for:
    - Training mode toggle
    - Number of training steps
    - Evaluation interval
    - Early stopping patience
    - Maximum new tokens for generation
    - Maximum number of checkpoints to keep

    Args:
        parser (argparse.ArgumentParser): The argument parser to add runtime arguments to
    """
    parser.add_argument(
        "--training",
        type=str,
        default=None,
        choices=list(TRUE_STRINGS) + list(FALSE_STRINGS),
        metavar=f"[{', '.join(TRUE_STRINGS)} | {', '.join(FALSE_STRINGS)}]",
        help="Override training in config",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="[int]",
        help="Override steps in config",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        metavar="[int]",
        help="Override interval in config",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        metavar="[int]",
        help="Override patience in config",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="[int]",
        help="Override max_new_tokens in config",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        metavar="[int]",
        help="Override max_checkpoints in config",
    )
