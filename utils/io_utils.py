"""Configuration and file I/O utility functions.

Includes:
- Loading and saving configuration files
- Metadata retrieval
- JSON formatting helpers
"""

import json
import os
import re
from typing import Any, TypeVar

T = TypeVar("T")


def get_metadata(path: str, key: str, default: T) -> T:
    """Retrieve a value from metadata.json; returns a default if not found.

    Args:
        path (str): Path to metadata.json.
        key (str): Key to retrieve from the metadata.
        default (T): Default value if key is not found.

    Returns:
        T: Either the value from metadata or the default.
    """
    data = default
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            data = metadata.get(key, default)
    return data


def save_config(config: dict[str, Any], cfg_path: str) -> None:
    """Save a configuration dictionary to a JSON file.

    Formats arrays to remove line breaks and adds line breaks between
    top-level sections that end with '},'.

    Args:
        config (dict[str, Any]): The configuration dictionary to save
        cfg_path (str): Path where the configuration file should be saved
    """
    cfg_str = json.dumps(config, indent=2)

    def fix_arrays(match):
        """Extract only the numbers from the array content.

        Args:
            match (re.Match): The match from the regex search.

        Returns:
            str: The formatted string with only the numbers from the array content.
        """
        numbers = re.findall(r"\d+", match.group(1))
        return "[" + ", ".join(numbers) + "]"

    # Collapse arrays to a single line
    cfg_str = re.sub(r"\[(.*?)\]", fix_arrays, cfg_str, flags=re.DOTALL)

    cfg_lines = cfg_str.split("\n")
    cfg_format = []

    for i, line in enumerate(cfg_lines):
        cfg_format.append(line)
        # Add a line break after top-level sections that end with '},'
        if line == "  }," and i < len(cfg_lines) - 1:
            cfg_format.append("")

    cfg_str = "\n".join(cfg_format) + "\n"

    with open(cfg_path, "w") as f:
        f.write(cfg_str)


def load_config(path: str) -> dict[str, Any]:
    """Load the entire config.json as a dict.

    Args:
        path (str): Path to config.json.

    Returns:
        dict[str, Any]: The loaded configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config(path: str, config_name: str) -> dict[str, Any]:
    """Load and return the configuration dictionary for the given model.

    Args:
        path (str): Path to config.json.
        config_name (str): Name of the model config to retrieve.

    Returns:
        dict[str, Any]: The configuration dictionary for the model.

    Raises:
        KeyError: If no config is found for the given config name.
        ValueError: If no value is found for the given config name.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)[config_name]
    except KeyError:
        raise KeyError(f"No config found for: {config_name}")
    if config is None:
        raise ValueError(f"No value found for config: {config_name}")
    return config
