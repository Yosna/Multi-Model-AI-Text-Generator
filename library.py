"""Dataset loading utilities for model training.

Supports local files, Hugging Face datasets, and
a built-in library of pre-configured datasets.

Includes:
- DATASET_LIBRARY: Pre-configured dataset metadata for quick access.
- get_dataset: Unified interface for loading datasets from various sources.
- _load_from_local: Helper for loading and concatenating local text files.
- _load_from_huggingface: Helper for loading text fields from Hugging Face datasets.
"""

import logging
import os

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DATASET_LIBRARY = {
    "news": {  # Dataset size: 0.03 GB
        "data_name": "ag_news",
        "config_name": None,
        "split": "train",
        "field": "text",
    },
    "squad": {  # Dataset size: 0.08 GB
        "data_name": "squad",
        "config_name": None,
        "split": "train",
        "field": "answers",
    },
    "science": {  # Dataset size: 0.41 GB
        "data_name": "pubmed_qa",
        "config_name": "pqa_artificial",
        "split": "train",
        "field": "long_answer",
    },
    "movies": {  # Dataset size: 0.49 GB
        "data_name": "imdb",
        "config_name": None,
        "split": "train",
        "field": "text",
    },
    "yelp": {  # Dataset size: 0.51 GB
        "data_name": "yelp_review_full",
        "config_name": None,
        "split": "train",
        "field": "text",
    },
    "tiny_stories": {  # Dataset size: 1.89 GB
        "data_name": "roneneldan/TinyStories",
        "config_name": None,
        "split": "train",
        "field": "text",
    },
    "stackoverflow": {  # Dataset size: 5.75 GB
        "data_name": "pacovaldez/stackoverflow-questions",
        "config_name": None,
        "split": "train",
        "field": "body",
    },
    "wikipedia": {  # Dataset size: 18.81 GB
        "data_name": "wikimedia/wikipedia",
        "config_name": "20231101.en",
        "split": "train",
        "field": "text",
    },
}


def _load_from_local(directory: str, extension: str) -> str:
    """Load and concatenate all files with a given extension from a directory.

    Both arguments are configurable in `config.json`.

    Args:
        directory (str): Path to the directory containing text files.
        extension (str): File extension to filter (e.g., "txt").

    Returns:
        str: Newline-separated contents of all matching files.
    """
    logger.debug(f"Loading local files from {directory} with extension {extension}")

    files = sorted(os.listdir(directory))
    matching_files = [file for file in files if file.endswith(extension)]
    logger.debug(f"Found {len(matching_files)} files matching extension {extension}")

    text = ""
    total_size = 0
    for i, file in enumerate(matching_files):
        path = os.path.join(directory, file)
        file_size = os.path.getsize(path)
        total_size += file_size

        logger.debug(
            f"Loading file {i+1}/{len(matching_files)}: {file} ({file_size} bytes)"
        )

        with open(path, "r", encoding="utf-8") as f:
            file_text = f.read()
            text += f"{file_text}\n"

    logger.info(
        f"Loaded {len(matching_files)} files, "
        f"total size: {total_size} bytes, "
        f"text length: {len(text)} characters"
    )
    return text


def _load_from_huggingface(
    data_name: str,
    config_name: str | None,
    split: str,
    field: str,
) -> str:
    """Load and extract text from a Hugging Face dataset.

    All arguments are configurable in `config.json`.

    Args:
        data_name (str): Dataset name on the Hugging Face Hub.
        config_name (str | None): Optional configuration name for multi-config datasets.
        split (str): Split to load (e.g., "train").
        field (str): Field name to extract (e.g., "text", "context").

    Returns:
        str: Newline-separated text from the selected field.

    Raises:
        ValueError: If the field is not found in the dataset.
    """
    logger.debug(
        f"Loading HuggingFace dataset: {data_name}, config: {config_name}, "
        f"split: {split}, field: {field}"
    )

    dataset = load_dataset(data_name, config_name, split=split)
    if isinstance(dataset, dict):
        dataset = dataset[split]

    if isinstance(dataset, Dataset):
        dataset_size = len(dataset)
        logger.debug(f"Dataset loaded with {dataset_size} examples")
    else:
        logger.debug("Dataset loaded with unknown number of examples")

    logger.debug(f"Available fields: {dataset.column_names}")

    if dataset.column_names and field not in dataset.column_names:
        raise ValueError(
            f"Field {field} not found in dataset. Fields: {dataset.column_names}"
        )

    lines = [data[field] for data in dataset]  # type: ignore
    text = "\n".join(lines)

    logger.info(
        f"Extracted text from {len(lines)} examples, "
        f"total length: {len(text)} characters"
    )
    return text


def get_dataset(source: str, locations: dict[str, dict[str, str]]) -> str:
    """Load and return a text dataset from the specified source.

    Args:
        source (str): The source of the dataset. Must be one of:
            - "local": Load from local text files in a directory.
            - "library": Load from a pre-configured dataset in the built-in library.
            - "huggingface": Load from a custom Hugging Face dataset.
        locations (dict): A dictionary with configuration for each possible source.
            - For "local": expects a dict with "directory" and "extension" keys.
            - For "library": expects a dict with "data_name" key.
            - For "huggingface": expects a dict with Hugging Face dataset parameters.

    Returns:
        str: The concatenated text data from the selected dataset source.

    Raises:
        ValueError: If the specified library dataset is not found in DATASET_LIBRARY.
        ValueError: If the dataset source is unknown.
    """
    logger.info(f"Loading dataset from source: {source}")

    if source == "local":
        local = locations["local"]
        logger.debug(f"Local dataset config: {local}")
        return _load_from_local(local["directory"], local["extension"])
    elif source == "library":
        lib = locations["library"]
        logger.debug(f"Library dataset config: {lib}")
        if lib["data_name"] not in DATASET_LIBRARY:
            raise ValueError(f"No {lib['data_name']} dataset found in library")
        return _load_from_huggingface(**DATASET_LIBRARY[lib["data_name"]])
    elif source == "huggingface":
        logger.debug(f"HuggingFace dataset config: {locations['huggingface']}")
        return _load_from_huggingface(**locations["huggingface"])
    else:
        raise ValueError(f"Unknown dataset source: {source}")
