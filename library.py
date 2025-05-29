from datasets import load_dataset
import os

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
    """
    Load and concatenate all files with a given extension from a directory.
    Both arguments are configurable in `config.json`.

    Args:
        directory (str): Path to the directory containing text files.
        extension (str): File extension to filter (e.g., "txt").

    Returns:
        str: Newline-separated contents of all matching files.
    """
    text = ""
    for file in sorted(os.listdir(directory)):
        if file.endswith(extension):
            path = os.path.join(directory, file)
            with open(path, "r", encoding="utf-8") as f:
                text += f"{f.read()}\n"
    return text


def _load_from_huggingface(
    data_name: str,
    config_name: str | None,
    split: str,
    field: str,
) -> str:
    """
    Load and extract text from a Hugging Face dataset.
    All arguments are configurable in `config.json`.

    Args:
        data_name (str): Dataset name on the Hugging Face Hub.
        config_name (str | None): Optional configuration name for multi-config datasets.
        split (str): Split to load (e.g., "train").
        field (str): Field name to extract (e.g., "text", "context").

    Returns:
        str: Newline-separated text from the selected field.
    """
    dataset = load_dataset(data_name, config_name, split=split)
    if isinstance(dataset, dict):
        dataset = dataset[split]

    if dataset.column_names and field not in dataset.column_names:
        raise ValueError(
            f"Field {field} not found in dataset. Fields: {dataset.column_names}"
        )

    return "\n".join([data[field] for data in dataset])  # type: ignore


def get_dataset(source: str, locations: dict[str, dict[str, str]]) -> str:
    """
    Load and return a text dataset from the specified source.

    Args:
        source (str): The source of the dataset. Must be one of:
            - "local": Load from local text files in a directory.
            - "library": Load from a pre-configured dataset in the built-in library.
            - "huggingface": Load from a custom Hugging Face dataset.
        locations (dict): A dictionary containing configuration for each possible source.
            - For "local": expects a dict with "directory" and "extension" keys.
            - For "library": expects a dict with "data_name" key.
            - For "huggingface": expects a dict with Hugging Face dataset parameters.

    Returns:
        str: The concatenated text data from the selected dataset source.

    Raises:
        ValueError: If the specified library dataset is not found in DATASET_LIBRARY.
        ValueError: If the dataset source is unknown.
    """
    if source == "local":
        local = locations["local"]
        return _load_from_local(local["directory"], local["extension"])
    elif source == "library":
        lib = locations["library"]
        if lib["data_name"] not in DATASET_LIBRARY:
            raise ValueError(f"No {lib['data_name']} dataset found in library")
        return _load_from_huggingface(**DATASET_LIBRARY[lib["data_name"]])
    elif source == "huggingface":
        return _load_from_huggingface(**locations["huggingface"])
    else:
        raise ValueError(f"Unknown dataset source: {source}")
