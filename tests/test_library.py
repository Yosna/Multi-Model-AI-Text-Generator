from library import _load_from_local, _load_from_huggingface, get_dataset
from unittest.mock import patch
import pytest


class MockSplit:
    def __init__(self, data):
        self.data = data
        self.column_names = ["text"]

    def __iter__(self):
        return iter(self.data)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def build_dir(tmp_path):
    build_file(tmp_path, "file1.txt", "Hello,")
    build_file(tmp_path, "file2.txt", " World!")
    build_file(tmp_path, "file3.pdf", "PDF_CONTENT")
    build_file(tmp_path, "file4.docx", "DOCX_CONTENT")


def test_load_from_local(tmp_path):
    build_dir(tmp_path)
    text = _load_from_local(tmp_path, "txt")
    expected = "Hello,\n World!\n"
    assert all(word in text for word in ["Hello", "World"])
    assert text == expected
    assert len(text) == len(expected)


def test_local_extension_filtering(tmp_path):
    build_dir(tmp_path)
    text = _load_from_local(tmp_path, "txt")
    expected = "Hello,\n World!\n"
    assert expected in text
    assert "PDF_CONTENT" not in text
    assert "DOCX_CONTENT" not in text


def test_load_from_huggingface():
    text = _load_from_huggingface("Yosna/test-dataset", None, "train", "text")
    assert text == "Hello, World!"


def test_load_from_huggingface_dict():
    dataset = {"train": MockSplit([{"text": "Hello, World!"}])}
    with patch("library.load_dataset", return_value=dataset):
        text = _load_from_huggingface("Yosna/test-dataset", None, "train", "text")
        assert text == "Hello, World!"


def test_get_dataset_library():
    with patch("library._load_from_huggingface", return_value="Hello, World!"):
        text = get_dataset("library", {"library": {"data_name": "news"}})
    assert text == "Hello, World!"


def test_get_dataset_library_invalid_source():
    datasets = {
        "source": "library",
        "locations": {
            "library": {
                "data_name": "invalid",
            }
        },
    }
    with pytest.raises(ValueError):
        get_dataset(datasets["source"], datasets["locations"])


def test_get_dataset_huggingface():
    datasets = {
        "source": "huggingface",
        "locations": {
            "huggingface": {
                "data_name": "Yosna/test-dataset",
                "config_name": None,
                "split": "train",
                "field": "text",
            }
        },
    }
    text = get_dataset(datasets["source"], datasets["locations"])
    assert text == "Hello, World!"


def test_get_dataset_invalid_source():
    datasets = {
        "source": "invalid",
        "locations": {
            "huggingface": {
                "data_name": "Yosna/test-dataset",
                "config_name": None,
                "split": "train",
                "field": "text",
            }
        },
    }
    with pytest.raises(ValueError):
        get_dataset(datasets["source"], datasets["locations"])


def test_get_dataset_invalid_field():
    datasets = {
        "source": "huggingface",
        "locations": {
            "huggingface": {
                "data_name": "Yosna/test-dataset",
                "config_name": None,
                "split": "train",
                "field": "invalid",
            }
        },
    }
    with pytest.raises(ValueError):
        get_dataset(datasets["source"], datasets["locations"])
