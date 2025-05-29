from library import _load_from_local, _load_from_huggingface, get_dataset


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def build_dir(tmp_path):
    build_file(tmp_path, "file1.txt", "Hello,")
    build_file(tmp_path, "file2.txt", " World!")
    build_file(tmp_path, "file3.pdf", "PDF_CONTENT")
    build_file(tmp_path, "file4.docx", "DOCX_CONTENT")


def test_load_from_local(tmp_path, extension="txt"):
    build_dir(tmp_path)
    text = _load_from_local(tmp_path, extension)
    expected = "Hello,\n World!\n"
    assert all(word in text for word in ["Hello", "World"])
    assert text == expected
    assert len(text) == len(expected)


def test_local_extension_filtering(tmp_path, extension="txt"):
    build_dir(tmp_path)
    text = _load_from_local(tmp_path, extension)
    expected = "Hello,\n World!\n"
    assert expected in text
    assert "PDF_CONTENT" not in text
    assert "DOCX_CONTENT" not in text


def test_load_from_huggingface(
    name="ag_news", config=None, split="train", field="text"
):
    text = _load_from_huggingface(name, config, split, field)
    assert text is not None
    assert len(text) > 0


def test_get_dataset():
    datasets = {
        "source": "huggingface",
        "locations": {
            "huggingface": {
                "data_name": "ag_news",
                "config_name": None,
                "split": "train",
                "field": "text",
            }
        },
    }
    text = get_dataset(datasets["source"], datasets["locations"])
    assert text is not None
    assert len(text) > 0
