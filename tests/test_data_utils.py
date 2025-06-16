import torch

from utils.data_utils import decode_data, encode_data, split_data


def test_encode_data():
    stoi = {"!": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    data = encode_data(list("Hello!"), stoi)
    assert data.tolist() == [1, 2, 3, 3, 4, 0]
    assert data.dtype == torch.long


def test_decode_data_char():
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    data = torch.tensor([1, 2, 3, 3, 4, 0])
    decoded = decode_data(data, itos, "char")
    assert decoded == "Hello!"


def test_decode_data_word():
    itos = {0: "Hello,", 1: "World!"}
    data = torch.tensor([0, 1])
    decoded = decode_data(data, itos, "word")
    assert decoded == "Hello, World!"


def test_split_data():
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_data, val_data = split_data(data)
    assert train_data.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert val_data.tolist() == [10]
