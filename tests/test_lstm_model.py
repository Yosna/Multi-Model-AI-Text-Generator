import torch
import torch.nn as nn
from models.lstm_model import LSTMLanguageModel


def get_lstm_model():
    return LSTMLanguageModel(
        vocab_size=5,
        embedding_dim=4,
        hidden_size=8,
        num_layers=1,
    )


def test_lstm_model():
    model = get_lstm_model()
    assert model is not None


def test_lstm_model_init():
    model = get_lstm_model()
    assert model.name == "lstm"
    assert model.vocab_size == 5
    assert model.embedding_dim == 4
    assert model.hidden_size == 8
    assert model.num_layers == 1


def test_lstm_model_embedding_layer():
    model = get_lstm_model()
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 5
    assert model.embedding.embedding_dim == 4


def test_lstm_model_lstm_layer():
    model = get_lstm_model()
    assert isinstance(model.lstm, nn.LSTM)
    assert model.lstm.input_size == 4
    assert model.lstm.hidden_size == 8
    assert model.lstm.num_layers == 1
    assert model.lstm.batch_first


def test_lstm_model_fc_layer():
    model = get_lstm_model()
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 8
    assert model.fc.out_features == 5
    assert model.fc.bias is not None


def test_lstm_model_repr():
    model = get_lstm_model()
    assert str(model) == (
        "LSTMLanguageModel("
        "vocab_size=5, "
        "embedding_dim=4, "
        "hidden_size=8, "
        "num_layers=1)"
    )


def test_lstm_model_forward():
    model = get_lstm_model()
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    targets = torch.tensor([1, 2, 3, 4, 0])
    logits, loss, hidden = model(idx, targets)
    assert logits.shape == torch.Size([5, 5])
    assert loss > 0
    assert type(hidden) == tuple
    assert len(hidden) == 2
    assert hidden[0].shape == torch.Size([1, 1, 8])
    assert hidden[1].shape == torch.Size([1, 1, 8])


def test_lstm_model_generate():
    model = get_lstm_model()
    model.device = torch.device("cpu")
    start_idx = 1
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    max_new_tokens = 5
    generated = model.generate(start_idx, itos, max_new_tokens)
    assert type(generated) == str
    assert generated[0] == "H"
    assert len(generated) == max_new_tokens + 1
