from models.registry import ModelRegistry as Model
import torch
import torch.nn as nn


def get_lstm_config():
    return {
        "runtime": {
            "training": True,
            "steps": 1,
            "interval": 1,
            "patience": 10,
            "max_new_tokens": 10,
            "max_checkpoints": 1,
        },
        "hparams": {
            "batch_size": 2,
            "block_size": 4,
            "lr": 0.0015,
            "embedding_dim": 4,
            "hidden_size": 8,
            "num_layers": 1,
        },
    }


def get_lstm_model():
    return Model.LSTMLM(
        config=get_lstm_config(),
        cfg_path="test_config.json",
        vocab_size=5,
    )


def test_lstm_model():
    model = get_lstm_model()
    assert model is not None


def test_lstm_model_no_vocab():
    model_not_initialized = False
    try:
        Model.LSTMLM(
            config=get_lstm_config(), cfg_path="test_config.json", vocab_size=0
        )
    except ValueError:
        model_not_initialized = True
    assert model_not_initialized


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
        f"LSTMLanguageModel(\n"
        f"\tvocab_size={model.vocab_size},\n"
        f"\tembedding_dim={model.embedding_dim},\n"
        f"\thidden_size={model.hidden_size},\n"
        f"\tnum_layers={model.num_layers}\n"
        f")"
    ).expandtabs(4)


def test_lstm_model_forward():
    model = get_lstm_model()
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    logits = model(idx)
    assert logits.shape == torch.Size([1, 5, 5])
    assert isinstance(model.hidden, tuple)
    assert len(model.hidden) == 2
    assert model.hidden[0].shape == torch.Size([1, 1, 8])
    assert model.hidden[1].shape == torch.Size([1, 1, 8])


def test_lstm_model_generate():
    model = get_lstm_model()
    model.device = torch.device("cpu")
    start_idx = 1
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    generated = model.generate(start_idx, itos)
    assert isinstance(generated, str)
    assert generated[0] == "H"
    assert len(generated) == model.max_new_tokens + 1
