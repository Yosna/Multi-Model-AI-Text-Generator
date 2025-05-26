import torch
import torch.nn as nn
from models.bigram_model import BigramLanguageModel


def test_bigram_model():
    model = BigramLanguageModel(vocab_size=10)
    assert model is not None


def test_bigram_model_init():
    model = BigramLanguageModel(vocab_size=10)
    assert model.name == "bigram"
    assert model.vocab_size == 10


def test_bigram_model_embedding_layer():
    model = BigramLanguageModel(vocab_size=10)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 10


def test_bigram_model_repr():
    model = BigramLanguageModel(vocab_size=10)
    assert str(model) == (
        f"BigramLanguageModel(\n" f"\tvocab_size={model.vocab_size}\n)"
    ).expandtabs(4)


def test_bigram_model_forward():
    model = BigramLanguageModel(vocab_size=10)
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    targets = torch.tensor([[2, 3, 4, 5, 6]])
    logits, loss = model(idx, targets)
    assert logits.shape == torch.Size([5, 10])
    assert loss > 0


def test_bigram_model_generate():
    model = BigramLanguageModel(vocab_size=5)
    model.device = torch.device("cpu")
    start_idx = 1
    itos = {0: "!", 1: "H", 2: "e", 3: "l", 4: "o"}
    max_new_tokens = 5
    generated = model.generate(start_idx, itos, max_new_tokens)
    assert type(generated) == str
    assert generated[0] == "H"
    assert len(generated) == max_new_tokens + 1
