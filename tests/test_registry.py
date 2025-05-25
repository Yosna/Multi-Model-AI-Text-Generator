from models.registry import ModelRegistry
from models.bigram_model import BigramLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel


def test_model_registry_contains_all_models():
    assert ModelRegistry.BigramLanguageModel is BigramLanguageModel
    assert ModelRegistry.LSTMLanguageModel is LSTMLanguageModel
    assert ModelRegistry.TransformerLanguageModel is TransformerLanguageModel
