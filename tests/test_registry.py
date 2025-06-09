from models.registry import ModelRegistry as Model
from models.base_model import BaseLanguageModel
from models.bigram_model import BigramLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.gru_model import GRULanguageModel
from models.transformer_model import TransformerLanguageModel
from models.distilgpt2_model import DistilGPT2LanguageModel


def test_model_registry_contains_all_models():
    assert Model.BaseLM is BaseLanguageModel
    assert Model.BigramLM is BigramLanguageModel
    assert Model.LSTMLM is LSTMLanguageModel
    assert Model.GRULM is GRULanguageModel
    assert Model.TransformerLM is TransformerLanguageModel
    assert Model.DistilGPT2LM is DistilGPT2LanguageModel
