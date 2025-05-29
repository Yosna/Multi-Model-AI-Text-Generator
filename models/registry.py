from models.base_model import BaseLanguageModel
from models.bigram_model import BigramLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel


class ModelRegistry:
    """Registry for mapping model names to their respective classes."""

    BaseLM = BaseLanguageModel
    BigramLM = BigramLanguageModel
    LSTMLM = LSTMLanguageModel
    TransformerLM = TransformerLanguageModel
