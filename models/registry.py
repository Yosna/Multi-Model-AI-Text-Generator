from models.bigram_model import BigramLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel


class ModelRegistry:
    """Registry for mapping model names to their respective classes."""

    BigramLanguageModel = BigramLanguageModel
    LSTMLanguageModel = LSTMLanguageModel
    TransformerLanguageModel = TransformerLanguageModel
