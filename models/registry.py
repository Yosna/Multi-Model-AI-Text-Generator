"""Model registry for dynamic selection and instantiation of language models."""

from models.base_model import BaseLanguageModel
from models.bigram_model import BigramLanguageModel
from models.distilgpt2_model import DistilGPT2LanguageModel
from models.gru_model import GRULanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel


class ModelRegistry:
    """Registry for mapping model names to their respective classes."""

    BaseLM = BaseLanguageModel
    BigramLM = BigramLanguageModel
    LSTMLM = LSTMLanguageModel
    GRULM = GRULanguageModel
    TransformerLM = TransformerLanguageModel
    DistilGPT2LM = DistilGPT2LanguageModel
