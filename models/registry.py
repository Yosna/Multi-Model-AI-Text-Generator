from models.bigram_model import BigramLanguageModel
from models.lstm_model import LSTMLanguageModel


class ModelRegistry:
    BigramLanguageModel = BigramLanguageModel
    LSTMLanguageModel = LSTMLanguageModel
