from models.registry import ModelRegistry as Model


def get_transformer_config():
    return {
        "runtime": {
            "block_size": 4,
            "max_new_tokens": 10,
        },
        "model": {},
    }


def get_transformer_model():
    return Model.TransformerLM(config=get_transformer_config(), cfg_path="config.json")


def test_transformer_model():
    model = get_transformer_model()
    assert model is not None


def test_transformer_model_init():
    model = get_transformer_model()
    assert model.name == "transformer"
    assert model.tokenizer is not None
    assert model.model is not None


def test_transformer_model_tokenizer():
    model = get_transformer_model()
    assert model.tokenizer.name_or_path == "distilgpt2"
    assert model.tokenizer.model_max_length == 1024
    assert model.tokenizer.vocab_size == 50257


def test_transformer_model_model():
    model = get_transformer_model()
    assert model.model.name_or_path == "distilgpt2"
    assert model.model.config.architectures == ["GPT2LMHeadModel"]
    assert model.model.config.model_type == "gpt2"
    assert model.model.config.vocab_size == 50257


def test_transformer_model_run():
    model = get_transformer_model()
    text = "Hello!"
    result = model.run(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result[: model.block_size] in text
