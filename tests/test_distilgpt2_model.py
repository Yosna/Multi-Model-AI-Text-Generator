from models.registry import ModelRegistry as Model


def get_distilgpt2_config():
    return {
        "runtime": {
            "block_size": 4,
            "max_new_tokens": 10,
        },
        "model": {},
    }


def get_distilgpt2_model():
    return Model.DistilGPT2LM(
        config=get_distilgpt2_config(), cfg_path="test_config.json"
    )


def test_distilgpt2_model():
    model = get_distilgpt2_model()
    assert model is not None


def test_distilgpt2_model_init():
    model = get_distilgpt2_model()
    assert model.name == "distilgpt2"
    assert model.tokenizer is not None
    assert model.model is not None


def test_distilgpt2_model_tokenizer():
    model = get_distilgpt2_model()
    assert model.tokenizer.name_or_path == "distilgpt2"
    assert model.tokenizer.model_max_length == 1024
    assert model.tokenizer.vocab_size == 50257


def test_distilgpt2_model_model():
    model = get_distilgpt2_model()
    assert model.model.name_or_path == "distilgpt2"
    assert model.model.config.architectures == ["GPT2LMHeadModel"]
    assert model.model.config.model_type == "gpt2"
    assert model.model.config.vocab_size == 50257


def test_distilgpt2_model_run():
    model = get_distilgpt2_model()
    text = "Hello!"
    result = model.run(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result[: model.block_size] in text
