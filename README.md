# Character-Level Language Models in PyTorch

This project implements two character-level language models using PyTorch:

* **Bigram model** — a simple neural network that learns character-to-character transition probabilities
* **LSTM model** — a recurrent neural network capable of learning longer-range character sequences using memory and context

The codebase is modular, config-driven, and supports training, checkpointing, early stopping, and generation from either model via CLI. A full suite of unit tests is included for all `utils.py` functions.

## Features

* Character-level tokenization across multiple input files
* Dynamic vocabulary and index mapping
* Modular model registry for Bigram and LSTM
* Configurable training via `config.json`
* Adam optimizer with early stopping
* Automatic checkpoint rotation and resumption
* Multinomial sampling for randomized generation
* CLI interface to toggle models and behavior
* Full unit test coverage for utility functions

## Model Architectures

### Bigram Model

A lightweight model that uses an embedding table to predict the next character from the current character only. Fast and simple, but limited in predictive capability.

### LSTM Model

A recurrent neural network using embedding, multi-layer LSTM, and projection back to vocab size. Learns long-range dependencies across sequences for an improved generation.

## Input Format

The `dataset/` directory (included in this repo) contains **100 filtered sample texts** for training, preprocessed from [Project Gutenberg](https://www.gutenberg.org).

For more training data, you can download the **full cleaned dataset (4,437 books)** [on Hugging Face](https://huggingface.co/datasets/Yosna/Project-Gutenberg-Training-Data).

All files were filtered using length, markup, and English word ratio heuristics. Each `.txt` file contains a single book written in English.

To train on your own data, replace or add `.txt` files to the `dataset/` folder. Every `.txt` file in the folder will be loaded and concatenated for training.

Each file should contain English plain text, such as books, articles, or other character-rich sources.

## Configuration

All behavior is driven by a single `config.json` file:

```json
{
  "bigram": {
    "runtime": {
      "training": true,
      "batch_size": 8,
      "block_size": 4,
      "steps": 10000,
      "interval": 100,
      "lr": 0.001,
      "patience": 10,
      "max_new_tokens": 100,
      "max_checkpoints": 10
    }
  },
  "lstm": {
    "runtime": {
      "training": true,
      "batch_size": 16,
      "block_size": 64,
      "steps": 50000,
      "interval": 500,
      "lr": 0.0015,
      "patience": 10,
      "max_new_tokens": 200,
      "max_checkpoints": 10
    },
    "model": {
      "embedding_dim": 64,
      "hidden_size": 128,
      "num_layers": 2
    }
  }
}
```

You can configure training, model size, learning rate, and checkpointing for each model independently.

## Usage

### Train a model

```bash
# Train the Bigram model
python main.py --model bigram

# Train the LSTM model
python main.py --model lstm
```

### Generate text

After training, switch to generation mode by setting `"training": false` inside the appropriate section of `config.json`:

```json
"training": false
```

Then run the same command to generate text:

```bash
python main.py --model lstm
```

The output will begin with a randomly selected seed character and continue for the configured number of tokens.

## Example Output (LSTM)

```
“Quite pit to well find out of his seed a smarters.

‘Ha! you would
a vounty presending out a glanced the busband. The lamb”
```

While not yet semantically accurate, the model shows proper word shapes, spacing, punctuation, and primitive grammar. More training will improve realism and coherence.

## Dependencies

* Python 3.10+
* PyTorch

Install dependencies with:

```bash
pip install torch
```

## Future Improvements

* Unit tests for model behavior and CLI
* Add temperature scaling for more controllable sampling
* Optionally add Transformer-based model for comparison
* Loss visualization with matplotlib or TensorBoard

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Yosna/AI-Character-Level-Language-Models/blob/main/LICENSE) for details.
