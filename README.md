# Text Generation Language Models in PyTorch

This project implements three text generation language models using PyTorch:

- **Bigram model** — a simple neural network that learns character-to-character transition probabilities
- **LSTM model** — a recurrent neural network capable of learning longer-range character sequences using memory and context
- **Transformer model** — integration with a pre-built transformer model for high-quality text generation

The codebase is modular, config-driven, and supports training, checkpointing, early stopping, and generation from any model via CLI. Full unit testing is included for all `utils.py` and `visualizer.py` functions.

## Table of Contents

- [Features](#features)
- [Model Architectures](#model-architectures)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Usage](#usage)
- [Loss Visualization](#loss-visualization)
- [GPU Acceleration](#gpu-acceleration)
- [Example Output (LSTM)](#example-output-lstm)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- Character-level tokenization across multiple input files
- Dynamic vocabulary and index mapping
- Modular model registry for Bigram, LSTM, and Transformer
- Configurable training via `config.json` (not currently implemented for Transformer)
- Adam optimizer with early stopping
- Automatic checkpoint rotation and resumption
- Multinomial sampling for randomized generation
- CLI interface to toggle models and behavior
- Full unit test coverage for utility functions
- Loss visualization with matplotlib, including smoothing and saving plots
- GPU-accelerated training by default
- Integrated dataset library with pre-configured datasets
- Support for local files, Hugging Face datasets, and built-in library datasets

## Model Architectures

### Bigram Model

A lightweight model that uses an embedding table to predict the next character from the current character only. Fast and simple, but limited in predictive capability.

### LSTM Model

A recurrent neural network using embedding, multi-layer LSTM, and projection back to vocab size. Learns long-range dependencies across sequences for an improved generation.

### Transformer Model

Integration with a pre-built transformer model that uses self-attention mechanisms for sophisticated text generation. Currently supports text generation with configurable context length.

**Note:** The Transformer model currently supports inference only. It loads a prebuilt model and generates text using self-attention mechanisms, but does not yet support training or fine-tuning.

## Datasets

The project supports three types of datasets:

### Local Dataset

The `dataset/` directory (included in this repo) contains **100 filtered sample texts** for training, preprocessed from [Project Gutenberg](https://www.gutenberg.org). These texts have been cleaned and filtered using length, markup, and English word ratio heuristics.

For more training data, you can download the **full cleaned dataset (4,437 books)** [on Hugging Face](https://huggingface.co/datasets/Yosna/Project-Gutenberg-Training-Data).

### Built-in Library

The project includes a pre-configured library of datasets:

- **News** (0.03 GB) - AG News dataset
- **Science** (0.41 GB) - PubMed QA dataset
- **Movies** (0.49 GB) - IMDB dataset
- **Yelp** (0.51 GB) - Yelp Review Full dataset
- **SQuAD** (0.08 GB) - Stanford Question Answering Dataset
- **Tiny Stories** (1.89 GB) - Tiny Stories dataset
- **Stack Overflow** (5.75 GB) - Stack Overflow Questions dataset
- **Wikipedia** (18.81 GB) - English Wikipedia dataset

### Custom Hugging Face Datasets

You can use any dataset from the Hugging Face Hub by specifying the dataset name and configuration in `config.json`. This allows for flexible experimentation with different text sources.

## Configuration

All behavior is driven by a single `config.json` file:

```json
{
  "datasets": {
    "source": "library",
    "locations": {
      "local": {
        "directory": "dataset",
        "extension": "txt"
      },
      "library": {
        "data_name": "news"
      },
      "huggingface": {
        "data_name": "pubmed_qa",
        "config_name": "pqa_artificial",
        "split": "train",
        "field": "context"
      }
    }
  },
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
    },
    "model": {}
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
  },
  "transformer": {
    "runtime": {
      "block_size": 32,
      "max_new_tokens": 240
    },
    "model": {}
  },
  "visualization": {
    "show_plot": true,
    "smooth_loss": true,
    "smooth_val_loss": true,
    "weight": 0.9,
    "save_data": true
  }
}
```

You can configure:

- Dataset source and location
- Training parameters for each model
- Model architecture parameters
- Loss visualization settings

## Usage

### Train a model

```bash
# Train the Bigram model
python main.py --model bigram

# Train the LSTM model
python main.py --model lstm

# Use the Transformer model
python main.py --model transformer
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

## Loss Visualization

- Plotting both training and validation loss curves
- Optional exponential smoothing for clearer trends
- Saving plots with timestamped filenames to a model-specific directory
- Configurable via `config.json`

## GPU Acceleration

- GPU acceleration is used by default if a CUDA device is available.

## Example Output (LSTM)

```
"Quite pit to well find out of his seed a smarters.

'Ha! you would
a vounty presending out a glanced the busband. The lamb"
```

While not yet semantically coherent, the model demonstrates accurate word shapes, spacing, punctuation, and rudimentary grammar. More training will improve realism and coherence.

## Dependencies

- Python 3.10+
- PyTorch
- matplotlib
- datasets (Hugging Face)

Install dependencies with:

```bash
# To run using your CPU
pip install torch matplotlib datasets

# To run using your GPU with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib datasets
```

## Testing

- The project includes full unit testing for utility and loss visualization functions.
- Tests are written using `pytest`.
- To run all tests, use the following command from the project root:

```bash
pytest
```

- You can also run a specific test file, for example:

```bash
pytest tests/test_utils.py
```

- Test output will show which tests passed or failed
- Coverage includes data processing, plotting, and other core utilities.

## Future Improvements

- Unit tests for model behavior and CLI
- Add temperature scaling for more controllable sampling
- Add automatic hyperparameter tuning
- Implement training for transformer model
- Implement transformer model from scratch

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Yosna/Multi-Model-AI-Text-Generator/blob/main/LICENSE) for details.
