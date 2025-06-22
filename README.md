# Text Generation Language Models in PyTorch

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project implements five text generation language models using PyTorch:

- **Bigram model** — a simple neural network that learns character-to-character transition probabilities
- **LSTM model** — a recurrent neural network capable of learning longer-range character sequences using memory and context
- **GRU model** — a gated recurrent unit network for efficient sequence modeling with fewer parameters than LSTM
- **Transformer model** — a trainable transformer supporting both training and generation
- **DistilGPT2 model** — inference-only; uses a pre-trained Hugging Face transformer for high-quality text generation

The codebase is modular, config-driven, and supports training, checkpointing, early stopping, hyperparameter tuning, and generation from any model via CLI. Comprehensive unit tests are included for all major modules, including training, library, utilities, visualization, tuning, model/CLI behavior, and profiling (**current coverage**: 100%).

## Table of Contents

- [Features](#features)
- [Model Architectures](#model-architectures)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Configuration Editor](#configuration-editor)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Optuna Dashboard](#optuna-dashboard)
- [Profiling](#profiling)
- [Usage](#usage)
- [Loss Visualization](#loss-visualization)
- [GPU Acceleration](#gpu-acceleration)
- [Example Output (LSTM)](#example-output-lstm)
- [Dependencies](#dependencies)
- [Docker Usage](#docker-usage)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- Character-level or word-level tokenization across multiple input files (configurable via `token_level`)
- Dynamic vocabulary and index mapping
- Modular model registry for Bigram, LSTM, GRU, Transformer, and DistilGPT2 (inference-only)
- Configurable training and hyperparameter tuning via `config.json`
- Automatic hyperparameter tuning with Optuna
- Optuna Dashboard for visualizing hyperparameter optimization studies
- Adam optimizer with early stopping
- Automatic checkpoint rotation and resumption with metadata tracking
- Multinomial sampling for randomized generation
- Temperature scaling for controllable randomness in generation (configurable via `temperature`)
- Comprehensive CLI interface with model selection, runtime, hyperparameter, model options, tuning, and visualization configuration
- Full unit test coverage (100%) for all modules
- Tests include generation and training for all models, tuning, visualization, CLI behavior, argument parsing helpers, and profiling
- Loss visualization with matplotlib, including smoothing and saving plots
- GPU-accelerated training by default
- Integrated dataset library with pre-configured datasets
- Support for local files, Hugging Face datasets, and built-in library datasets
- Built-in profiling for performance analysis
- Interactive GUI for editing `config.json`
- **Statistics**: 185 unit tests, 100% coverage, 993 stmts / 0 miss

## Model Architectures

### Bigram Model

A lightweight model that uses an embedding table to predict the next character from the current character only. Fast and simple, but limited in predictive capability.

### LSTM Model

A recurrent neural network using embedding, multi-layer LSTM, and projection back to vocab size. Learns long-range dependencies across sequences for improved generation.

### GRU Model

A gated recurrent unit network that efficiently models sequences with fewer parameters than LSTM, providing a balance between speed and performance.

### Transformer Model

A trainable transformer model using self-attention mechanisms for sophisticated text generation. Supports both training and generation. Architecture includes token and position embeddings, multi-head attention, feedforward layers, and stacking of encoder layers.

### DistilGPT2 Model

Integration with a pre-trained Hugging Face DistilGPT2 model for high-quality text generation. **Inference-only** (cannot be trained or fine-tuned).

## Datasets

Three types of datasets are supported:

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

All behavior is driven by a single `config.json` file. You can edit this file manually or use the user-friendly [Configuration Editor](#configuration-editor).

<details>
<summary><b>Example</b> <code>config.json</code> (<i>click to expand</i>)</summary>

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
        "data_name": "science"
      },
      "huggingface": {
        "data_name": "pubmed_qa",
        "config_name": "pqa_artificial",
        "split": "train",
        "field": "question"
      }
    }
  },

  "model_options": {
    "save_model": true,
    "token_level": "word",
    "temperature": 1.0
  },

  "models": {
    "bigram": {
      "runtime": {
        "training": true,
        "steps": 10000,
        "interval": 100,
        "patience": 10,
        "max_new_tokens": 128,
        "max_checkpoints": 10
      },
      "hparams": {
        "batch_size": 16,
        "block_size": 32,
        "lr": 0.001
      }
    },
    "lstm": {
      "runtime": {
        "training": true,
        "steps": 50000,
        "interval": 500,
        "patience": 10,
        "max_new_tokens": 256,
        "max_checkpoints": 10
      },
      "hparams": {
        "batch_size": 32,
        "block_size": 64,
        "lr": 0.0015,
        "embedding_dim": 64,
        "hidden_size": 128,
        "num_layers": 2
      }
    },
    "gru": {
      "runtime": {
        "training": true,
        "steps": 50000,
        "interval": 500,
        "patience": 10,
        "max_new_tokens": 256,
        "max_checkpoints": 10
      },
      "hparams": {
        "batch_size": 32,
        "block_size": 64,
        "lr": 0.0015,
        "embedding_dim": 64,
        "hidden_size": 128,
        "num_layers": 2
      }
    },
    "transformer": {
      "runtime": {
        "training": true,
        "steps": 100000,
        "interval": 1000,
        "patience": 10,
        "max_new_tokens": 256,
        "max_checkpoints": 10
      },
      "hparams": {
        "batch_size": 32,
        "block_size": 64,
        "lr": 0.001,
        "embedding_dim": 32,
        "max_seq_len": 128,
        "num_heads": 2,
        "ff_dim": 128,
        "num_layers": 2
      }
    },
    "distilgpt2": {
      "runtime": {
        "max_new_tokens": 256
      },
      "hparams": {
        "block_size": 32
      }
    }
  },

  "pruners": {
    "median": {
      "n_startup_trials": 5,
      "n_warmup_steps": 1000
    },
    "halving": {
      "min_resource": 5,
      "reduction_factor": 2,
      "min_early_stopping_rate": 1
    },
    "hyperband": {
      "min_resource": 5,
      "reduction_factor": 2
    }
  },

  "tuning_options": {
    "auto_tuning": true,
    "save_tuning": true,
    "save_study": true,
    "n_trials": 100,
    "pruner": "hyperband",
    "step_divisor": 10
  },

  "tuning_ranges": {
    "batch_size": {
      "type": "int",
      "min": 16,
      "max": 128,
      "step": 16
    },
    "block_size": {
      "type": "int",
      "min": 32,
      "max": 256,
      "step": 32
    },
    "lr": {
      "type": "float",
      "min": 0.0001,
      "max": 0.01,
      "log": true
    },
    "embedding_dim": {
      "type": "int",
      "min": 16,
      "max": 128,
      "step": 16
    },
    "hidden_size": {
      "type": "int",
      "min": 32,
      "max": 256,
      "step": 32
    },
    "num_layers": {
      "type": "categorical",
      "values": [1, 2, 3, 4]
    },
    "max_seq_len": {
      "type": "int",
      "min": 32,
      "max": 256,
      "step": 32
    },
    "num_heads": {
      "type": "int",
      "min": 2,
      "max": 8,
      "step": 2
    },
    "ff_dim": {
      "type": "int",
      "min": 32,
      "max": 256,
      "step": 32
    }
  },

  "visualization": {
    "save_plot": true,
    "show_plot": true,
    "smooth_loss": true,
    "smooth_val_loss": true,
    "weight": 0.9
  }
}
```

</details><br>

You can configure:

- **Datasets** (`datasets`): Source and location to pull from
- **Model Options** (`model_options`): Model saving, tokenization level, and temperature scaling for generation
- **Runtime** (`runtime`): Training and generation settings for each model
- **Hyperparameters** (`hparams`): Model-specific architecture and optimization parameters
- **Pruners** (`pruners`): Configuration for Optuna pruners
- **Tuning Options** (`tuning_options`): Optuna tuning configuration, pruner, and trial settings
- **Tuning Ranges** (`tuning_ranges`): Hyperparameter search spaces for automatic tuning
- **Visualization** (`visualization`): Loss plotting, smoothing, and saving options

## Configuration Editor

An interactive GUI for editing `config.json` is included to simplify configuration management.

- Launch with:
  ```bash
  python -m run config
  ```
- This will open a window where you can view and modify all settings from `config.json`.
- The config editor supports saving changes back to the file.
- [DearPyGui](https://dearpygui.readthedocs.io/en/latest/) was used to build the editor.

## Hyperparameter Tuning

Automatic hyperparameter tuning is supported via [Optuna](https://optuna.org/).

- Enable tuning by setting `"auto_tuning": true` in `config.json`.
- Define search spaces in the `"tuning_ranges"` section (supports `int`, `float`, and `categorical` types).
- Tuning is integrated into the training workflow and can be controlled via the CLI or config.
- Results are saved and can be used to update model hyperparameters automatically if `"save_tuning": true`.
- Tunable fields include: `batch_size`, `block_size`, `lr`, `embedding_dim`, `hidden_size`, `num_layers`, `max_seq_len`, `num_heads`, `ff_dim`.
- Pruners and tuning options are configurable in `config.json`.

## Optuna Dashboard

Visualize your hyperparameter optimization studies interactively with Optuna Dashboard.

- Launch with:
  ```bash
  python -m run dashboard
  ```
- The dashboard will open at **localhost:8080**.
- Requires `optuna-dashboard` (install with `pip install optuna-dashboard`).
- Shows study history, parameter importance, and more.

## Profiling

A built-in profiling tool is included to help you analyze performance bottlenecks in your code.

- The profiler is located at `run/profiler.py` and is accessible via the modular run package.
- It runs the main application under `cProfile` and saves filtered, timestamped reports to `profiles/`.
- Reports are filtered to exclude virtual environment and bootstrap code, and show the top functions by calls, time, and cumulative time.
- You can run the profiler with:
  ```bash
  python -m run profiler
  ```
- You can also import and use the profiler programmatically in your own scripts or tests.
- The profiling tool is unit tested with 100% coverage.

## Usage

### Command Line Interface

The project provides a flexible CLI for controlling model options, runtime and hyperparameters, tuning options, and visualization:

- **Model Options** (`--save-model`, `--token-level`, `--temperature`)
- **Runtime** (`--training`, `--steps`, `--interval`, etc.)
- **Hyperparameters** (`--batch-size`, `--block-size`, `--lr`, etc.)
- **Tuning Options** (`--auto-tuning`, `--save-tuning`, `--save-study`, etc.)
- **Visualization** (`--save-plot`, `--show-plot`, `--smooth-loss`, etc.)

For a full list of arguments, run:

```bash
python main.py --help
```

<details><summary><b>Available CLI Arguments:</b> (<i>click to expand</i>)</summary><br>

\*_arg for all models_, \*\*_arg for all models excluding distilgpt2_

- `--model`: Select model type (**default**: transformer, **options**: [bigram | lstm | gru | transformer | distilgpt2])
- `--training`: Toggle training mode \*\*
- `--steps`: Number of training steps \*\*
- `--interval`: Validation interval during training \*\*
- `--patience`: Early stopping patience \*\*
- `--max-new-tokens`: Maximum tokens to generate \*
- `--max-checkpoints`: Maximum checkpoints to keep \*\*
- `--batch-size`: Override batch size for training \*\*
- `--block-size`: Override context window size \*
- `--lr`: Override learning rate \*\*
- `--embedding-dim`: Override embedding dimension size (LSTM / GRU)
- `--hidden-size`: Override hidden layer size (LSTM / GRU)
- `--num-layers`: Override number of model layers (LSTM / GRU / transformer)
- `--max-seq-len`: Override maximum sequence length (transformer)
- `--num-heads`: Override number of attention heads (transformer)
- `--ff-dim`: Override feedforward dimension (transformer)
- `--save-model`: Override model saving (model_options)
- `--token-level`: Override tokenization level (model_options)
- `--temperature`: Override temperature for generation (model_options)
- `--auto-tuning`: Enable/disable hyperparameter tuning (tuning_options)
- `--save-tuning`: Enable/disable saving tuned hyperparameters (tuning_options)
- `--save-study`: Enable/disable saving Optuna study (tuning_options)
- `--n-trials`: Number of Optuna trials (tuning_options)
- `--pruner`: Pruner type for Optuna (tuning_options)
- `--step-divisor`: Step divisor for tuning (tuning_options)
- `--save-plot`: Enable/disable saving loss plots (visualization)
- `--show-plot`: Enable/disable showing loss plots (visualization)
- `--smooth-loss`: Enable/disable smoothing of loss curves (visualization)
- `--smooth-val-loss`: Enable/disable smoothing of validation loss (visualization)
- `--weight`: Smoothing weight (visualization)
</details>

#### Notes

- The model argument will default to `--model transformer` if omitted.
- Any arguments omitted will default to the respective value defined in `config.json`.
- Boolean flags support flexible input: `true`, `false`, `on`, `off`, `yes`, `no`, `1`, `0`.
- **distilgpt2** uses a pre-trained Hugging Face model and is inference-only (cannot be trained).

### Train a model

```bash
# Train the Transformer model
python main.py --model transformer --training true
```

### Generate text

After training, switch to generation mode by setting `"training": false` inside the appropriate section of `config.json`:

```json
"training": false
```

Then run the same command to generate text:

```bash
python main.py --model transformer
```

The output will begin with a randomly selected seed character or word (depending on `token_level`) and continue for the configured number of tokens.

You can control the randomness of generation using the `temperature` argument in `config.json`. Lower values make output more deterministic; higher values make it more random.

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
- numpy
- transformers
- pytest (for unit testing)
- coverage (for test coverage reporting)
- optuna (for hyperparameter tuning)
- dearpygui (for configuration editor)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Note:**  
The `torch` package in `requirements.txt` is the CPU version of PyTorch.  
If you want to use a GPU, it is recommended to **install the appropriate CUDA-enabled version of PyTorch** for your system **before running** `pip install -r requirements.txt` for a quicker install.  
You can find the correct install command for your system and CUDA version at the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

For example, to install PyTorch with CUDA 11.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Docker Usage

You can run this project in a Docker container for a fully isolated and reproducible environment.

### Build the Docker image

```bash
docker build -t <your-image-name> .
```

### Run the Docker container

```bash
docker run --rm -it <your-image-name>:latest
```

### Notes

- The provided `Dockerfile` uses the `python:3.11-slim` base image for a smaller footprint.
- It explicitly installs the **CPU-only** version of PyTorch to avoid large image sizes. If you need GPU support, modify the Dockerfile and requirements accordingly.
- The `.dockerignore` file is configured to exclude unnecessary files (such as datasets, checkpoints, and virtual environments) from the image. If you add new large files or folders, update `.dockerignore` to keep your image size small.
- If you encounter issues with image size, check that you are not copying large files or using the GPU version of torch by accident.

You can modify the `CMD` in the Dockerfile to run other scripts or pass arguments as needed.

## Testing

- The project includes comprehensive unit tests for all major modules: training, datasets, utility functions, loss visualization, tuning, model/CLI behavior, and profiling.
- Tests are written using `pytest` with `coverage` for reporting. Both are required and included in `requirements.txt`
- All unit tests are located in the `tests/` directory.
- **Statistics**: 189 unit tests, 100% coverage, 964 stmts / 0 miss
- To run all tests:
  ```bash
  pytest
  ```
- To check coverage:
  ```bash
  coverage run -m pytest
  coverage report -m
  ```
- You can also run a specific test file, for example:
  ```bash
  pytest tests/test_utils.py
  ```
- Test output will show which tests passed or failed, and coverage will report which lines are tested.
- Coverage includes data processing, plotting, model logic, CLI argument parsing, tuning, profiling, and more.

## Future Improvements

- Add learning rate scheduling during training
- Add visualization for transformer attention
- Add beam search to model generation

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Yosna/Multi-Model-AI-Text-Generator/blob/main/LICENSE) for details.
