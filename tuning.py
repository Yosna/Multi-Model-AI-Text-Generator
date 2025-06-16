"""Hyperparameter tuning utilities using Optuna for language model training.

This module provides automated hyperparameter optimization using Optuna.
It supports multiple pruning strategies (Median, Successive Halving, and Hyperband)
and persists optimization studies to SQLite for later analysis.

Key Features:
- Automated hyperparameter search with configurable number of trials
- Multiple pruning strategies to optimize search efficiency
- Study persistence for long-running optimizations
- Integration with Optuna Dashboard for visualization
- Configurable tuning ranges and optimization parameters

Example:
    To enable hyperparameter tuning, set in config.json:
    {
        "tuning_options": {
            "auto_tuning": true,
            "save_study": true,
            "pruner": "hyperband"
        }
    }
"""

import optuna
import torch
from optuna import pruners

from models.registry import ModelRegistry as Model
from training import train
from utils.io_utils import get_config, load_config, save_config
from utils.model_utils import get_model


def optimize_and_train(model: Model.BaseLM, data: torch.Tensor):
    """Run hyperparameter optimization (if enabled) and train the model.

    This function manages the complete optimization and training workflow:
    1. Loads configuration and checks if auto-tuning is enabled
    2. Creates and runs an Optuna study if tuning is enabled
    3. Updates model hyperparameters with best found values
    4. Runs final training with optimized parameters

    The optimization process can be configured through config.json:
    - Number of trials
    - Pruning strategy
    - Study persistence
    - Step divisor for faster evaluation

    Args:
        model (Model.BaseLM): The model instance being trained
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.

    Returns:
        tuple: (trained model, validation losses) from the final training run
    """
    config = load_config(model.cfg_path)
    model_config = config.get("models", {}).get(model.name, {})
    hparams = model_config.get("hparams", {})
    tuning_options = config.get("tuning_options", {})

    if tuning_options.get("auto_tuning", False):
        save_study = tuning_options.get("save_study", False)
        n_trials = tuning_options.get("n_trials", 50)

        objective = make_objective(model, data)
        study = optuna.create_study(
            study_name=f"{model.name}_loss_tuning",
            direction="minimize",
            storage="sqlite:///optuna.db" if save_study else None,
            load_if_exists=save_study,
            pruner=create_pruner(model),
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params

        if tuning_options.get("save_tuning", False):
            hparams.update(best_params)
            save_config(config, model.cfg_path)

    return train(model, data)


def make_objective(model: Model.BaseLM, data: torch.Tensor):
    """Create an Optuna objective function for hyperparameter search.

    The returned function suggests values for each hyperparameter in the tuning
    ranges, runs a training step, and returns the final validation loss.

    The objective function handles three types of hyperparameters:
    - Integer parameters (e.g., batch_size)
    - Float parameters (e.g., learning_rate)
    - Categorical parameters (e.g., optimizer type)

    Args:
        model (Model.BaseLM): The model instance being trained
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.

    Returns:
        Callable[[optuna.Trial], float]: Objective function for Optuna study.
            The function returns the final validation loss for the trial.

    Raises:
        ValueError: If model.vocab_size is not set.
    """
    if not model.vocab_size:
        raise ValueError("Vocab size is not set for the current model")

    config = load_config(model.cfg_path)
    models = config.get("models", {})
    model_config = models.get(model.name, {})
    tuning_options = config.get("tuning_options", {})
    tune = config.get("tuning_ranges", {})
    model = get_model(
        model.name, models, model.cfg_path, model.vocab_size, model.token_level
    )
    hparams = model_config.get("hparams", {})
    step_divisor = tuning_options.get("step_divisor", 10)

    def objective(trial: optuna.Trial):
        for key in hparams.keys():
            if tune[key] and tune[key]["type"] == "int":
                hparams[key] = trial.suggest_int(
                    key, tune[key]["min"], tune[key]["max"], step=tune[key]["step"]
                )
            elif tune[key] and tune[key]["type"] == "float":
                hparams[key] = trial.suggest_float(
                    key, tune[key]["min"], tune[key]["max"], log=tune[key]["log"]
                )
            elif tune[key] and tune[key]["type"] == "categorical":
                hparams[key] = trial.suggest_categorical(key, tune[key]["values"])

        _, val_losses = train(model, data, trial=trial, step_divisor=step_divisor)

        return val_losses[-1]

    return objective


def create_pruner(model: Model.BaseLM):
    """Create an Optuna pruner based on configuration settings.

    This function supports three pruning strategies:
    - MedianPruner: Prunes based on the median of previous trials
    - SuccessiveHalvingPruner: Prunes based on resource allocation
    - HyperbandPruner: Prunes based on bracket-based resource allocation

    The pruner configuration is read from config.json under the "pruners" section.
    Each pruner type has its own configuration parameters.

    Args:
        model (Model.BaseLM): The model instance being trained

    Returns:
        Optional[optuna.pruners.BasePruner]: Configured pruner instance or None if
            an unknown pruner type is specified.
    """
    config = load_config(model.cfg_path)
    tuning_options = config.get("tuning_options", {})
    pruner_name = tuning_options.get("pruner", "median")

    if pruner_name == "median":
        median = get_config(model.cfg_path, "pruners").get("median", {})
        pruner = pruners.MedianPruner(
            n_startup_trials=median.get("n_startup_trials", 5),
            n_warmup_steps=median.get("n_warmup_steps", 1000),
        )
    elif pruner_name == "halving":
        halving = get_config(model.cfg_path, "pruners").get("halving", {})
        pruner = pruners.SuccessiveHalvingPruner(
            min_resource=halving.get("min_resource", 1),
            reduction_factor=halving.get("reduction_factor", 2),
            min_early_stopping_rate=halving.get("min_early_stopping_rate", 0),
        )
    elif pruner_name == "hyperband":
        hyperband = get_config(model.cfg_path, "pruners").get("hyperband", {})
        pruner = pruners.HyperbandPruner(
            min_resource=hyperband.get("min_resource", 1),
            max_resource=model.steps // tuning_options.get("step_divisor", 10),
            reduction_factor=hyperband.get("reduction_factor", 2),
        )
    else:
        print(
            f"""
            Unknown pruner: {pruner_name}
            Please use one of the following pruners:
            - median
            - halving
            - hyperband
            The pruner can be set from your config:
            config.json -> tuning_options -> pruner
            No pruner will be used for this study.
            """
        )
        pruner = None

    return pruner
