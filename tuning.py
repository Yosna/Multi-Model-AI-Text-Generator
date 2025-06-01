"""
Hyperparameter tuning utilities using Optuna for language model training.

Includes:
- optimize_and_train: Main entry point for auto-tuning and training.
- make_objective: Creates an Optuna objective function for hyperparameter search.
"""

from models.registry import ModelRegistry as Model
import torch
import optuna
from optuna.pruners import MedianPruner
import json
from training import train
from utils import load_config, get_config, get_model


def optimize_and_train(model: Model.BaseLM, data: torch.Tensor, n_trials: int = 50):
    """
    Run hyperparameter optimization (if enabled) and train the model.

    Runs Optuna hyperparameter search using the objective from make_objective.
    Updates model hyperparameters and config if save_tuning is enabled.
    Always runs final training after tuning. Optional and enabled by default.

    Args:
        model (Model.BaseLM): The model instance
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.
    """
    config = load_config(model.cfg_path)
    hparams = config["models"][model.name]["hparams"]

    if config.get("auto_tuning", False):
        objective = make_objective(model, data)
        study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=model.interval * 2),
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params

        if config.get("save_tuning", False):
            hparams.update(best_params)
            with open(model.cfg_path, "w") as f:
                json.dump(config, f, indent=2)

    return train(model, data)


def make_objective(model: Model.BaseLM, data: torch.Tensor):
    """
    Create an Optuna objective function for hyperparameter search.

    The returned function suggests values for each hyperparameter in the tuning
    ranges, runs a training step, and returns the final validation loss.

    Args:
        model (Model.BaseLM): The model instance
        data (torch.Tensor): Full dataset as a 1D tensor of encoded characters.

    Returns:
        Callable[[optuna.Trial], float]: Objective function for Optuna study.

    Raises:
        ValueError: If model.vocab_size is None.
    """
    if model.vocab_size is None:
        raise ValueError("Vocab size is not set for the current model")

    models = get_config(model.cfg_path, "models")
    tune = get_config(model.cfg_path, "tuning_ranges")
    model = get_model(Model, model.name, models, model.cfg_path, model.vocab_size)
    hparams = models[model.name].get("hparams", {})

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

        _, val_losses = train(model, data, trial=trial, step_divisor=10)

        return val_losses[-1]

    return objective
