"""Training / experiment-tracking public surface."""

from ampops.training.bakeoff import (
    load_config_from_run,
    parse_param_overrides,
    train_candidate,
)
from ampops.training.evaluate import evaluate_holdout
from ampops.training.pipeline import run_experiment_pipeline
from ampops.training.registry import register_champion, select_champion

__all__ = [
    "train_candidate",
    "load_config_from_run",
    "parse_param_overrides",
    "evaluate_holdout",
    "run_experiment_pipeline",
    "select_champion",
    "register_champion",
]
