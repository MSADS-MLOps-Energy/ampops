"""Sealed holdout evaluation — scored once, after champion selection.

The bake-off only ever sees the validation tail. This module loads the
champion's logged model and scores `data/processed/test.parquet`, logging a
dedicated MLflow run with eval_name, metrics, and wall-clock timing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import pandas as pd

from ampops import config
from ampops.features.build import feature_columns
from ampops.training.bakeoff import evaluate
from ampops.utils.io import get_logger

logger = get_logger(__name__)

EVAL_NAME_TEST = "test_holdout"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_holdout(
    champion: dict[str, Any],
    test_path: str | None = None,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    eval_name: str = EVAL_NAME_TEST,
) -> dict[str, Any]:
    """Score the champion on the sealed test set and log a tracked MLflow run.

    Links back to the bake-off via `parent_run_id` / `champion_run_id` tags so
    the Databricks experiment UI stays navigable.
    """
    test_path = test_path or str(config.TEST_PARQUET)
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or config.MLFLOW_EXPERIMENT)

    run_id = champion["run_id"]
    model_name = champion["model_name"]
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    df = pd.read_parquet(test_path)
    features = feature_columns(df)
    x_test, y_test = df[features], df[config.TARGET]

    run_name = f"{model_name}__{eval_name}"
    with mlflow.start_run(run_name=run_name) as run:
        started_at = _utc_now_iso()
        t0 = time.perf_counter()
        preds = model.predict(x_test)
        metrics = evaluate(y_test, preds)
        duration = time.perf_counter() - t0
        ended_at = _utc_now_iso()

        mlflow.set_tags(
            {
                "eval_name": eval_name,
                "split": "test",
                "model_name": model_name,
                "champion_run_id": run_id,
                "pipeline": "ampops-holdout",
            }
        )
        mlflow.log_param("eval_name", eval_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("split", "test")
        mlflow.log_param("champion_run_id", run_id)
        mlflow.log_param("n_test", len(df))
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("horizon_hours", config.HORIZON_HOURS)
        mlflow.log_param("eval_started_at", started_at)
        mlflow.log_param("eval_ended_at", ended_at)
        mlflow.log_metrics({**metrics, "duration_seconds": duration})

        result = {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "champion_run_id": run_id,
            "eval_name": eval_name,
            "n_test": len(df),
            "duration_seconds": duration,
            "eval_started_at": started_at,
            "eval_ended_at": ended_at,
            **metrics,
        }

    logger.info(
        "Holdout %s | %s | MAPE %.4f | RMSE %.1f MW | %.1fs | run %s",
        model_name,
        eval_name,
        metrics["mape"],
        metrics["rmse"],
        duration,
        result["run_id"],
    )
    return result
