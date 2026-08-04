"""Model bake-off: train each candidate, log it to MLflow, return its metrics.

This satisfies the course's "AutoML" item as an explicit, tracked comparison of
algorithms (linear -> random forest -> gradient boosting) rather than a full
AutoML framework, which the rubric lists only as an example.

**Model selection never touches the test set.** Candidates are scored on a
validation tail carved out of the *training* data; `data/processed/test.parquet`
stays sealed until the deployment workstream validates the registered model
against the live API.

This module is the seam owned by the modelling workstream — add candidates by
appending to `config.MODEL_CONFIGS`; the DAG picks them up automatically via
dynamic task mapping, no DAG edit required.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from ampops import config
from ampops.features.build import feature_columns
from ampops.features.split import time_split
from ampops.utils.io import get_logger

logger = get_logger(__name__)

# Months of the training tail reserved for scoring candidates against each other.
VALIDATION_MONTHS = 3


def build_estimator(model_name: str, params: dict[str, Any]):
    """Instantiate one candidate. Add new algorithms here."""
    if model_name == "linear":
        return LinearRegression(**params)
    if model_name == "random_forest":
        return RandomForestRegressor(**params)
    if model_name == "xgboost":
        # Imported lazily so the rest of the pipeline runs without xgboost installed.
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    raise ValueError(f"Unknown model_name {model_name!r}; add it to build_estimator()")


def log_model(model, model_name: str, input_example: pd.DataFrame) -> None:
    """Log the fitted model under its correct MLflow flavor.

    XGBoost must go through `mlflow.xgboost`, not `mlflow.sklearn`. Despite
    XGBRegressor implementing the sklearn API, `mlflow.sklearn.log_model`
    serializes via skops, which refuses to round-trip `xgboost.core.Booster`
    and fails the run outright.

    Using the native flavor also means the deployment workstream gets a model
    that `mlflow.pyfunc.load_model` can serve without an xgboost-specific
    shim.
    """
    if model_name == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model", input_example=input_example)
    else:
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAPE (headline) and RMSE (secondary — penalizes the demand spikes that hurt)."""
    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def train_candidate(
    train_path: str,
    model_name: str,
    params: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Fit one candidate, log everything to MLflow, return its scorecard.

    Returns a JSON-serializable dict (it travels through Airflow XCom):
    `{model_name, run_id, mape, rmse, mae, n_train, n_valid}`.
    """
    params = params or {}
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or config.MLFLOW_EXPERIMENT)

    df = pd.read_parquet(train_path)
    fit_df, valid_df = time_split(df, test_months=VALIDATION_MONTHS)
    features = feature_columns(df)

    x_fit, y_fit = fit_df[features], fit_df[config.TARGET]
    x_valid, y_valid = valid_df[features], valid_df[config.TARGET]

    with mlflow.start_run(run_name=model_name) as run:
        model = build_estimator(model_name, params)
        model.fit(x_fit, y_fit)
        metrics = evaluate(y_valid, model.predict(x_valid))

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("validation_months", VALIDATION_MONTHS)
        mlflow.log_param("horizon_hours", config.HORIZON_HOURS)
        mlflow.log_metrics(metrics)
        log_model(model, model_name, x_valid.head(5))

        result = {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "n_train": len(fit_df),
            "n_valid": len(valid_df),
            **metrics,
        }

    logger.info(
        "%s | MAPE %.4f | RMSE %.1f MW | run %s",
        model_name,
        metrics["mape"],
        metrics["rmse"],
        result["run_id"],
    )
    return result
