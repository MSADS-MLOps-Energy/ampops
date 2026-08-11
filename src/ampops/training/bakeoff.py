"""Model bake-off: train each candidate, log it to MLflow, return its metrics.

This satisfies the course's "AutoML" item as an explicit, tracked comparison of
algorithms (linear -> random forest -> gradient boosting) rather than a full
AutoML framework, which the rubric lists only as an example.

**Model selection never touches the test set.** Candidates are scored on a
validation tail carved out of the *training* data; `data/processed/test.parquet`
stays sealed until holdout evaluation / the deployment workstream.

This module is the seam owned by the modelling workstream — add candidates by
appending to `config.MODEL_CONFIGS`; the DAG picks them up automatically via
dynamic task mapping, no DAG edit required.

Each run logs:
  - eval_name (e.g. validation_tail)
  - hyperparameters (MLflow params + `hyperparams.json` artifact for re-runs)
  - metrics (mape, rmse, mae)
  - timing (duration_seconds, fit_duration_seconds, eval_started_at / eval_ended_at)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
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
EVAL_NAME_VALIDATION = "validation_tail"
HYPERPARAMS_ARTIFACT = "hyperparams.json"

# Limit BLAS/OpenMP threads before XGBoost/sklearn can fight over them.
# Prevents intermittent SIGSEGV on macOS when multiple candidates share a process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Coerce estimator params into JSON / MLflow-safe scalars."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


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


def hyperparams_payload(
    model_name: str,
    params: dict[str, Any],
    model: Any | None = None,
    *,
    eval_name: str = EVAL_NAME_VALIDATION,
    n_features: int | None = None,
) -> dict[str, Any]:
    """Structured hyperparameter record — logged as params + artifact for re-runs."""
    explicit = {k: _jsonable(v) for k, v in params.items()}
    estimator_params: dict[str, Any] = {}
    if model is not None and hasattr(model, "get_params"):
        estimator_params = {k: _jsonable(v) for k, v in model.get_params().items()}

    return {
        "model_name": model_name,
        "eval_name": eval_name,
        "params": explicit,
        "estimator_params": estimator_params,
        "validation_months": VALIDATION_MONTHS,
        "horizon_hours": config.HORIZON_HOURS,
        "random_seed": config.RANDOM_SEED,
        "n_features": n_features,
        "primary_metric": config.PRIMARY_METRIC,
    }


def log_hyperparameters(payload: dict[str, Any]) -> None:
    """Write hyperparameters to the active MLflow run (UI params + JSON artifact).

    Explicit bake-off knobs are logged twice:
      - plain keys (`n_estimators`, ...) for the Databricks/MLflow params table
      - `hp.*` prefixed copies so filters can isolate hyperparameters from
        bookkeeping fields like `eval_name`
    """
    explicit = payload.get("params") or {}
    if explicit:
        mlflow.log_params({k: v for k, v in explicit.items() if v is not None})
        mlflow.log_params({f"hp.{k}": v for k, v in explicit.items() if v is not None})

    mlflow.log_dict(payload, HYPERPARAMS_ARTIFACT)
    # Compact tag so the run list shows the search space at a glance.
    mlflow.set_tag("hyperparams_json", json.dumps(explicit, sort_keys=True, default=str))


def load_config_from_run(
    run_id: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Reload `{model_name, params}` from a prior run for exact re-training.

    Prefers the `hyperparams.json` artifact; falls back to `hp.*` / known param
    keys on the run when the artifact is missing (older runs).
    """
    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        local = client.download_artifacts(run_id, HYPERPARAMS_ARTIFACT)
        with open(local, encoding="utf-8") as f:
            payload = json.load(f)
        model_name = payload["model_name"]
        params = dict(payload.get("params") or {})
        logger.info(
            "Loaded hyperparams from run %s artifact (%s, %s keys)",
            run_id,
            model_name,
            len(params),
        )
        return {"model_name": model_name, "params": params, "source_run_id": run_id}
    except Exception as artifact_err:  # noqa: BLE001
        logger.warning(
            "Could not read %s from run %s (%s); falling back to run params",
            HYPERPARAMS_ARTIFACT,
            run_id,
            artifact_err,
        )

    run = client.get_run(run_id)
    raw = dict(run.data.params)
    model_name = raw.get("model_name")
    if not model_name:
        raise ValueError(f"Run {run_id} has no model_name param; cannot re-run")

    params: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("hp."):
            params[key[3:]] = _parse_param_value(value)
    if not params:
        # Older runs logged bare keys without the hp. prefix.
        skip = {
            "model_name",
            "eval_name",
            "split",
            "n_features",
            "validation_months",
            "horizon_hours",
            "eval_started_at",
            "eval_ended_at",
            "parent_run_id",
        }
        params = {k: _parse_param_value(v) for k, v in raw.items() if k not in skip}

    logger.info(
        "Loaded hyperparams from run %s params (%s, %s keys)",
        run_id,
        model_name,
        len(params),
    )
    return {"model_name": model_name, "params": params, "source_run_id": run_id}


def _parse_param_value(value: str) -> Any:
    """Best-effort cast of MLflow string params back to Python scalars."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_param_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI `key=value` overrides into a params dict."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = _parse_param_value(raw.strip())
    return overrides


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
    # Float-cast avoids MLflow integer-missing-value schema warnings on infer.
    example = input_example.astype("float64")
    # MLflow 3 prefers `name=`; fall back for the Airflow image's 2.16 client.
    try:
        kwargs = {"name": "model", "input_example": example}
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, **kwargs)
        else:
            mlflow.sklearn.log_model(model, **kwargs)
    except TypeError:
        kwargs = {"artifact_path": "model", "input_example": example}
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, **kwargs)
        else:
            mlflow.sklearn.log_model(model, **kwargs)


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
    eval_name: str = EVAL_NAME_VALIDATION,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """Fit one candidate, log everything to MLflow, return its scorecard.

    Returns a JSON-serializable dict (it travels through Airflow XCom):
    `{model_name, run_id, eval_name, params, mape, rmse, mae, ...}`.
    """
    params = dict(params or {})
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or config.MLFLOW_EXPERIMENT)

    df = pd.read_parquet(train_path)
    fit_df, valid_df = time_split(df, test_months=VALIDATION_MONTHS)
    features = feature_columns(df)

    x_fit, y_fit = fit_df[features], fit_df[config.TARGET]
    x_valid, y_valid = valid_df[features], valid_df[config.TARGET]

    run_name = f"{model_name}__{eval_name}"
    with mlflow.start_run(run_name=run_name) as run:
        started_at = _utc_now_iso()
        wall_t0 = time.perf_counter()

        model = build_estimator(model_name, params)
        fit_t0 = time.perf_counter()
        model.fit(x_fit, y_fit)
        fit_duration = time.perf_counter() - fit_t0

        metrics = evaluate(y_valid, model.predict(x_valid))
        duration = time.perf_counter() - wall_t0
        ended_at = _utc_now_iso()

        tags = {
            "eval_name": eval_name,
            "split": "validation",
            "model_name": model_name,
            "pipeline": "ampops-bakeoff",
        }
        if parent_run_id:
            tags["parent_run_id"] = parent_run_id
            tags["rerun_of"] = parent_run_id
        mlflow.set_tags(tags)

        mlflow.log_param("eval_name", eval_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("split", "validation")
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("validation_months", VALIDATION_MONTHS)
        mlflow.log_param("horizon_hours", config.HORIZON_HOURS)
        mlflow.log_param("eval_started_at", started_at)
        mlflow.log_param("eval_ended_at", ended_at)
        if parent_run_id:
            mlflow.log_param("parent_run_id", parent_run_id)

        payload = hyperparams_payload(
            model_name,
            params,
            model,
            eval_name=eval_name,
            n_features=len(features),
        )
        log_hyperparameters(payload)

        mlflow.log_metrics(
            {
                **metrics,
                "duration_seconds": duration,
                "fit_duration_seconds": fit_duration,
            }
        )
        log_model(model, model_name, x_valid.head(5))

        result = {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "eval_name": eval_name,
            "params": params,
            "n_train": len(fit_df),
            "n_valid": len(valid_df),
            "duration_seconds": duration,
            "fit_duration_seconds": fit_duration,
            "eval_started_at": started_at,
            "eval_ended_at": ended_at,
            **metrics,
        }

    logger.info(
        "%s | %s | params=%s | MAPE %.4f | RMSE %.1f MW | %.1fs | run %s",
        model_name,
        eval_name,
        params,
        metrics["mape"],
        metrics["rmse"],
        duration,
        result["run_id"],
    )
    return result
