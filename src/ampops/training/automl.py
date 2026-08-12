"""H2O AutoML training: one automated search over algorithms and hyperparameters.

Replaces the manual bake-off (`ampops.training.bakeoff.train_candidate`, now
removed) with `H2OAutoML`, which trains and ranks a pool of models itself.
`ampops.training.registry.select_champion` is gone for the same reason — H2O
already hands back a single leader, so there is nothing left to pick between.

`run_h2o_automl` only ever sees train/validate data — model selection never
touches the sealed test holdout. `evaluate_on_test` is a separate, standalone
function that scores an *already-registered* model against that holdout; the
DAG calls it after `registry.register_champion`, not as part of this search.

**No k-fold cross-validation.** `data/processed/train.parquet` is time-ordered
(lag/rolling features reference the past, per `ampops.features.build`), so
shuffling rows across folds — H2O AutoML's default — would let future rows
leak into a fold's training data. We pass `nfolds=0` and an explicit,
chronologically-carved `validation_frame` instead (`time_split`, the same
`VALIDATION_MONTHS` convention the old bake-off used). Accepted trade-off:
H2O skips Stacked Ensembles whenever `nfolds=0`, because those require
cross-validated base-model predictions to train the metalearner. That's
intentional here, not an oversight.

**H2O cluster lifecycle and the Airflow retry case.** `DEFAULT_ARGS` in
`dags/ampops_training_pipeline.py` sets `retries: 1`. Airflow's LocalExecutor
runs tasks as subprocesses of the scheduler process, so a hard-killed first
attempt (OOM, SIGKILL, task timeout) can leave its child H2O Java process
orphaned and still bound to a port after the Python process that spawned it is
gone. `h2o.init()`'s documented default behavior is to *try connecting to
whatever is already listening on the target port first*, and only start a new
server if that fails ("Attempt to connect to a local server, or if not
successful start a new server and connect to it"). Left alone, that means an
unlucky retry would silently adopt whatever the killed attempt left behind —
not necessarily a clean cluster — rather than deliberately choosing to reuse
or replace it. To make that a deliberate non-event, every call to
`run_h2o_automl` grabs a fresh OS-assigned free TCP port immediately before
`h2o.init()`. Since nothing is listening on a port we just verified is free,
`h2o.init()` always takes the "start a new server" branch, regardless of what
a previous attempt left behind — no accidental reconnect is possible. We still
shut our own cluster down in a `finally` block on both the success and
exception paths. A hard kill (SIGKILL) skips `finally` entirely, so an orphan
JVM from *that* scenario is a known residual risk this function cannot fix by
itself; picking a fresh port on every attempt at least guarantees it can never
corrupt or block a subsequent attempt.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import h2o
import mlflow
import mlflow.h2o
import pandas as pd
from h2o.automl import H2OAutoML

from ampops import config
from ampops.features.build import feature_columns
from ampops.features.split import time_split
from ampops.training.bakeoff import VALIDATION_MONTHS, evaluate
from ampops.utils.io import get_logger

logger = get_logger(__name__)

# MLflow tags / params so runs stay comparable across local and Databricks.
# Selection still never uses the sealed test holdout.
EVAL_NAME_VALIDATION = "validation_tail"
EVAL_NAME_TEST = "test_holdout"

_HOMEBREW_JAVA_CANDIDATES = (
    Path("/opt/homebrew/opt/openjdk@17"),
    Path("/usr/local/opt/openjdk@17"),
    Path("/opt/homebrew/opt/openjdk@11"),
    Path("/usr/local/opt/openjdk@11"),
)


def _ensure_java_home() -> None:
    """Point JAVA_HOME at a real JDK when macOS only has the /usr/bin/java stub.

    Homebrew OpenJDK is keg-only, so H2O otherwise finds the stub and fails with
    `Command ['/usr/bin/java', '-version'] returned non-zero exit status 1`.
    """
    existing = os.environ.get("JAVA_HOME")
    if existing and (Path(existing) / "bin" / "java").is_file():
        return

    for home in _HOMEBREW_JAVA_CANDIDATES:
        java = home / "bin" / "java"
        if java.is_file():
            os.environ["JAVA_HOME"] = str(home)
            os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            logger.info("Using JAVA_HOME=%s for H2O", home)
            return


def _configure_tracking(tracking_uri: str | None = None) -> None:
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    if config.MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(config.MLFLOW_REGISTRY_URI)


def _free_port() -> int:
    """Return a currently-free localhost TCP port.

    Used so every `run_h2o_automl` call — including an Airflow retry after a
    killed first attempt — starts its own fresh H2O cluster instead of
    potentially reconnecting to a stale one left behind on H2O's default port.
    See the module docstring.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# H2O uses its own names; expose sklearn / XGBoost-style aliases in MLflow so
# runs are readable next to the old bake-off experiments.
_H2O_PARAM_ALIASES: dict[str, str] = {
    "ntrees": "n_estimators",
    "n_estimators": "n_estimators",
    "learn_rate": "learning_rate",
    "learning_rate": "learning_rate",
    "sample_rate": "subsample",
    "subsample": "subsample",
    "col_sample_rate": "colsample_bytree",
    "col_sample_rate_per_tree": "colsample_bytree",
    "colsample_bytree": "colsample_bytree",
    "max_depth": "max_depth",
    "min_rows": "min_rows",
    "min_child_weight": "min_child_weight",
    "nbins": "max_bins",
    "stopping_rounds": "early_stopping_rounds",
    "stopping_tolerance": "early_stopping_tolerance",
    "seed": "seed",
}

# Prefer these as top-level MLflow params (UI-friendly); full dump goes to JSON.
_PRIMARY_PARAM_KEYS = (
    "n_estimators",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "max_depth",
    "min_rows",
    "min_child_weight",
    "max_bins",
    "early_stopping_rounds",
    "seed",
)


def _jsonable(value: Any) -> Any:
    """Coerce an H2O param value into something JSON / MLflow can store."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        # H2O sometimes returns [name, value] pairs for enum-like fields.
        if len(value) == 2 and isinstance(value[0], str):
            return _jsonable(value[1])
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def _raw_leader_params(leader: Any) -> dict[str, Any]:
    """Pull a flat dict of hyperparameters from an H2O model object."""
    raw = getattr(leader, "actual_params", None)
    if not isinstance(raw, dict) or not raw:
        raw = getattr(leader, "params", None)
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for key, value in raw.items():
        # `params` on some estimators is {name: {actual, default, ...}}.
        if isinstance(value, dict) and "actual" in value:
            value = value["actual"]
        elif isinstance(value, dict) and "actual_value" in value:
            value = value["actual_value"]
        out[str(key)] = _jsonable(value)
    return out


def extract_leader_hyperparams(leader: Any) -> dict[str, Any]:
    """Build a hyperparams payload with H2O names plus familiar aliases."""
    raw = _raw_leader_params(leader)
    aliases: dict[str, Any] = {}
    for h2o_key, alias in _H2O_PARAM_ALIASES.items():
        if h2o_key in raw and raw[h2o_key] is not None:
            aliases.setdefault(alias, raw[h2o_key])

    return {
        "model_name": getattr(leader, "algo", None),
        "model_id": getattr(leader, "model_id", None),
        "params": raw,
        "aliases": aliases,
    }


def _log_param_safe(key: str, value: Any) -> None:
    if value is None or isinstance(value, (dict, list)):
        return
    text = str(value)
    if len(text) > 250:
        text = text[:247] + "..."
    try:
        mlflow.log_param(key, text)
    except Exception as exc:  # noqa: BLE001 — Databricks param-count / length limits
        logger.debug("Skipped MLflow param %s: %s", key, exc)


def _log_leader_params(leader: Any) -> dict[str, Any]:
    """Log readable hyperparams to MLflow params + a hyperparams.json artifact."""
    payload = extract_leader_hyperparams(leader)
    aliases = payload.get("aliases") or {}
    raw = payload.get("params") or {}

    for key in _PRIMARY_PARAM_KEYS:
        if key in aliases:
            _log_param_safe(key, aliases[key])

    # Also keep the H2O-native names under hp.* for exact re-inspection.
    for key in (
        "ntrees",
        "learn_rate",
        "sample_rate",
        "col_sample_rate",
        "max_depth",
        "min_rows",
        "nbins",
        "stopping_rounds",
        "seed",
    ):
        if key in raw:
            _log_param_safe(f"hp.{key}", raw[key])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hyperparams.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path="params")

    return payload


def run_h2o_automl(
    train_path: str,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Run H2O AutoML on `train_path`, log the leader to MLflow, return its scorecard.

    Returns a JSON-serializable dict (it travels through Airflow XCom and is a
    valid input to `ampops.training.registry.register_champion`):
    `{model_name, run_id, mape, rmse, mae, n_train, n_valid}`.
    """
    _configure_tracking(tracking_uri)
    mlflow.set_experiment(experiment or config.MLFLOW_EXPERIMENT)

    df = pd.read_parquet(train_path)
    fit_df, valid_df = time_split(df, test_months=VALIDATION_MONTHS)
    features = feature_columns(df)

    _ensure_java_home()
    port = _free_port()
    h2o.init(port=port, start_h2o=True)
    try:
        fit_h2o = h2o.H2OFrame(fit_df[[*features, config.TARGET]])
        valid_h2o = h2o.H2OFrame(valid_df[[*features, config.TARGET]])

        started = time.perf_counter()
        aml = H2OAutoML(
            max_runtime_secs=config.AUTOML_MAX_RUNTIME_SECS,
            max_models=config.AUTOML_MAX_MODELS,
            seed=config.RANDOM_SEED,
            nfolds=0,
            sort_metric="RMSE",
        )
        aml.train(
            x=features,
            y=config.TARGET,
            training_frame=fit_h2o,
            validation_frame=valid_h2o,
        )

        leader = aml.leader
        y_pred = leader.predict(valid_h2o).as_data_frame()["predict"].to_numpy()
        y_valid = valid_df[config.TARGET]
        metrics = evaluate(y_valid, y_pred)
        model_name = leader.algo
        duration_seconds = time.perf_counter() - started

        with mlflow.start_run(run_name=f"automl-{model_name}") as run:
            mlflow.log_param("eval_name", EVAL_NAME_VALIDATION)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_id", leader.model_id)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("validation_months", VALIDATION_MONTHS)
            mlflow.log_param("horizon_hours", config.HORIZON_HOURS)
            mlflow.log_param("automl_max_runtime_secs", config.AUTOML_MAX_RUNTIME_SECS)
            mlflow.log_param("automl_max_models", config.AUTOML_MAX_MODELS)
            mlflow.log_metrics({**metrics, "duration_seconds": duration_seconds})
            hyperparams = _log_leader_params(leader)
            mlflow.h2o.log_model(leader, artifact_path="model")

            result = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "n_train": len(fit_df),
                "n_valid": len(valid_df),
                "hyperparams": hyperparams.get("aliases") or {},
                **metrics,
            }

        logger.info(
            "H2O AutoML leader %s (%s) | MAPE %.4f | RMSE %.1f MW | run %s | hp %s",
            leader.model_id,
            model_name,
            metrics["mape"],
            metrics["rmse"],
            result["run_id"],
            result["hyperparams"],
        )
        return result
    finally:
        h2o.cluster().shutdown()


def evaluate_on_test(
    model_uri: str,
    test_path: str,
    tracking_uri: str | None = None,
) -> dict[str, float]:
    """Score an already-registered model against the sealed test holdout.

    Deliberately separate from `run_h2o_automl`: model selection is decided on
    train/validate alone (see module docstring), and this function only ever
    runs after a champion has already been chosen and registered. It reloads
    the model by its MLflow registry URI rather than reusing anything from the
    training run, so it works equally well on any registered version, not just
    one freshly produced by this pipeline. Same H2O cluster lifecycle
    discipline as `run_h2o_automl` (fresh port, `finally`-shutdown) since this
    runs as its own Airflow task/process with no cluster to inherit.
    """
    _configure_tracking(tracking_uri)

    test_df = pd.read_parquet(test_path)
    features = feature_columns(test_df)

    _ensure_java_home()
    port = _free_port()
    h2o.init(port=port, start_h2o=True)
    try:
        model = mlflow.h2o.load_model(model_uri)
        test_h2o = h2o.H2OFrame(test_df[[*features, config.TARGET]])
        y_pred = model.predict(test_h2o).as_data_frame()["predict"].to_numpy()
        metrics = evaluate(test_df[config.TARGET], y_pred)

        logger.info(
            "Test-set evaluation of %s | MAPE %.4f | RMSE %.1f MW | n=%d",
            model_uri,
            metrics["mape"],
            metrics["rmse"],
            len(test_df),
        )
        return {
            "test_mape": metrics["mape"],
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "n_test": len(test_df),
        }
    finally:
        h2o.cluster().shutdown()
