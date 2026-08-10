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

import socket
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
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or config.MLFLOW_EXPERIMENT)

    df = pd.read_parquet(train_path)
    fit_df, valid_df = time_split(df, test_months=VALIDATION_MONTHS)
    features = feature_columns(df)

    port = _free_port()
    h2o.init(port=port, start_h2o=True)
    try:
        fit_h2o = h2o.H2OFrame(fit_df[[*features, config.TARGET]])
        valid_h2o = h2o.H2OFrame(valid_df[[*features, config.TARGET]])

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

        with mlflow.start_run(run_name=f"automl-{model_name}") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_id", leader.model_id)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("validation_months", VALIDATION_MONTHS)
            mlflow.log_param("horizon_hours", config.HORIZON_HOURS)
            mlflow.log_param("automl_max_runtime_secs", config.AUTOML_MAX_RUNTIME_SECS)
            mlflow.log_param("automl_max_models", config.AUTOML_MAX_MODELS)
            mlflow.log_metrics(metrics)
            mlflow.h2o.log_model(leader, artifact_path="model")

            result = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "n_train": len(fit_df),
                "n_valid": len(valid_df),
                **metrics,
            }

        logger.info(
            "H2O AutoML leader %s (%s) | MAPE %.4f | RMSE %.1f MW | run %s",
            leader.model_id,
            model_name,
            metrics["mape"],
            metrics["rmse"],
            result["run_id"],
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
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)

    test_df = pd.read_parquet(test_path)
    features = feature_columns(test_df)

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
