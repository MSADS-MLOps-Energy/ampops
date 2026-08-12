"""MLflow Model Registry promotion (Databricks-aware).

Owned by the experiment-tracking workstream. The DAG calls `register_champion`
directly with the scorecard dict produced by `ampops.training.automl.run_h2o_automl`
(H2O AutoML already returns a single winner, so no separate champion-selection
step is needed).

On Databricks Free / restricted workspaces the legacy Model Registry is often
disabled and Unity Catalog may lack a usable catalog — registration then soft-
fails and returns a `runs:/…/model` URI so sealed-test scoring can still run.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def _configure_mlflow(tracking_uri: str | None = None) -> None:
    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    if config.MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(config.MLFLOW_REGISTRY_URI)


def register_champion(
    champion: dict[str, Any],
    model_name: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register the champion's model artifact and set the `@champion` alias.

    Soft-fails on Databricks registry/UC policy errors so training + holdout
    still complete. Callers should prefer `registration["model_uri"]` for
    reload (registry URI when successful, else `runs:/<run_id>/model`).
    """
    model_name = config.resolve_registered_model_name(model_name)
    _configure_mlflow(tracking_uri)

    run_model_uri = f"runs:/{champion['run_id']}/model"
    try:
        version = mlflow.register_model(model_uri=run_model_uri, name=model_name)

        client = MlflowClient()
        client.set_model_version_tag(
            model_name, version.version, "algorithm", champion["model_name"]
        )
        client.set_model_version_tag(
            model_name, version.version, "semantic_version", f"1.{version.version}.0"
        )
        for metric in ("mape", "rmse"):
            if metric in champion:
                client.set_model_version_tag(
                    model_name, version.version, metric, f"{champion[metric]:.6f}"
                )

        client.set_registered_model_alias(model_name, "champion", version.version)

        logger.info(
            "Registered %s v%s (%s, MAPE %.4f) and set alias @champion",
            model_name,
            version.version,
            champion["model_name"],
            champion.get("mape", float("nan")),
        )
        return {
            "registered_model": model_name,
            "version": version.version,
            "semantic_version": f"1.{version.version}.0",
            "run_id": champion["run_id"],
            "algorithm": champion["model_name"],
            "model_uri": f"models:/{model_name}@champion",
            "skipped": False,
        }
    except Exception as exc:  # noqa: BLE001 — Databricks policy / missing UC catalog
        logger.warning(
            "Model registry failed (%s). Continuing with run artifact %s. "
            "On Databricks Unity Catalog set AMPOPS_MODEL_NAME="
            "catalog.schema.model_name and MLFLOW_REGISTRY_URI=databricks-uc.",
            exc,
            run_model_uri,
        )
        return {
            "registered_model": None,
            "version": None,
            "semantic_version": None,
            "run_id": champion["run_id"],
            "algorithm": champion["model_name"],
            "model_uri": run_model_uri,
            "skipped": True,
            "registry_error": str(exc),
        }


def tag_test_metrics(
    registration: dict[str, Any],
    metrics: dict[str, float],
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Annotate sealed-test metrics on the registered version and/or the run.

    Always logs `test_*` metrics onto the training run. Version tags are only
    written when registration actually succeeded.
    """
    _configure_mlflow(tracking_uri)
    client = MlflowClient()
    run_id = registration["run_id"]

    for key in ("test_mape", "test_rmse", "test_mae"):
        if key in metrics:
            client.log_metric(run_id, key, metrics[key])

    if not registration.get("skipped") and registration.get("version"):
        for key in ("test_mape", "test_rmse", "test_mae"):
            if key in metrics:
                client.set_model_version_tag(
                    registration["registered_model"],
                    registration["version"],
                    key,
                    f"{metrics[key]:.6f}",
                )

    logger.info(
        "Logged test-set metrics for run %s (MAPE %.4f, RMSE %.1f MW)%s",
        run_id,
        metrics.get("test_mape", float("nan")),
        metrics.get("test_rmse", float("nan")),
        ""
        if registration.get("skipped")
        else f" and tagged {registration.get('registered_model')} "
        f"v{registration.get('version')}",
    )
    return {**registration, **metrics}
