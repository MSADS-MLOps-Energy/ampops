"""Champion selection and MLflow Model Registry promotion.

Owned by the experiment-tracking workstream. The DAG calls only
`select_champion` and `register_champion`; changing the promotion policy (say,
requiring the champion to beat the incumbent by some margin) means editing this
file, not the DAG.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def select_champion(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the candidate with the lowest primary metric (MAPE)."""
    if not results:
        raise ValueError("select_champion received no candidate results")

    champion = min(results, key=lambda r: r[config.PRIMARY_METRIC])
    ranking = sorted(results, key=lambda r: r[config.PRIMARY_METRIC])
    logger.info(
        "Bake-off ranking by %s: %s",
        config.PRIMARY_METRIC,
        ", ".join(f"{r['model_name']}={r[config.PRIMARY_METRIC]:.4f}" for r in ranking),
    )
    logger.info("Champion: %s (run %s)", champion["model_name"], champion["run_id"])
    return champion


def register_champion(
    champion: dict[str, Any],
    model_name: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register the champion's model artifact and promote it to Production.

    Semantic versioning: MLflow assigns the integer version, and we tag it with
    a `semantic_version` of `1.<version>.0` plus the metrics it won on, so the
    registry entry is self-describing for the deployment workstream.
    """
    model_name = model_name or config.REGISTERED_MODEL_NAME
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{champion['run_id']}/model"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_model_version_tag(model_name, version.version, "algorithm", champion["model_name"])
    client.set_model_version_tag(
        model_name, version.version, "semantic_version", f"1.{version.version}.0"
    )
    for metric in ("mape", "rmse"):
        if metric in champion:
            client.set_model_version_tag(
                model_name, version.version, metric, f"{champion[metric]:.6f}"
            )

    # Aliases are the modern replacement for stages; the serving layer resolves
    # `models:/<name>@champion` and always gets the current winner.
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
    }
