"""End-to-end experiment pipeline: bake-off → champion → (optional) registry → holdout.

Designed to run locally against Databricks MLflow (`MLFLOW_TRACKING_URI=databricks`)
or against a self-hosted tracking server. Airflow's DAG calls the same building
blocks; this module is the single-process entrypoint for Collin's workstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ampops import config
from ampops.training.bakeoff import train_candidate
from ampops.training.evaluate import evaluate_holdout
from ampops.training.registry import register_champion, select_champion
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def run_experiment_pipeline(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    model_configs: list[dict[str, Any]] | None = None,
    register: bool = True,
    run_holdout: bool = True,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Train all candidates, pick a champion, optionally register + score test.

    Returns a summary dict with bake-off results, champion, optional registry
    info, and optional holdout metrics — all JSON-serializable.
    """
    train_path = str(train_path or config.TRAIN_PARQUET)
    test_path = str(test_path or config.TEST_PARQUET)
    model_configs = model_configs or config.MODEL_CONFIGS
    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    experiment = experiment or config.MLFLOW_EXPERIMENT

    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run `make pipeline-local` first "
            "(requires raw CSVs in data/raw/)."
        )

    logger.info(
        "Experiment pipeline → tracking=%s experiment=%s candidates=%s",
        tracking_uri,
        experiment,
        [c["model_name"] for c in model_configs],
    )

    results = [
        train_candidate(
            train_path=train_path,
            model_name=cfg["model_name"],
            params=cfg.get("params") or {},
            tracking_uri=tracking_uri,
            experiment=experiment,
            parent_run_id=cfg.get("_parent_run_id"),
        )
        for cfg in model_configs
    ]

    champion = select_champion(results)
    summary: dict[str, Any] = {
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "bakeoff": results,
        "champion": champion,
    }

    if register:
        try:
            summary["registry"] = register_champion(
                champion, tracking_uri=tracking_uri
            )
        except Exception as exc:  # noqa: BLE001 — surface UC/workspace policy errors
            logger.warning(
                "Model registry failed (%s). Continuing without @champion. "
                "On Databricks Unity Catalog set AMPOPS_MODEL_NAME="
                "catalog.schema.model_name and MLFLOW_REGISTRY_URI=databricks-uc.",
                exc,
            )
            summary["registry_error"] = str(exc)

    if run_holdout:
        if not Path(test_path).exists():
            raise FileNotFoundError(
                f"Missing {test_path}. Run `make pipeline-local` first."
            )
        summary["holdout"] = evaluate_holdout(
            champion,
            test_path=test_path,
            tracking_uri=tracking_uri,
            experiment=experiment,
        )

    logger.info(
        "Done. champion=%s validation_mape=%.4f holdout_mape=%s",
        champion["model_name"],
        champion["mape"],
        f"{summary['holdout']['mape']:.4f}" if "holdout" in summary else "skipped",
    )
    return summary
