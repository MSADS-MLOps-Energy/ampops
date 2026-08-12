#!/usr/bin/env python
"""Run H2O AutoML → (best-effort) register → sealed test eval → Databricks MLflow.

Training still runs on the host (or in Airflow); Databricks is the MLflow
tracking/registry backend. Databricks AutoML is not used — see
`docs/automl_implementation.md` and `docs/databricks_experiment_tracking.md`.

Requires Java 8–17 on PATH (Homebrew `openjdk@17` is auto-detected) and `h2o`.

    conda activate ampops
    # .env should set MLFLOW_TRACKING_URI=databricks + DATABRICKS_HOST/TOKEN
    make pipeline-local   # or: make data-export after a DAG run
    make train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from ampops import config  # noqa: E402
from ampops.training import automl, registry  # noqa: E402
from ampops.utils.io import get_logger  # noqa: E402

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default=str(config.TRAIN_PARQUET),
        help="Path to train.parquet",
    )
    parser.add_argument(
        "--test-path",
        default=str(config.TEST_PARQUET),
        help="Path to sealed test.parquet",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip MLflow Model Registry promotion",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip sealed test evaluation",
    )
    args = parser.parse_args(argv)

    train_path = Path(args.train_path)
    if not train_path.exists():
        raise SystemExit(
            f"Missing {train_path}. Run `make pipeline-local` or "
            "`make data-export` after a DAG run first."
        )

    logger.info(
        "Tracking URI=%s experiment=%s",
        config.MLFLOW_TRACKING_URI,
        config.MLFLOW_EXPERIMENT,
    )

    champion = automl.run_h2o_automl(str(train_path))
    summary: dict = {"champion": champion}

    registration: dict | None = None
    if args.skip_register:
        registration = {
            "registered_model": None,
            "version": None,
            "run_id": champion["run_id"],
            "algorithm": champion["model_name"],
            "model_uri": f"runs:/{champion['run_id']}/model",
            "skipped": True,
        }
        summary["registration"] = registration
    else:
        registration = registry.register_champion(champion)
        summary["registration"] = registration

    if not args.skip_holdout:
        test_path = Path(args.test_path)
        if not test_path.exists():
            raise SystemExit(f"Missing {test_path}.")
        assert registration is not None
        metrics = automl.evaluate_on_test(registration["model_uri"], str(test_path))
        summary["holdout"] = registry.tag_test_metrics(registration, metrics)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
