#!/usr/bin/env python
"""One-off migration: copy the existing local-compose champion into Databricks
Unity Catalog.

Does NOT retrain. Loads the already-trained H2O artifact from the local
compose MLflow server, infers a signature (the original run has none, and
Unity Catalog registration requires one), re-logs it to Databricks tracking,
and registers + tags it via `ampops.training.registry` -- the same functions
a real training run uses -- so the migrated version looks identical to one
`make train` would have produced, with an added provenance tag pointing back
at the source run.

Prereqs:
  - .env points at Databricks (see docs/databricks_experiment_tracking.md)
  - local compose `mlflow` container reachable at http://localhost:5050
    (`make airflow-up` if not already up)
  - Java on PATH for H2O (see docs/automl_implementation.md)

Usage:
    python scripts/migrate_champion_to_databricks.py --dry-run
    python scripts/migrate_champion_to_databricks.py

`mlflow.register_model` has no idempotency -- each real run creates a new
Databricks model version -- so run with --dry-run first and only run for
real once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import h2o  # noqa: E402
import mlflow  # noqa: E402
import mlflow.h2o  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from ampops import config  # noqa: E402
from ampops.features.build import feature_columns  # noqa: E402
from ampops.training import registry  # noqa: E402
from ampops.training.automl import _ensure_java_home, _free_port  # noqa: E402
from ampops.utils.io import get_logger  # noqa: E402

logger = get_logger(__name__)

SOURCE_TRACKING_URI = "http://localhost:5050"
SOURCE_MODEL_NAME = "ampops-demand-forecaster"
SOURCE_MODEL_VERSION = "5"
SOURCE_RUN_ID = "f81f81d43dc14aba9532a0319a4f6fe4"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the source model and infer a signature, but don't touch Databricks",
    )
    args = parser.parse_args(argv)

    dest_name = config.resolve_registered_model_name()
    if dest_name.count(".") != 2:
        raise SystemExit(
            f"Expected a 3-level Unity Catalog name, got {dest_name!r} -- "
            "check AMPOPS_UC_MODEL_PREFIX in .env"
        )

    # 1. Pull tags from the source model version -- this is the already-
    #    established source of truth for v5's metrics, no need to recompute.
    # registry_uri must be pinned explicitly too: MlflowClient otherwise falls
    # back to the ambient MLFLOW_REGISTRY_URI env var (databricks-uc once .env
    # is fixed), which would point registry calls at Databricks even though
    # tracking_uri says local.
    src_client = MlflowClient(tracking_uri=SOURCE_TRACKING_URI, registry_uri=SOURCE_TRACKING_URI)
    src_version = src_client.get_model_version(SOURCE_MODEL_NAME, SOURCE_MODEL_VERSION)
    if src_version.run_id != SOURCE_RUN_ID:
        raise SystemExit(
            f"Source model version's run_id ({src_version.run_id}) doesn't match "
            f"the expected {SOURCE_RUN_ID} -- local registry state may have changed "
            "since this script was written; update SOURCE_RUN_ID/SOURCE_MODEL_VERSION."
        )
    tags = dict(src_version.tags)
    logger.info("Source %s v%s tags=%s", SOURCE_MODEL_NAME, SOURCE_MODEL_VERSION, tags)

    # 2. Load the H2O artifact from the local server.
    mlflow.set_tracking_uri(SOURCE_TRACKING_URI)
    _ensure_java_home()
    port = _free_port()
    h2o.init(port=port, start_h2o=True)
    try:
        model = mlflow.h2o.load_model(f"runs:/{SOURCE_RUN_ID}/model")

        # 3. Infer a signature -- the source run has none, and Unity Catalog
        #    registration requires one.
        sample_df = pd.read_parquet(config.TEST_PARQUET).head(5)
        cols = feature_columns(sample_df)
        sample_input = sample_df[cols]
        sample_pred = model.predict(h2o.H2OFrame(sample_input)).as_data_frame()["predict"].to_numpy()
        signature = infer_signature(sample_input, sample_pred)
        logger.info("Inferred signature from %d sample rows, %d features", len(sample_input), len(cols))

        if args.dry_run:
            logger.info(
                "DRY RUN -- would register %s from source run %s with tags %s",
                dest_name,
                SOURCE_RUN_ID,
                tags,
            )
            return

        # 4. Re-log to Databricks tracking so the model has a run there.
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="migrate-champion-v5-to-uc") as run:
            mlflow.set_tags({"source_run_id": SOURCE_RUN_ID, "migrated_from": "local-compose"})
            mlflow.h2o.log_model(
                model, artifact_path="model", signature=signature, input_example=sample_input
            )
            new_run_id = run.info.run_id
    finally:
        h2o.cluster().shutdown()

    # 5. Register + tag via the same code path a real training run uses.
    champion = {
        "run_id": new_run_id,
        "model_name": tags["algorithm"],
        "mape": float(tags["mape"]),
        "rmse": float(tags["rmse"]),
    }
    registration = registry.register_champion(champion)
    if registration["skipped"]:
        raise SystemExit(f"Registration failed: {registration.get('registry_error')}")

    test_metrics = {
        "test_mape": float(tags["test_mape"]),
        "test_rmse": float(tags["test_rmse"]),
        "test_mae": float(tags["test_mae"]),
    }
    registry.tag_test_metrics(registration, test_metrics)

    client = MlflowClient()
    client.set_model_version_tag(
        registration["registered_model"], registration["version"], "source_run_id", SOURCE_RUN_ID
    )
    client.set_model_version_tag(
        registration["registered_model"], registration["version"], "migrated_from", "local-compose"
    )

    logger.info(
        "Registered %s v%s @champion (migrated from local run %s v%s)",
        registration["registered_model"],
        registration["version"],
        SOURCE_RUN_ID,
        SOURCE_MODEL_VERSION,
    )


if __name__ == "__main__":
    main()
