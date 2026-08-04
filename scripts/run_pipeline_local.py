#!/usr/bin/env python
"""Run the data stages of the pipeline outside Airflow.

Same function calls the DAG makes, minus the scheduler. Use this to check a
change to the `ampops` package in seconds rather than waiting on a DAG run.
Training and registration are skipped — those need the MLflow server, so run
them through Airflow (`make dag-test`).

    make pipeline-local
    # or: python scripts/run_pipeline_local.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ampops import config  # noqa: E402
from ampops.data import clean, ingest, join, validate  # noqa: E402
from ampops.features.build import build_features, feature_columns  # noqa: E402
from ampops.features.split import time_split  # noqa: E402
from ampops.utils.io import write_parquet  # noqa: E402


def main() -> None:
    comed_raw = ingest.load_comed()
    weather_raw = ingest.load_weather_hourly()

    validate.validate_raw_comed(comed_raw)
    validate.validate_raw_weather(weather_raw)

    comed = clean.dedup_comed(comed_raw)
    comed_realigned = clean.trim_window(clean.realign_to_fixed_utc5(comed))
    weather = clean.trim_window(clean.normalize_weather_columns(weather_raw))

    joined = join.join_hourly(comed_realigned, weather)
    validate.validate_joined(joined)
    validate.check_dst_alignment(joined, join.naive_join(comed, weather))
    write_parquet(joined, config.JOINED_PARQUET)

    features = build_features(joined)
    write_parquet(features, config.FEATURES_PARQUET)

    train, test = time_split(features)
    write_parquet(train, config.TRAIN_PARQUET)
    write_parquet(test, config.TEST_PARQUET)

    print(
        f"\nDone. joined={joined.shape} features={features.shape} "
        f"model_features={len(feature_columns(features))} "
        f"train={len(train)} test={len(test)}"
    )


if __name__ == "__main__":
    main()
