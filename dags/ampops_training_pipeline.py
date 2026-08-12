"""AmpOps end-to-end training pipeline.

    ingest_raw -> validate_raw -> clean_and_join -> validate_joined
        -> build_features -> split_train_test
        -> run_automl (H2O AutoML search over algorithms + hyperparameters)
        -> register_champion -> evaluate_test

This module is orchestration only: every task body is a thin call into the
`ampops` package under src/. That separation is deliberate — the business logic
stays unit-testable without an Airflow scheduler, and the DAG stays readable as
a picture of the pipeline.

Tasks exchange file *paths* through XCom, never DataFrames. XCom is backed by
the Airflow metadata database and is not a data transport.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from ampops import config
from ampops.data import clean, ingest, join, validate
from ampops.features.build import build_features as build_features_fn
from ampops.features.split import time_split
from ampops.training import automl, registry
from ampops.utils.io import read_parquet, write_parquet

NAIVE_JOIN_PARQUET = config.INTERIM_DIR / "naive_join.parquet"

DEFAULT_ARGS = {
    "owner": "miguel",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=2),
}


@dag(
    dag_id="ampops_training_pipeline",
    description="Clean, join, feature-engineer, AutoML search, register, and test-evaluate the demand forecaster",
    # Manual trigger for the demo. Switch to "@daily" for a scheduled retrain.
    schedule=None,
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ampops", "training", "mlops-final"],
)
def ampops_training_pipeline():
    @task
    def ingest_raw() -> dict[str, str]:
        """Read both raw CSVs and land them as parquet in data/interim/."""
        comed = ingest.load_comed()
        weather = ingest.load_weather_hourly()
        return {
            "comed": write_parquet(comed, config.COMED_INTERIM),
            "weather": write_parquet(weather, config.WEATHER_INTERIM),
        }

    @task
    def validate_raw(paths: dict[str, str]) -> dict[str, str]:
        """Fail the run before any transformation if a source file is malformed."""
        validate.validate_raw_comed(read_parquet(paths["comed"]))
        validate.validate_raw_weather(read_parquet(paths["weather"]))
        return paths

    @task
    def clean_and_join(paths: dict[str, str]) -> dict[str, str]:
        """Dedup, realign onto the fixed UTC-5 grid, trim, normalize, join.

        Also writes the deliberately-uncorrected `naive_join` as the control
        case for the DST alignment gate downstream.
        """
        comed_raw = read_parquet(paths["comed"])
        weather_raw = read_parquet(paths["weather"])

        comed = clean.dedup_comed(comed_raw)
        comed_realigned = clean.trim_window(clean.realign_to_fixed_utc5(comed))

        weather = clean.trim_window(clean.normalize_weather_columns(weather_raw))

        joined = join.join_hourly(comed_realigned, weather)
        naive = join.naive_join(comed, weather)

        return {
            "joined": write_parquet(joined, config.JOINED_PARQUET),
            "naive": write_parquet(naive, NAIVE_JOIN_PARQUET),
        }

    @task
    def validate_joined(paths: dict[str, str]) -> str:
        """Quality gate plus the DST alignment proof.

        The alignment check is the one gate that can catch a silently-wrong
        join: row counts and null counts look perfect either way.
        """
        joined = read_parquet(paths["joined"])
        validate.validate_joined(joined)
        validate.check_dst_alignment(joined, read_parquet(paths["naive"]))
        return paths["joined"]

    @task
    def build_features(joined_path: str) -> str:
        """Calendar, cyclical, lag, and rolling features (all horizon-safe)."""
        features = build_features_fn(read_parquet(joined_path))
        return write_parquet(features, config.FEATURES_PARQUET)

    # multiple_outputs is explicit here because `splits["train"]` below is
    # resolved at DAG-parse time — it needs each dict key pushed as its own
    # XCom. TaskFlow would infer this from the return annotation, but relying
    # on that inference is the kind of thing that breaks silently on upgrade.
    @task(multiple_outputs=True)
    def split_train_test(features_path: str) -> dict[str, str]:
        """Chronological split. The test tail stays sealed until API validation."""
        train, test = time_split(read_parquet(features_path))
        return {
            "train": write_parquet(train, config.TRAIN_PARQUET),
            "test": write_parquet(test, config.TEST_PARQUET),
        }

    @task
    def run_automl(train_path: str) -> dict:
        """H2O AutoML search over algorithms + hyperparameters; returns the leader's scorecard."""
        return automl.run_h2o_automl(train_path)

    @task
    def register(champion: dict) -> dict:
        """Promote the champion into the MLflow Model Registry as @champion."""
        return registry.register_champion(champion)

    @task
    def evaluate_test(registration: dict, test_path: str) -> dict:
        """Score the sealed test holdout using the registration's model_uri.

        Prefer the registry URI when promotion succeeded; otherwise fall back to
        `runs:/<run_id>/model` (Databricks soft-fail path). Selection never uses
        this holdout.
        """
        model_uri = registration["model_uri"]
        metrics = automl.evaluate_on_test(model_uri, test_path)
        return registry.tag_test_metrics(registration, metrics)

    raw = ingest_raw()
    validated_raw = validate_raw(raw)
    joined_paths = clean_and_join(validated_raw)
    joined_path = validate_joined(joined_paths)
    features_path = build_features(joined_path)
    splits = split_train_test(features_path)

    evaluate_test(register(run_automl(splits["train"])), splits["test"])


ampops_training_pipeline()
