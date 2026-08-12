"""Single source of truth for paths, constants, and training config.

Paths are env-driven so the same code runs in three places without edits:
  - a notebook (cwd = notebooks/)
  - a pytest run (cwd = repo root)
  - the Airflow container (data mounted at /opt/airflow/data)

Set AMPOPS_DATA_DIR to override; otherwise we walk up from this file to the
repo root, which is correct for every local invocation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --- Paths ------------------------------------------------------------------

# src/ampops/config.py -> src/ampops -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("AMPOPS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

COMED_CSV = RAW_DIR / "COMED_hourly.csv"
WEATHER_CSV = RAW_DIR / "open-meteo-41.86N87.65W179m.csv"

# Stage outputs. Each DAG task writes one of these so a failed run is
# debuggable at exactly the task that broke.
COMED_INTERIM = INTERIM_DIR / "comed_hourly.parquet"
WEATHER_INTERIM = INTERIM_DIR / "weather_hourly.parquet"
JOINED_PARQUET = PROCESSED_DIR / "joined_hourly.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"

# --- Data contract ----------------------------------------------------------

TARGET = "COMED_MW"
TIME_COL = "time"

# The Open-Meteo export is two concatenated exports in one file: an hourly
# block, then a blank line, a second header, and a daily-aggregated re-export
# of the same period. These two constants slice out the hourly block only.
WEATHER_SKIPROWS = 3  # 3-line metadata preamble; header lands on line 4
WEATHER_NROWS = 87648  # stops before the blank separator + second header

# The overlap window is COMED's true range, not padded to end-of-day.
WINDOW_START = pd.Timestamp("2011-01-01 01:00:00")
WINDOW_END = pd.Timestamp("2018-08-03 00:00:00")

# Weather is stamped on a fixed UTC-5 offset that never shifts for DST
# (verified: utc_offset_seconds=-18000, and summer evapotranspiration peaks in
# hour 13, matching Chicago solar noon under UTC-5).
WEATHER_UTC_OFFSET_HOURS = -5
SOURCE_TZ = "America/Chicago"

# How COMED's DST-aware timestamps are mapped onto the weather grid.
#
#   "eastern" (default) — the two sources are one hour apart in summer and
#       aligned in winter, so summer rows shift -1h and winter rows shift 0.
#       Two mechanisms explain this and both predict the same correction: PJM
#       publishes every zone on Eastern time (including ComEd, which sits in
#       Central), and/or the stamps are hour-*ending*. They are observationally
#       equivalent here.
#   "chicago_local" — the original reading in notebooks/01_join_and_eda.ipynb:
#       stamps are Chicago wall clock, hour-beginning, so winter rows shift +1h
#       and summer rows shift 0. Kept so the notebook remains reproducible.
#
# Adopted after the summer load-vs-temperature test came out in favour of the
# correction; see docs/timezone_alignment_finding.md. Override with
# AMPOPS_TZ_STRATEGY.
TIMEZONE_STRATEGY = os.getenv("AMPOPS_TZ_STRATEGY", "eastern")
COMED_EASTERN_TZ = "America/New_York"

# Expected shape of the joined dataset, validated after every run. Matches
# notebooks/01_join_and_eda.ipynb exactly — a drift here means the port broke.
EXPECTED_JOINED_ROWS = 66_493
EXPECTED_JOINED_COLS = 32
ROW_COUNT_TOLERANCE = 0.01  # 1%

# Plausible operating band for ComEd regional load, used by the raw/joined
# quality gates. Observed range in the source data is 7,237-23,753 MW.
LOAD_MIN_MW = 1_000.0
LOAD_MAX_MW = 40_000.0

# --- Feature engineering ----------------------------------------------------

# Day-ahead framing: predictions are made 24h out, so no feature may reference
# load newer than t-24. This constant is what enforces that everywhere.
HORIZON_HOURS = 24

LOAD_LAGS = [24, 168]  # t-24h (yesterday), t-168h (same hour last week)
LOAD_ROLLING_WINDOWS = [24, 168]  # rolling means, all shifted by HORIZON_HOURS

HOLIDAY_COUNTRY = "US"
HOLIDAY_SUBDIV = "IL"

# --- Train/test split -------------------------------------------------------

# README specifies a final-12-months holdout; docs/AmpOps_Project_Context.md
# suggests 2-3 months. Config-driven so the team can flip it in one place.
TEST_MONTHS = 12

# --- Training ---------------------------------------------------------------

# Default tracking is Databricks MLflow (`MLFLOW_TRACKING_URI=databricks` plus
# DATABRICKS_HOST / DATABRICKS_TOKEN). Local compose MLflow remains available
# by setting MLFLOW_TRACKING_URI=http://mlflow:5000 (in Docker) or
# http://localhost:5050 (host).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ampops-demand-forecasting")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
REGISTERED_MODEL_NAME = os.getenv("AMPOPS_MODEL_NAME", "ampops-demand-forecaster")
# When the registered name is not already `catalog.schema.model`, prefix it.
# Leave empty if using workspace registry or a fully-qualified AMPOPS_MODEL_NAME.
UC_MODEL_PREFIX = os.getenv("AMPOPS_UC_MODEL_PREFIX", "").strip()


def resolve_registered_model_name(name: str | None = None) -> str:
    """Return a registry model name, applying AMPOPS_UC_MODEL_PREFIX if needed."""
    resolved = name or REGISTERED_MODEL_NAME
    if resolved.count(".") >= 2:
        return resolved
    if UC_MODEL_PREFIX:
        return f"{UC_MODEL_PREFIX}.{resolved}"
    return resolved


RANDOM_SEED = 42

# H2O AutoML search budget. These constants also govern smoke/CI runs (not
# just full retrains), so keep the defaults to "a few minutes," not "as good
# as possible" — override with the env vars for a deeper search on demand.
AUTOML_MAX_RUNTIME_SECS = int(os.getenv("AMPOPS_AUTOML_MAX_RUNTIME_SECS", "300"))
AUTOML_MAX_MODELS = int(os.getenv("AMPOPS_AUTOML_MAX_MODELS", "10"))

# MAPE is the headline metric (scale-free, easy to justify); RMSE is the
# secondary check because demand spikes are the failure mode that matters.
PRIMARY_METRIC = "mape"
