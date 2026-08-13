"""Parity tests proving the serving-time feature frame matches training exactly.

`docs/serving_contract.md` §1 spells out the real risk this guards against:
`mlflow.h2o.log_model` was called with no `signature=`/`input_example=`, so
MLflow and H2O perform **zero** column or dtype validation at predict time. A
wrong dtype (e.g. an int64 weather column arriving as float64 after a Redis
round trip, or a column in the wrong order) does not raise anywhere — H2O
silently reinterprets the column and predictions quietly degrade. The only
thing standing between that failure mode and production is this test: it
takes real rows out of `data/processed/train.parquet` (what the champion was
actually trained on) and proves `app.features.build_inference_frame`,
building the same rows from `data/processed/joined_hourly.parquet` through
the feature store, reproduces them byte-for-byte, dtypes included.

Both parquet files are gitignored data artifacts (see CLAUDE.md), so the whole
module is skipped when they are not present on disk rather than failing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ampops import config
from ampops.features.build import feature_columns

if not (config.TRAIN_PARQUET.exists() and config.JOINED_PARQUET.exists()):
    pytest.skip(
        "data/processed/train.parquet and joined_hourly.parquet are gitignored "
        "data artifacts and are not present in this environment",
        allow_module_level=True,
    )

from app.features import build_inference_frame  # noqa: E402
from app.store import InsufficientHistoryError, ParquetFeatureStore  # noqa: E402

# Single-grid dataset; "COMED" matches the documented AMPOPS_GRID_ID default
# (serving_contract.md §9). Not otherwise pinned by the module surface, so this
# is a naming convention this test file establishes for Build to follow.
GRID_ID = "COMED"

TRAIN = pd.read_parquet(config.TRAIN_PARQUET)
FEATURES = feature_columns(TRAIN)


@pytest.fixture(scope="module")
def store() -> ParquetFeatureStore:
    """A `ParquetFeatureStore` seeded from the full joined dataset.

    `joined_hourly.parquet` is the feature store's source of truth per
    serving_contract.md §4a/§2 — it is what `scripts/seed_redis.py` would load
    into Redis in the real deployment.
    """
    joined = pd.read_parquet(config.JOINED_PARQUET)
    return ParquetFeatureStore(joined)


def _training_row(ts: pd.Timestamp) -> pd.DataFrame:
    """The single training row for `ts`, restricted to the 49 feature columns."""
    row = TRAIN[TRAIN[config.TIME_COL] == ts]
    assert len(row) == 1, f"expected exactly one training row for {ts}, found {len(row)}"
    return row[FEATURES].reset_index(drop=True)


class TestFrameShape:
    def test_the_frame_has_exactly_49_feature_columns(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        frame = build_inference_frame(store, GRID_ID, [ts])

        assert len(frame.columns) == 49

    def test_comed_mw_and_time_are_not_feature_columns(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        frame = build_inference_frame(store, GRID_ID, [ts])

        assert config.TARGET not in frame.columns
        assert config.TIME_COL not in frame.columns

    def test_columns_are_returned_in_feature_columns_order(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        frame = build_inference_frame(store, GRID_ID, [ts])

        assert list(frame.columns) == FEATURES


class TestParityWithTraining:
    """The crown-jewel test: serving-time features must equal training-time features.

    Each case covers a distinct code path that could silently diverge: a
    summer timestamp (CDT, no local-time shift versus the UTC-5 grid), a
    winter timestamp (CST, one-hour local-time shift — see
    `test_calendar_hour_is_chicago_local_not_the_utc5_grid` in
    test_features.py), a US holiday (exercises `is_holiday`), and a 24h batch
    (exercises the single-window batch path required by serving_contract.md §2).
    """

    def test_features_match_training_for_a_summer_timestamp(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")  # CDT, no local-time shift
        frame = build_inference_frame(store, GRID_ID, [ts])

        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), _training_row(ts), check_like=False
        )

    def test_features_match_training_for_a_winter_timestamp(self, store):
        ts = pd.Timestamp("2015-01-15 12:00:00")  # CST, one-hour local shift
        frame = build_inference_frame(store, GRID_ID, [ts])

        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), _training_row(ts), check_like=False
        )

    def test_features_match_training_for_a_us_holiday(self, store):
        ts = pd.Timestamp("2015-07-04 12:00:00")  # Independence Day
        frame = build_inference_frame(store, GRID_ID, [ts])

        row = _training_row(ts)
        assert row["is_holiday"].iloc[0] == 1

        pd.testing.assert_frame_equal(frame.reset_index(drop=True), row, check_like=False)

    def test_features_match_training_for_a_24_hour_batch(self, store):
        timestamps = list(pd.date_range("2015-06-01 00:00:00", periods=24, freq="h"))
        frame = build_inference_frame(store, GRID_ID, timestamps)

        expected = pd.concat([_training_row(ts) for ts in timestamps], ignore_index=True)

        assert len(frame) == 24
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), expected.reset_index(drop=True), check_like=False
        )


class TestInsufficientHistory:
    def test_a_timestamp_too_close_to_the_start_of_the_store_raises(self, store):
        """The store's earliest data is 2011-01-01 01:00; 192h of history is
        unavailable for anything within the first 192 hours of that range.
        """
        joined_start = pd.Timestamp("2011-01-01 01:00:00")
        ts = joined_start + pd.Timedelta(hours=10)

        with pytest.raises(InsufficientHistoryError):
            build_inference_frame(store, GRID_ID, [ts])

    def test_a_timestamp_entirely_outside_the_seeded_range_raises(self, store):
        with pytest.raises(InsufficientHistoryError):
            build_inference_frame(store, GRID_ID, [pd.Timestamp("2030-01-01 00:00:00")])
