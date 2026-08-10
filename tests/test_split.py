"""Tests for the chronological train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ampops import config
from ampops.features.split import time_split


@pytest.fixture
def three_years() -> pd.DataFrame:
    times = pd.date_range("2015-01-01", "2018-01-01", freq="h")
    return pd.DataFrame({"time": times, "COMED_MW": np.arange(len(times), dtype=float)})


def test_train_strictly_precedes_test(three_years):
    train, test = time_split(three_years)

    assert train["time"].max() < test["time"].min()


def test_test_span_matches_configured_months(three_years):
    train, test = time_split(three_years, test_months=6)
    expected_cutoff = three_years["time"].max() - pd.DateOffset(months=6)

    assert test["time"].min() > expected_cutoff
    assert train["time"].max() <= expected_cutoff


def test_no_rows_are_lost_or_duplicated(three_years):
    train, test = time_split(three_years)

    assert len(train) + len(test) == len(three_years)
    assert set(train["time"]).isdisjoint(set(test["time"]))


def test_default_uses_configured_test_months(three_years):
    _, test = time_split(three_years)
    span_days = (test["time"].max() - test["time"].min()).days

    assert span_days == pytest.approx(config.TEST_MONTHS * 30.4, abs=10)


def test_unsorted_input_is_handled(three_years):
    shuffled = three_years.sample(frac=1, random_state=0).reset_index(drop=True)
    train, test = time_split(shuffled)

    assert train["time"].is_monotonic_increasing
    assert train["time"].max() < test["time"].min()


def test_raises_when_history_is_too_short():
    df = pd.DataFrame(
        {"time": pd.date_range("2018-01-01", periods=48, freq="h"), "COMED_MW": 1.0}
    )
    with pytest.raises(ValueError, match="empty side"):
        time_split(df, test_months=12)
