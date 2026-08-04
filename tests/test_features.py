"""Tests for feature engineering, centred on the no-leakage guarantee.

Leakage is the failure that makes a model look excellent in the bake-off and
useless in production. `test_no_feature_sees_load_newer_than_horizon` is the
test that has to keep passing as the team adds features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ampops import config
from ampops.features import build


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    """Two months of synthetic hourly data with a distinctive load signal."""
    times = pd.date_range("2015-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time": times,
            "COMED_MW": 10_000 + 500 * np.sin(np.arange(len(times)) / 24) + rng.normal(0, 10, len(times)),
            "temperature_2m": rng.normal(5, 3, len(times)),
        }
    )


class TestCalendarFeatures:
    def test_expected_columns_are_added(self, hourly_frame):
        out = build.add_calendar_features(hourly_frame)

        for col in [
            "hour", "dayofweek", "month", "is_weekend", "is_holiday",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        ]:
            assert col in out.columns

    def test_calendar_hour_is_chicago_local_not_the_utc5_grid(self):
        """Winter rows are UTC-6 locally, so the local hour trails the grid by one.

        Demand follows the wall clock, so this offset matters for the model.
        """
        df = pd.DataFrame({"time": pd.to_datetime(["2015-01-15 12:00"]), "COMED_MW": [1.0]})
        assert build.add_calendar_features(df)["hour"].iloc[0] == 11

    def test_summer_local_hour_matches_the_grid(self):
        """CDT is UTC-5, identical to the weather grid — no shift."""
        df = pd.DataFrame({"time": pd.to_datetime(["2015-07-15 12:00"]), "COMED_MW": [1.0]})
        assert build.add_calendar_features(df)["hour"].iloc[0] == 12

    def test_holiday_flag_fires_on_july_fourth(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2015-07-04 12:00", "2015-07-07 12:00"]),
                "COMED_MW": [1.0, 1.0],
            }
        )
        assert build.add_calendar_features(df)["is_holiday"].tolist() == [1, 0]

    def test_weekend_flag(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2015-07-04 12:00", "2015-07-06 12:00"]),
                "COMED_MW": [1.0, 1.0],
            }
        )
        assert build.add_calendar_features(df)["is_weekend"].tolist() == [1, 0]

    def test_cyclical_encoding_makes_hour_23_adjacent_to_hour_0(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2015-07-01 23:00", "2015-07-02 00:00"]),
                "COMED_MW": [1.0, 1.0],
            }
        )
        out = build.add_calendar_features(df)
        distance = np.hypot(
            out["hour_sin"].iloc[0] - out["hour_sin"].iloc[1],
            out["hour_cos"].iloc[0] - out["hour_cos"].iloc[1],
        )
        assert distance < 0.3  # one step around a unit circle of 24


class TestLagFeatures:
    def test_lags_match_manual_shift_on_a_contiguous_series(self, hourly_frame):
        out = build.add_lag_features(hourly_frame)
        expected = hourly_frame["COMED_MW"].shift(24)

        pd.testing.assert_series_equal(
            out["load_lag_24h"], expected, check_names=False
        )

    def test_lags_respect_hours_not_rows_across_a_gap(self):
        """A row-based shift would pull the wrong observation across a missing hour."""
        times = list(pd.date_range("2015-01-01 00:00", periods=30, freq="h"))
        del times[5]  # puncture the series
        df = pd.DataFrame({"time": times, "COMED_MW": np.arange(len(times), dtype=float)})

        out = build.add_lag_features(df)
        row = out[out["time"] == pd.Timestamp("2015-01-02 00:00")].iloc[0]

        # 24h earlier is 2015-01-01 00:00, whose load is 0.0 — not the value a
        # naive .shift(24) would land on after the gap.
        assert row["load_lag_24h"] == pytest.approx(0.0)

    def test_gap_hours_are_not_reintroduced_as_rows(self):
        times = list(pd.date_range("2015-01-01 00:00", periods=30, freq="h"))
        del times[5]
        df = pd.DataFrame({"time": times, "COMED_MW": np.arange(len(times), dtype=float)})

        out = build.add_lag_features(df)

        assert len(out) == len(df)
        assert pd.Timestamp("2015-01-01 05:00") not in set(out["time"])


class TestNoLeakage:
    def test_no_t_minus_1_lag_exists(self, hourly_frame):
        """A t-1 lag is illegal under a 24h-ahead framing."""
        out = build.build_features(hourly_frame)
        assert "load_lag_1h" not in out.columns

    def test_no_feature_sees_load_newer_than_horizon(self, hourly_frame):
        """The core guarantee, tested empirically.

        Perturb the target at every timestamp strictly newer than t-24 relative
        to a probe row, rebuild features, and assert the probe row's features
        are byte-identical. If any feature peeked into that window, its value
        would move.
        """
        probe_time = hourly_frame["time"].iloc[500]
        cutoff = probe_time - pd.Timedelta(hours=config.HORIZON_HOURS)

        baseline = build.build_features(hourly_frame)
        baseline_row = baseline[baseline["time"] == probe_time].iloc[0]

        perturbed = hourly_frame.copy()
        mask = perturbed["time"] > cutoff
        perturbed.loc[mask, "COMED_MW"] += 5_000.0

        after = build.build_features(perturbed)
        after_row = after[after["time"] == probe_time].iloc[0]

        for col in build.feature_columns(baseline):
            assert baseline_row[col] == pytest.approx(after_row[col]), (
                f"feature {col!r} changed when load newer than t-{config.HORIZON_HOURS}h "
                "was perturbed — it leaks the future"
            )

    def test_rolling_windows_are_horizon_shifted(self, hourly_frame):
        """The 24h rolling mean at t must cover [t-48, t-24], not [t-24, t]."""
        out = build.build_features(hourly_frame)
        probe = out.iloc[100]
        window_end = probe["time"] - pd.Timedelta(hours=config.HORIZON_HOURS)
        window_start = window_end - pd.Timedelta(hours=23)

        source = hourly_frame.set_index("time")["COMED_MW"]
        expected = source.loc[window_start:window_end].mean()

        assert probe["load_roll_mean_24h"] == pytest.approx(expected)


class TestBuildFeatures:
    def test_warm_up_rows_are_dropped(self, hourly_frame):
        out = build.build_features(hourly_frame)

        assert out[[c for c in out.columns if c.startswith("load_")]].notna().all().all()
        assert len(out) == len(hourly_frame) - max(config.LOAD_ROLLING_WINDOWS) - config.HORIZON_HOURS + 1

    def test_feature_columns_excludes_time_and_target(self, hourly_frame):
        out = build.build_features(hourly_frame)
        features = build.feature_columns(out)

        assert config.TIME_COL not in features
        assert config.TARGET not in features
        assert "temperature_2m" in features
