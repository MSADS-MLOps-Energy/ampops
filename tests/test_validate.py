"""Tests for the data quality gates.

Each gate is tested against a deliberately corrupted frame. The corruption
patterns here mirror the drift scenarios in the monitoring workstream
(docs/AmpOps_Project_Context.md §5) — out-of-bounds meter readings, schema
drift, sensor dropout — so this file doubles as the seed for that work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ampops import config
from ampops.data import validate
from ampops.data.validate import DataValidationError


@pytest.fixture
def comed() -> pd.DataFrame:
    times = pd.date_range("2015-01-01", periods=500, freq="h")
    return pd.DataFrame({"Datetime": times, "COMED_MW": np.full(len(times), 11_000.0)})


@pytest.fixture
def weather() -> pd.DataFrame:
    times = pd.date_range("2015-01-01", periods=500, freq="h")
    return pd.DataFrame({"time": times, "temperature_2m": np.zeros(len(times))})


@pytest.fixture
def joined() -> pd.DataFrame:
    """A frame sized to pass the row-count gate."""
    n = config.EXPECTED_JOINED_ROWS
    return pd.DataFrame(
        {
            "time": pd.date_range("2011-01-01 01:00", periods=n, freq="h"),
            "COMED_MW": np.full(n, 11_000.0),
            "temperature_2m": np.zeros(n),
        }
    )


class TestValidateRawComed:
    def test_passes_on_clean_input(self, comed):
        validate.validate_raw_comed(comed)

    def test_rejects_missing_column(self, comed):
        with pytest.raises(DataValidationError, match="missing expected column"):
            validate.validate_raw_comed(comed.drop(columns=["COMED_MW"]))

    def test_rejects_sensor_dropout(self, comed):
        comed.loc[10:20, "COMED_MW"] = np.nan
        with pytest.raises(DataValidationError, match="null load readings"):
            validate.validate_raw_comed(comed)

    def test_rejects_out_of_bounds_meter_reading(self, comed):
        comed.loc[5, "COMED_MW"] = 900_000.0  # 10-100x historical max
        with pytest.raises(DataValidationError, match="plausible"):
            validate.validate_raw_comed(comed)

    def test_rejects_negative_load(self, comed):
        comed.loc[5, "COMED_MW"] = -50.0
        with pytest.raises(DataValidationError, match="plausible"):
            validate.validate_raw_comed(comed)

    def test_rejects_unsorted_input(self, comed):
        with pytest.raises(DataValidationError, match="not sorted"):
            validate.validate_raw_comed(comed.iloc[::-1].reset_index(drop=True))


class TestValidateRawWeather:
    def test_passes_on_clean_input(self, weather):
        validate.validate_raw_weather(weather)

    def test_rejects_duplicate_timestamps(self, weather):
        doubled = pd.concat([weather, weather.head(3)]).sort_values("time")
        with pytest.raises(DataValidationError, match="duplicate timestamps"):
            validate.validate_raw_weather(doubled)

    def test_rejects_non_hourly_grid(self, weather):
        punctured = weather.drop(index=100).reset_index(drop=True)
        with pytest.raises(DataValidationError, match="hourly grid"):
            validate.validate_raw_weather(punctured)


class TestValidateJoined:
    def test_passes_on_clean_input(self, joined):
        validate.validate_joined(joined)

    def test_rejects_schema_drift(self, joined):
        with pytest.raises(DataValidationError, match="missing expected column"):
            validate.validate_joined(joined.rename(columns={"COMED_MW": "load_mw"}))

    def test_rejects_nulls(self, joined):
        joined.loc[0, "temperature_2m"] = np.nan
        with pytest.raises(DataValidationError, match="null values"):
            validate.validate_joined(joined)

    def test_rejects_duplicate_timestamps(self, joined):
        joined.loc[1, "time"] = joined.loc[0, "time"]
        with pytest.raises(DataValidationError, match="duplicate timestamps"):
            validate.validate_joined(joined)

    def test_rejects_unexpected_row_count(self, joined):
        with pytest.raises(DataValidationError, match="outside"):
            validate.validate_joined(joined.head(1000))


class TestCheckDstAlignment:
    """The alignment gate, exercised against a synthetic year.

    The synthetic COMED series is generated on Chicago local time with a strong
    hour-of-day shape, then realigned exactly as the pipeline does. The naive
    control is the same series joined on uncorrected local timestamps.
    """

    @staticmethod
    def _build(realign: bool):
        from ampops.data import clean, join

        local = pd.date_range("2015-01-01", "2015-12-31 23:00", freq="h")
        # Match the COMED source shape: the 03:00 label is absent on the
        # spring-forward day (see ampops.data.clean.realign_to_fixed_utc5).
        local = local[local != pd.Timestamp("2015-03-08 03:00")]
        # Load peaks in the local afternoon — the signal the gate looks for.
        load = 10_000 + 3_000 * np.sin((local.hour - 6) / 24 * 2 * np.pi)
        comed = pd.DataFrame({"Datetime": local, "COMED_MW": load})

        grid = pd.date_range("2015-01-01", "2015-12-31 23:00", freq="h")
        weather = pd.DataFrame({"time": grid, "temperature_2m": np.zeros(len(grid))})

        comed_shifted = clean.realign_to_fixed_utc5(comed) if realign else None
        naive = join.naive_join(comed, weather)
        if realign:
            return join.join_hourly(comed_shifted, weather), naive
        return naive.copy(), naive

    def test_passes_when_realignment_was_applied(self):
        joined, naive = self._build(realign=True)
        result = validate.check_dst_alignment(joined, naive)

        # Default strategy is "eastern": summer is the corrected season.
        assert result["strategy"] == "eastern"
        assert result["summer_corr_rolled"] > result["summer_corr_unshifted"]
        assert result["winter_corr"] > 0.99

    def test_fails_when_realignment_was_skipped(self):
        """The whole point: an uncorrected join must be caught here."""
        joined, naive = self._build(realign=False)
        with pytest.raises(DataValidationError, match="DST check failed"):
            validate.check_dst_alignment(joined, naive)
