"""Tests for the cleaning layer, concentrated on the DST realignment.

The realignment is the one transformation whose failure mode is invisible: a
broken version yields a dataset with the right row count, no nulls, and no
duplicates that is nonetheless misaligned by an hour for seven months a year.
These tests pin the behaviour on the exact days where it is hard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ampops.data import clean


def _frame(timestamps: list[str], load: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(timestamps),
            "COMED_MW": load if load is not None else list(range(len(timestamps))),
        }
    )


class TestDedupComed:
    def test_averages_duplicate_fall_back_hour(self):
        df = _frame(
            ["2014-11-02 01:00", "2014-11-02 02:00", "2014-11-02 02:00", "2014-11-02 03:00"],
            load=[100.0, 200.0, 300.0, 400.0],
        )
        out = clean.dedup_comed(df)

        assert len(out) == 3
        assert out.loc[out["Datetime"] == pd.Timestamp("2014-11-02 02:00"), "COMED_MW"].iloc[
            0
        ] == pytest.approx(250.0)

    def test_leaves_unique_timestamps_untouched(self):
        df = _frame(["2015-06-01 00:00", "2015-06-01 01:00"], load=[10.0, 20.0])
        out = clean.dedup_comed(df)

        assert len(out) == 2
        assert out["COMED_MW"].tolist() == [10.0, 20.0]


class TestRealignChicagoLocal:
    def test_winter_cst_rows_shift_forward_one_hour(self):
        """Chicago is UTC-6 in January; the weather grid is UTC-5, so add 1h."""
        df = _frame(["2015-01-15 08:00", "2015-01-15 09:00"])
        out = clean.realign_chicago_local(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-01-15 09:00"),
            pd.Timestamp("2015-01-15 10:00"),
        ]

    def test_summer_cdt_rows_are_unchanged(self):
        """Chicago is UTC-5 in July, already on the weather grid."""
        df = _frame(["2015-07-15 08:00", "2015-07-15 09:00"])
        out = clean.realign_chicago_local(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-07-15 08:00"),
            pd.Timestamp("2015-07-15 09:00"),
        ]

    def test_spring_forward_day_matches_the_comed_labelling(self):
        """Real COMED labels a spring-forward day 00, 01, 02, 04 — 03:00 is the gap.

        The 02:00 label is not a real Chicago wall-clock hour; fold=0 resolves it
        as CST so it shifts to 03:00, which is the true instant it holds. The
        result is a contiguous, collision-free run on the UTC-5 grid.
        """
        df = _frame(
            ["2015-03-08 00:00", "2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 04:00"]
        )
        out = clean.realign_chicago_local(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-03-08 01:00"),
            pd.Timestamp("2015-03-08 02:00"),
            pd.Timestamp("2015-03-08 03:00"),
            pd.Timestamp("2015-03-08 04:00"),
        ]

    def test_impossible_spring_forward_pair_is_rejected(self):
        """Both 02:00 and 03:00 on a spring-forward day cannot both be real.

        They collide on the UTC-5 grid; the function must raise rather than
        silently drop one.
        """
        df = _frame(["2015-03-08 02:00", "2015-03-08 03:00"])
        with pytest.raises(ValueError, match="spring-forward"):
            clean.realign_chicago_local(df)

    def test_fall_back_day_shifts_per_row_not_per_day(self):
        """On 2015-11-01 the offset changes at 02:00: CDT before, CST after."""
        df = _frame(["2015-11-01 00:00", "2015-11-01 02:00", "2015-11-01 03:00"])
        out = clean.realign_chicago_local(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-11-01 00:00"),  # CDT, unchanged
            pd.Timestamp("2015-11-01 03:00"),  # CST, +1h
            pd.Timestamp("2015-11-01 04:00"),  # CST, +1h
        ]

    @pytest.mark.parametrize(
        ("day", "spring_forward"),
        [
            ("2015-03-08", True),
            ("2016-03-13", True),
            ("2015-11-01", False),
            ("2016-11-06", False),
        ],
    )
    def test_transition_days_produce_no_duplicate_timestamps(self, day, spring_forward):
        """The regression this guards: a per-date is_dst flag created 8 dupes.

        Days are built the way the COMED source actually stores them — the
        03:00 label omitted on spring-forward days (see `dedup_comed` for the
        fall-back side, where the duplicate 02:00 is averaged before this runs).
        """
        hours = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
        if spring_forward:
            hours = hours[hours != pd.Timestamp(f"{day} 03:00")]

        df = pd.DataFrame({"Datetime": hours, "COMED_MW": 1.0})
        out = clean.realign_chicago_local(df)

        assert not out["time"].duplicated().any(), f"duplicates produced on {day}"
        assert out["time"].is_monotonic_increasing

    def test_output_schema_is_the_join_contract(self):
        df = _frame(["2015-01-15 08:00"])
        out = clean.realign_chicago_local(df)

        assert list(out.columns) == ["time", "COMED_MW"]

    def test_raises_when_realignment_would_collide(self):
        """Duplicates entering realignment must not pass silently."""
        df = _frame(["2015-01-15 08:00", "2015-01-15 08:00"])
        with pytest.raises(ValueError, match="duplicate timestamps"):
            clean.realign_chicago_local(df)


class TestColumnNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("temperature_2m (°C)", "temperature_2m"),
            ("relative_humidity_2m (%)", "relative_humidity_2m"),
            ("Wind Speed 10m (km/h)", "wind_speed_10m"),
            ("weather_code (wmo code)", "weather_code"),
        ],
    )
    def test_clean_col(self, raw, expected):
        assert clean.clean_col(raw) == expected

    def test_join_key_is_preserved(self):
        df = pd.DataFrame(columns=["time", "temperature_2m (°C)"])
        assert list(clean.normalize_weather_columns(df).columns) == ["time", "temperature_2m"]


class TestTrimWindow:
    def test_trims_inclusive_of_both_bounds(self):
        df = pd.DataFrame(
            {"time": pd.to_datetime(["2010-06-01", "2012-06-01", "2019-06-01"]), "v": [1, 2, 3]}
        )
        out = clean.trim_window(df)

        assert out["v"].tolist() == [2]


class TestRealignEastern:
    """The default strategy: summer rows shift -1h, winter rows are untouched."""

    def test_summer_rows_shift_back_one_hour(self):
        df = _frame(["2015-07-15 08:00", "2015-07-15 09:00"])
        out = clean.realign_eastern(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-07-15 07:00"),
            pd.Timestamp("2015-07-15 08:00"),
        ]

    def test_winter_rows_are_unchanged(self):
        """EST is UTC-5, identical to the weather grid."""
        df = _frame(["2015-01-15 08:00", "2015-01-15 09:00"])
        out = clean.realign_eastern(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-01-15 08:00"),
            pd.Timestamp("2015-01-15 09:00"),
        ]

    @pytest.mark.parametrize(
        ("day", "spring_forward"),
        [
            ("2015-03-08", True),
            ("2016-03-13", True),
            ("2015-11-01", False),
            ("2016-11-06", False),
        ],
    )
    def test_transition_days_produce_no_duplicate_timestamps(self, day, spring_forward):
        """Days built the way COMED actually stores them: no 03:00 label on
        spring-forward days (the file keeps 02:00 and skips 03:00)."""
        hours = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h")
        if spring_forward:
            hours = hours[hours != pd.Timestamp(f"{day} 03:00")]

        df = pd.DataFrame({"Datetime": hours, "COMED_MW": 1.0})
        out = clean.realign_eastern(df)

        assert not out["time"].duplicated().any(), f"duplicates produced on {day}"
        assert out["time"].is_monotonic_increasing

    def test_output_schema_is_the_join_contract(self):
        out = clean.realign_eastern(_frame(["2015-01-15 08:00"]))

        assert list(out.columns) == ["time", "COMED_MW"]


class TestTimezoneStrategy:
    """Both realignment strategies are wired up and behave as documented.

    See docs/timezone_alignment_finding.md — "eastern" is the adopted default;
    "chicago_local" is kept so the original notebook stays reproducible.
    """

    def test_default_strategy_is_eastern(self):
        """The team adopted the summer -1h correction; see the finding doc."""
        from ampops import config

        assert config.TIMEZONE_STRATEGY == "eastern"

    def test_the_two_strategies_are_mirror_images(self):
        """chicago_local shifts winter +1h; eastern shifts summer -1h."""
        df = _frame(["2015-01-15 08:00", "2015-07-15 08:00"])

        chi = clean.realign_chicago_local(df)["time"].tolist()
        eas = clean.realign_eastern(df)["time"].tolist()

        assert chi == [pd.Timestamp("2015-01-15 09:00"), pd.Timestamp("2015-07-15 08:00")]
        assert eas == [pd.Timestamp("2015-01-15 08:00"), pd.Timestamp("2015-07-15 07:00")]

    def test_dispatcher_matches_the_named_implementation(self):
        df = _frame(["2015-01-15 08:00", "2015-07-15 08:00"])

        pd.testing.assert_frame_equal(
            clean.realign_to_fixed_utc5(df, strategy="chicago_local"),
            clean.realign_chicago_local(df),
        )
        pd.testing.assert_frame_equal(
            clean.realign_to_fixed_utc5(df, strategy="eastern"),
            clean.realign_eastern(df),
        )

    def test_eastern_shifts_summer_back_and_leaves_winter(self):
        """The mirror image of chicago_local: -1h in summer, 0h in winter."""
        df = _frame(["2015-01-15 08:00", "2015-07-15 08:00"])
        out = clean.realign_eastern(df)

        assert out["time"].tolist() == [
            pd.Timestamp("2015-01-15 08:00"),  # EST = UTC-5, already on grid
            pd.Timestamp("2015-07-15 07:00"),  # EDT = UTC-4, -1h onto grid
        ]

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown timezone strategy"):
            clean.realign_to_fixed_utc5(_frame(["2015-01-15 08:00"]), strategy="mountain")
