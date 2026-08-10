"""Data quality gates.

Every function here raises `DataValidationError` on failure. That is the whole
point: an Airflow task that raises turns red and halts the DAG, so bad data
cannot reach training. Silently-wrong data is the failure mode this project
exists to demonstrate catching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset fails a quality gate."""


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label}: missing expected column(s) {missing}")


def validate_raw_comed(df: pd.DataFrame) -> None:
    """Gate the raw COMED load series before any transformation."""
    _require_columns(df, ["Datetime", "COMED_MW"], "raw COMED")
    if df.empty:
        raise DataValidationError("raw COMED: file is empty")
    if df["COMED_MW"].isna().any():
        raise DataValidationError(
            f"raw COMED: {int(df['COMED_MW'].isna().sum())} null load readings"
        )
    lo, hi = df["COMED_MW"].min(), df["COMED_MW"].max()
    if lo < config.LOAD_MIN_MW or hi > config.LOAD_MAX_MW:
        raise DataValidationError(
            f"raw COMED: load range [{lo:.0f}, {hi:.0f}] MW falls outside the plausible "
            f"band [{config.LOAD_MIN_MW:.0f}, {config.LOAD_MAX_MW:.0f}] — likely a meter fault"
        )
    if not df["Datetime"].is_monotonic_increasing:
        raise DataValidationError("raw COMED: not sorted chronologically after ingest")
    logger.info("raw COMED passed: %s rows, load %.0f-%.0f MW", len(df), lo, hi)


def validate_raw_weather(df: pd.DataFrame) -> None:
    """Gate the raw hourly weather block before any transformation."""
    _require_columns(df, ["time"], "raw weather")
    if df.empty:
        raise DataValidationError("raw weather: file is empty")
    if df["time"].duplicated().any():
        raise DataValidationError(
            f"raw weather: {int(df['time'].duplicated().sum())} duplicate timestamps — the "
            "daily-aggregate block was probably read in alongside the hourly one"
        )
    if not df["time"].is_monotonic_increasing:
        raise DataValidationError("raw weather: timestamps are not monotonic")
    gaps = df["time"].diff().dropna().unique()
    if len(gaps) != 1 or gaps[0] != np.timedelta64(1, "h"):
        raise DataValidationError(
            f"raw weather: expected a strictly hourly grid, found spacings {gaps}"
        )
    logger.info("raw weather passed: %s rows on a strict hourly grid", len(df))


def validate_joined(df: pd.DataFrame) -> None:
    """Gate the joined dataset — the contract every downstream workstream reads."""
    _require_columns(df, [config.TIME_COL, config.TARGET], "joined")

    nulls = df.isna().sum()
    if nulls.any():
        offenders = nulls[nulls > 0].to_dict()
        raise DataValidationError(f"joined: null values present in {offenders}")

    if df[config.TIME_COL].duplicated().any():
        raise DataValidationError(
            f"joined: {int(df[config.TIME_COL].duplicated().sum())} duplicate timestamps"
        )

    expected = config.EXPECTED_JOINED_ROWS
    tolerance = expected * config.ROW_COUNT_TOLERANCE
    if abs(len(df) - expected) > tolerance:
        raise DataValidationError(
            f"joined: {len(df)} rows is outside {config.ROW_COUNT_TOLERANCE:.0%} of the "
            f"expected {expected} — the port or the source data changed"
        )

    lo, hi = df[config.TARGET].min(), df[config.TARGET].max()
    if lo < config.LOAD_MIN_MW or hi > config.LOAD_MAX_MW:
        raise DataValidationError(f"joined: load range [{lo:.0f}, {hi:.0f}] MW is implausible")

    logger.info("joined passed: %s rows x %s cols, no nulls, no dupes", len(df), df.shape[1])


def hourly_profile(df: pd.DataFrame, months: list[int], col: str = config.TARGET) -> pd.Series:
    """Mean value by hour-of-day, restricted to a set of months."""
    sub = df[df[config.TIME_COL].dt.month.isin(months)]
    return sub.assign(hour=sub[config.TIME_COL].dt.hour).groupby("hour")[col].mean()


def check_dst_alignment(
    joined: pd.DataFrame, naive: pd.DataFrame, strategy: str | None = None
) -> dict[str, float]:
    """Prove the DST realignment actually did what it claims.

    The realignment is invisible in row counts and null counts — a broken
    version produces a dataset that looks perfectly clean and trains a
    perfectly plausible, quietly wrong model. This gate is what catches that.

    Method: compare hour-of-day load profiles between the realigned join and
    the naive (uncorrected) one. The season that was corrected must match the
    naive profile *rolled by the shift*; the season left alone must match it
    unshifted.

    Which season is which depends on the strategy (see config.TIMEZONE_STRATEGY):
      - "eastern": summer rows shifted -1h, winter untouched
      - "chicago_local": winter rows shifted +1h, summer untouched

    Note what this does and does not prove. It confirms the transformation ran
    as intended — it cannot tell you the intent was right, because both
    strategies pass their own version of this check. The external evidence for
    choosing between them is in docs/timezone_alignment_finding.md.
    """
    from ampops import config

    strategy = strategy or config.TIMEZONE_STRATEGY
    winter, summer = [12, 1, 2], [6, 7, 8]

    if strategy == "eastern":
        shifted_months, shifted_label, roll = summer, "summer", -1
        untouched_months, untouched_label = winter, "winter"
    else:
        shifted_months, shifted_label, roll = winter, "winter", 1
        untouched_months, untouched_label = summer, "summer"

    naive_shift = hourly_profile(naive, shifted_months)
    real_shift = hourly_profile(joined, shifted_months)
    naive_keep = hourly_profile(naive, untouched_months)
    real_keep = hourly_profile(joined, untouched_months)

    unshifted = float(np.corrcoef(real_shift.values, naive_shift.values)[0, 1])
    rolled = float(np.corrcoef(real_shift.values, np.roll(naive_shift.values, roll))[0, 1])
    untouched_corr = float(np.corrcoef(real_keep.values, naive_keep.values)[0, 1])

    if rolled <= unshifted:
        raise DataValidationError(
            f"DST check failed ({strategy}): {shifted_label} profile correlates "
            f"{unshifted:.3f} unshifted vs {rolled:.3f} rolled {roll:+d}h. The expected "
            "one-hour correction is absent — the two sources are misaligned."
        )
    if untouched_corr <= 0.99:
        raise DataValidationError(
            f"DST check failed ({strategy}): {untouched_label} profile correlation is "
            f"{untouched_corr:.3f}, expected ~1.0. Those rows are already aligned and "
            "must not have been shifted."
        )

    logger.info(
        "DST check passed (%s) | %s unshifted=%.3f rolled%+d h=%.3f | %s=%.3f",
        strategy,
        shifted_label,
        unshifted,
        roll,
        rolled,
        untouched_label,
        untouched_corr,
    )
    return {
        "strategy": strategy,
        f"{shifted_label}_corr_unshifted": unshifted,
        f"{shifted_label}_corr_rolled": rolled,
        f"{untouched_label}_corr": untouched_corr,
    }
