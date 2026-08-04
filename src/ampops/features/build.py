"""Feature engineering for day-ahead load forecasting.

**The leakage rule:** predictions are made HORIZON_HOURS (24h) ahead, so no
feature at time t may reference `COMED_MW` newer than t-24. That is why there
is no t-1 lag here, and why every rolling window is shifted by the horizon
before it is computed. `tests/test_features.py` enforces this mechanically.

Two subtleties worth knowing before editing this file:

1. **Calendar features use Chicago local time, not the index.** The joined
   dataset lives on a fixed UTC-5 grid (see data/clean.py). Chicago local time
   equals UTC-5 in summer but UTC-6 in winter, so a naive `time.dt.hour` is off
   by one for half the year. Human demand behavior follows the wall clock, so
   we convert back to local time for the calendar features.

2. **Lags are computed on a gap-free grid.** The joined data has ~11 missing
   timestamps (a window-start edge effect plus fall-back-hour shortfalls). A
   plain `.shift(24)` counts *rows*, not hours, so it would silently pull the
   wrong observation across every gap. We reindex onto a complete hourly range
   first, shift there, then drop back to the originally-present rows. No target
   values are imputed.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def _cyclical(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    """Sin/cos encoding so the model sees hour 23 and hour 0 as adjacent."""
    radians = 2 * np.pi * values / period
    return pd.DataFrame({f"{name}_sin": np.sin(radians), f"{name}_cos": np.cos(radians)})


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour/day/month/weekend/holiday features, derived from Chicago local time."""
    out = df.copy()

    # UTC-5 fixed grid -> true Chicago wall clock (see module docstring).
    local = (
        out[config.TIME_COL]
        .dt.tz_localize(f"Etc/GMT+{-config.WEATHER_UTC_OFFSET_HOURS}")
        .dt.tz_convert(config.SOURCE_TZ)
        .dt.tz_localize(None)
    )

    out["hour"] = local.dt.hour
    out["dayofweek"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["dayofyear"] = local.dt.dayofyear
    out["year"] = local.dt.year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    years = range(int(local.dt.year.min()), int(local.dt.year.max()) + 1)
    us_holidays = holidays.country_holidays(
        config.HOLIDAY_COUNTRY, subdiv=config.HOLIDAY_SUBDIV, years=years
    )
    out["is_holiday"] = local.dt.date.map(lambda d: d in us_holidays).astype(int)

    out = pd.concat(
        [
            out,
            _cyclical(out["hour"], 24, "hour"),
            _cyclical(out["dayofweek"], 7, "dow"),
            _cyclical(out["dayofyear"], 365.25, "doy"),
        ],
        axis=1,
    )
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling-mean features on the target, all horizon-safe.

    Computed on a complete hourly index so a shift of N genuinely means N hours
    ago even across the gaps in the source data.
    """
    out = df.copy().sort_values(config.TIME_COL).reset_index(drop=True)

    original_times = out[config.TIME_COL]
    full_index = pd.date_range(original_times.min(), original_times.max(), freq="h")
    gapped = out.set_index(config.TIME_COL).reindex(full_index)
    target = gapped[config.TARGET]

    engineered = {}
    for lag in config.LOAD_LAGS:
        engineered[f"load_lag_{lag}h"] = target.shift(lag)

    # Shift by the horizon *first*, then roll: guarantees no window can see
    # load newer than t-HORIZON_HOURS.
    horizon_safe = target.shift(config.HORIZON_HOURS)
    for window in config.LOAD_ROLLING_WINDOWS:
        engineered[f"load_roll_mean_{window}h"] = horizon_safe.rolling(window).mean()
        engineered[f"load_roll_std_{window}h"] = horizon_safe.rolling(window).std()

    features = pd.DataFrame(engineered, index=full_index)

    # Back to only the timestamps that actually existed in the source.
    features = features.loc[original_times]
    out = pd.concat([out, features.reset_index(drop=True)], axis=1)

    logger.info("Added %s lag/rolling features", len(engineered))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: calendar + lags, then drop the un-lagged warm-up rows."""
    out = add_calendar_features(df)
    out = add_lag_features(out)

    before = len(out)
    lag_cols = [c for c in out.columns if c.startswith("load_")]
    out = out.dropna(subset=lag_cols).reset_index(drop=True)
    logger.info(
        "Dropped %s warm-up rows lacking full lag history; %s rows x %s cols remain",
        before - len(out),
        len(out),
        out.shape[1],
    )
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the timestamp and the target."""
    return [c for c in df.columns if c not in (config.TIME_COL, config.TARGET)]
