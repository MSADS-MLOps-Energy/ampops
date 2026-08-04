"""Load the two raw source files.

Both sources have a non-obvious gotcha that this module encapsulates:
  - the Open-Meteo CSV is two concatenated exports in one file
  - the COMED CSV is not stored in chronological order
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def load_weather_hourly(path: str | Path | None = None) -> pd.DataFrame:
    """Load only the hourly block of the Open-Meteo export.

    The file contains a 3-line metadata preamble, then the hourly block, then a
    blank separator, a *second* header row, and a daily-aggregated re-export of
    the same 2010-2019 period. `nrows` is what stops the parse before that
    second block — without it pandas silently ingests the daily rows and every
    downstream dtype turns to object.
    """
    path = Path(path or config.WEATHER_CSV)
    df = pd.read_csv(path, skiprows=config.WEATHER_SKIPROWS, nrows=config.WEATHER_NROWS)
    df["time"] = pd.to_datetime(df["time"])
    logger.info(
        "Weather hourly rows: %s (%s -> %s)", len(df), df["time"].min(), df["time"].max()
    )
    return df


def load_comed(path: str | Path | None = None) -> pd.DataFrame:
    """Load COMED hourly load, sorted chronologically.

    The raw file is stored as per-day blocks in *descending* date order within
    each year. The sort is required for correctness, not cosmetics — every
    downstream lag and rolling feature assumes chronological order.
    """
    path = Path(path or config.COMED_CSV)
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    logger.info(
        "COMED rows: %s (%s -> %s)", len(df), df["Datetime"].min(), df["Datetime"].max()
    )
    return df
