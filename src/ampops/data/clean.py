"""Cleaning steps that make the two sources joinable.

The load-bearing function here is `realign_to_fixed_utc5`. Everything else is
bookkeeping.
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)

CHICAGO = ZoneInfo(config.SOURCE_TZ)


def dedup_comed(df: pd.DataFrame) -> pd.DataFrame:
    """Average the duplicated fall-back-hour readings.

    Four timestamps (2014-11-02 02:00 and the equivalent in 2015-2017) appear
    twice with different MW values — both are real readings for an ambiguous
    clock hour when the clock rolls 02:00 -> 01:00. Averaging keeps the series
    continuous and is more defensible than arbitrarily keeping the first.
    """
    before = len(df)
    out = df.groupby("Datetime", as_index=False)["COMED_MW"].mean()
    logger.info("Deduped COMED: %s -> %s rows (dropped %s)", before, len(out), before - len(out))
    return out


def utc_offset_hours(ts: pd.Timestamp) -> float:
    """Real America/Chicago UTC offset for a naive local timestamp."""
    return ts.to_pydatetime().replace(tzinfo=CHICAGO).utcoffset().total_seconds() / 3600


def realign_to_fixed_utc5(df: pd.DataFrame, strategy: str | None = None) -> pd.DataFrame:
    """Move COMED's DST-aware timestamps onto the weather file's fixed UTC-5 grid.

    Dispatches on `config.TIMEZONE_STRATEGY`. See that constant, and
    docs/timezone_alignment_finding.md, for why there are two strategies and
    which one the evidence favours.
    """
    strategy = strategy or config.TIMEZONE_STRATEGY
    if strategy == "chicago_local":
        return realign_chicago_local(df)
    if strategy == "eastern":
        return realign_eastern(df)
    raise ValueError(
        f"Unknown timezone strategy {strategy!r}; expected 'chicago_local' or 'eastern'"
    )


def realign_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Treat COMED stamps as PJM Eastern Prevailing Time and convert via UTC.

    PJM publishes every zone's series on Eastern time, including ComEd, which
    physically sits in Central. Under this reading the stamps run one hour ahead
    of Chicago wall clock, so summer rows need -1h onto the UTC-5 grid and
    winter rows need none — the mirror image of `realign_chicago_local`.

    `ambiguous=False` resolves the fall-back hour to standard time. It affects
    only the four hours already averaged by `dedup_comed`.
    """
    out = df.copy()
    utc = (
        out["Datetime"]
        .dt.tz_localize(config.COMED_EASTERN_TZ, ambiguous=False, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    out["time"] = utc.dt.tz_localize(None) + pd.Timedelta(
        hours=config.WEATHER_UTC_OFFSET_HOURS
    )

    n_dupes = int(out["time"].duplicated().sum())
    if n_dupes:
        raise ValueError(f"Eastern realignment produced {n_dupes} duplicate timestamps")

    logger.info("Realigned %s rows from Eastern onto the fixed UTC-5 grid", len(out))
    return out.drop(columns=["Datetime"])[["time", "COMED_MW"]]


def realign_chicago_local(df: pd.DataFrame) -> pd.DataFrame:
    """Move DST-aware Chicago local time onto the weather file's fixed UTC-5 grid.

    The weather export is stamped at a constant UTC-5 that never shifts for
    daylight saving (`utc_offset_seconds=-18000`). COMED's `Datetime` is local
    Chicago clock time that *does* observe DST. A naive string-match join
    therefore misaligns the two sources by an hour for roughly seven months of
    every year — silently, with no error and no null.

    The fix: for each row, look up the true Chicago offset at that exact
    timestamp and add the difference. CST rows (offset -6) shift +1h; CDT rows
    (offset -5) are already aligned and shift 0.

    The offset must be resolved per *row*, not per *date*. An earlier version
    flagged `is_dst` once per calendar day (sampled at noon); on the two
    transition days themselves, the hours before the transition instant still
    need the previous offset, so that version produced 8 spurious duplicate
    timestamps. The assertion at the end of this function is what caught it.

    Per-row lookup is safe because Python's `zoneinfo` is fold-aware (PEP 495):
    it resolves the ambiguous and nonexistent clock hours at the transitions
    deterministically (fold=0 by default) rather than raising.

    A quirk of the COMED source worth knowing: on spring-forward days the file
    labels its hours 00, 01, 02, 04, ... — it *keeps* the 02:00 label (which is
    not a real Chicago wall-clock hour) and omits 03:00 instead. That works out
    exactly right here. fold=0 resolves the label 02:00 as CST, so it shifts +1h
    to 03:00 on the UTC-5 grid, which is the true instant (08:00Z) that row
    holds. Because the 03:00 label is absent, nothing collides with it.

    If a source ever supplies *both* 02:00 and 03:00 on a spring-forward day,
    they would both map to 03:00 and this function raises rather than silently
    dropping one.
    """
    out = df.copy()
    offsets = out["Datetime"].map(utc_offset_hours)
    shift = (config.WEATHER_UTC_OFFSET_HOURS - offsets).astype(int)
    out["time"] = out["Datetime"] + pd.to_timedelta(shift, unit="h")

    n_dupes = int(out["time"].duplicated().sum())
    if n_dupes:
        collisions = out.loc[out["time"].duplicated(keep=False), "time"].unique()[:5]
        raise ValueError(
            f"DST realignment produced {n_dupes} duplicate timestamps (e.g. {collisions}). "
            "Either the input contains both the 02:00 and 03:00 labels on a spring-forward "
            "day, or duplicate timestamps survived dedup_comed()."
        )

    logger.info(
        "Realigned to fixed UTC-5: %s rows shifted +1h (CST), %s unchanged (CDT)",
        int((shift == 1).sum()),
        int((shift == 0).sum()),
    )
    return out.drop(columns=["Datetime"])[["time", "COMED_MW"]]


def clean_col(c: str) -> str:
    """Normalize an Open-Meteo column name: strip unit suffix, lower, snake_case."""
    c = re.sub(r"\s*\(.*?\)", "", c)  # e.g. "temperature_2m (°C)" -> "temperature_2m"
    return c.strip().lower().replace(" ", "_")


def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `clean_col` to every weather column except the join key."""
    return df.rename(columns={c: clean_col(c) for c in df.columns if c != "time"})


def trim_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """Trim to the overlap window shared by both sources.

    Weather covers 2010-2019 and COMED covers 2011-01-01 to 2018-08-03; rows
    outside the intersection can never be joined.
    """
    start = start if start is not None else config.WINDOW_START
    end = end if end is not None else config.WINDOW_END
    out = df[(df[time_col] >= start) & (df[time_col] <= end)].reset_index(drop=True)
    logger.info("Trimmed to %s -> %s: %s rows", start, end, len(out))
    return out
