"""Join the realigned load series to the weather series."""

from __future__ import annotations

import pandas as pd

from ampops.utils.io import get_logger

logger = get_logger(__name__)


def join_hourly(comed: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Inner-join load and weather on the shared fixed-UTC-5 `time` grid.

    Inner is deliberate: an outer join would paper over realignment bugs by
    filling nulls instead of dropping unmatched rows, and the row-count gate in
    validate.py would never see the problem.
    """
    joined = pd.merge(comed, weather, on="time", how="inner").sort_values("time")
    joined = joined.reset_index(drop=True)
    logger.info("Joined: %s rows x %s cols", len(joined), joined.shape[1])
    return joined


def naive_join(comed_local: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Deliberately *wrong* join on uncorrected local timestamps.

    Only used by `validate.check_dst_alignment` as the control case: the
    realigned join must differ from this one by exactly +1h in winter and not
    at all in summer. Never feed this to a model.
    """
    left = comed_local[["Datetime", "COMED_MW"]].rename(columns={"Datetime": "time"})
    return pd.merge(left, weather, on="time", how="inner")
