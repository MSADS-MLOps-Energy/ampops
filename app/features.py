"""Assemble the 49-column inference frame — by delegation, never by arithmetic.

**This module computes no features.** It fetches a window from the feature
store, blanks the target at the timestamps being predicted, and hands the frame
straight to `ampops.features.build`'s `add_calendar_features` /
`add_lag_features` / `feature_columns` — the exact functions that produced
`train.parquet`. That is the whole anti-skew design (serving_contract.md §2):
if a feature definition changes upstream, serving changes with it, and there is
no second implementation to forget to update. Reimplementing so much as an
`hour` column here would reintroduce the skew this design exists to prevent.

Two rules that look like bugs and are not:

1. `build_features()` is deliberately *not* called. Its trailing
   `dropna(subset=[...load_*])` would discard exactly the row being predicted,
   because that row's target is blank by construction.
2. `COMED_MW` is `NaN` at every target timestamp, on purpose. Every feature
   references `t-24` or older (the horizon rule in `ampops.features.build`), so
   nothing reads it — but the row must still occupy its slot in the hourly grid
   or the lag/rolling shifts would land one hour off.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ampops import config
from ampops.features.build import add_calendar_features, add_lag_features, feature_columns
from app.config import HISTORY_HOURS
from app.store import FeatureStore


def build_inference_frame(
    store: FeatureStore,
    grid_id: str,
    timestamps: Sequence[pd.Timestamp],
    history_hours: int = HISTORY_HOURS,
) -> pd.DataFrame:
    """The model's 49 input columns for `timestamps`, in the requested order.

    One window and one feature pass for the whole batch — looping per timestamp
    would recompute 192 hours of lags N times over. Raises
    `InsufficientHistoryError` (via the store) when the window cannot be
    covered.
    """
    targets = [pd.Timestamp(ts) for ts in timestamps]
    window = store.get_window(grid_id, targets, history_hours)
    window = window.sort_values(config.TIME_COL).reset_index(drop=True)

    # Blank the target at the hours being predicted. Cast afterwards: assigning
    # into a float column can flip its dtype to object, which would break the
    # shifts inside add_lag_features.
    window.loc[window[config.TIME_COL].isin(targets), config.TARGET] = np.nan
    window[config.TARGET] = window[config.TARGET].astype("float64")

    frame = add_lag_features(add_calendar_features(window))

    rows = frame.set_index(config.TIME_COL).loc[targets]
    return rows[feature_columns(frame)].reset_index(drop=True)
