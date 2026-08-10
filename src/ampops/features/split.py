"""Chronological train/test split.

Random splits leak the future into training for time series — the model would
see tomorrow's load while predicting today's. A deployed forecaster only ever
has the past, so the split must respect that. The test tail also stays isolated
until production validation, per the course requirement.
"""

from __future__ import annotations

import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)


def time_split(
    df: pd.DataFrame, test_months: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the final `test_months` of history as the test set.

    Returns (train, test). Guarantees max(train.time) < min(test.time).
    """
    test_months = test_months if test_months is not None else config.TEST_MONTHS
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    cutoff = df[config.TIME_COL].max() - pd.DateOffset(months=test_months)
    train = df[df[config.TIME_COL] <= cutoff].reset_index(drop=True)
    test = df[df[config.TIME_COL] > cutoff].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            f"time_split with test_months={test_months} produced an empty side "
            f"(train={len(train)}, test={len(test)}) — the history is too short."
        )

    logger.info(
        "Split at %s | train %s rows (%s -> %s) | test %s rows (%s -> %s)",
        cutoff,
        len(train),
        train[config.TIME_COL].min(),
        train[config.TIME_COL].max(),
        len(test),
        test[config.TIME_COL].min(),
        test[config.TIME_COL].max(),
    )
    return train, test
