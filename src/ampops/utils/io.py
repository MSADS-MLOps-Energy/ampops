"""Parquet read/write helpers and a shared logger.

Airflow tasks pass file *paths* through XCom rather than DataFrames — XCom is
backed by the metadata DB and is not meant to carry 66k-row frames. These
helpers keep the read/write/mkdir dance in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Logger that plays nicely with Airflow's own handlers.

    Under Airflow the task logger already has handlers attached, so we only
    configure a fallback handler when running standalone (pytest, notebooks).
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def write_parquet(df: pd.DataFrame, path: str | Path) -> str:
    """Write a frame to parquet, creating parent dirs. Returns the path as str.

    Returning a plain str (not Path) keeps the value JSON-serializable for XCom.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    get_logger(__name__).info("Wrote %s rows x %s cols -> %s", len(df), df.shape[1], path)
    return str(path)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file written by an upstream task."""
    df = pd.read_parquet(path)
    get_logger(__name__).info("Read %s rows x %s cols <- %s", len(df), df.shape[1], path)
    return df
