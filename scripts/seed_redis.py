#!/usr/bin/env python
"""Seed the Redis feature store from `data/processed/joined_hourly.parquet`.

The serving API reads its 192-hour window out of Redis, never off disk, so the
store has to be populated before the first request. Layout (serving_contract.md
§4a):

    ampops:load:{grid_id}              hash: epoch-hour -> COMED_MW
    ampops:weather:{grid_id}:{epoch}   hash: the 30 weather columns for that hour

Values are written with `str()` because Redis has no types. That is fine — and
survivable — only because `app.store` re-applies the §1a dtype map on read. If
you add a column here, add it to `app.store.WEATHER_DTYPES` in the same commit
or it will silently arrive as the wrong type at predict time.

    python scripts/seed_redis.py                       # everything
    python scripts/seed_redis.py --since 2018-01-01    # just the recent tail
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# sys.path[0] is scripts/, not the repo root, so both `ampops` (src layout) and
# `app` need adding explicitly before the imports below.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ampops import config  # noqa: E402
from ampops.utils.io import get_logger  # noqa: E402
from app.store import LOAD_KEY, WEATHER_COLUMNS, WEATHER_KEY  # noqa: E402

logger = get_logger(__name__)


def seed(
    df: pd.DataFrame,
    redis_url: str,
    grid_id: str,
    batch_size: int = 2_000,
) -> int:
    """Write every row of `df` into the load and weather hashes. Returns row count."""
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    load_key = LOAD_KEY.format(grid_id=grid_id)

    pipe = client.pipeline()
    written = 0
    for _, row in df.iterrows():
        epoch = int(row[config.TIME_COL].timestamp())
        pipe.hset(load_key, str(epoch), str(row[config.TARGET]))
        pipe.hset(
            WEATHER_KEY.format(grid_id=grid_id, epoch=epoch),
            mapping={col: str(row[col]) for col in WEATHER_COLUMNS},
        )
        written += 1
        # Pipelining is what makes this minutes-vs-hours: one round trip per
        # batch instead of two per hour of history.
        if written % batch_size == 0:
            pipe.execute()
            pipe = client.pipeline()
            logger.info("Seeded %s / %s hours", written, len(df))

    pipe.execute()
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(config.JOINED_PARQUET))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--grid-id", default=os.getenv("AMPOPS_GRID_ID", "COMED"))
    parser.add_argument(
        "--since",
        default=None,
        help="Only seed hours at or after this timestamp (e.g. 2018-01-01)",
    )
    args = parser.parse_args(argv)

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `make pipeline-local`, or `make data-export` after a DAG run."
        )

    df = pd.read_parquet(path).sort_values(config.TIME_COL).reset_index(drop=True)
    if args.since:
        df = df[df[config.TIME_COL] >= pd.Timestamp(args.since)]
    if df.empty:
        raise SystemExit(f"Nothing to seed from {path} (--since={args.since}).")

    logger.info(
        "Seeding %s hours (%s … %s) for grid %s into %s",
        len(df),
        df[config.TIME_COL].min(),
        df[config.TIME_COL].max(),
        args.grid_id,
        args.redis_url,
    )
    written = seed(df, args.redis_url, args.grid_id)
    logger.info("Seeded %s hours. Latest seeded hour: %s", written, df[config.TIME_COL].max())


if __name__ == "__main__":
    main()
