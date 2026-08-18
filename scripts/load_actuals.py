#!/usr/bin/env python
"""Load held-out actual demand values into Postgres for forecast monitoring.

This is a historical production-replay utility. The daily forecast DAG stores
predictions in Postgres, while this script loads the corresponding ground truth
from the sealed test set after the forecast has been generated.

Example:
    python scripts/load_actuals.py --date 2018-08-02
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampops import config  # noqa: E402

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS actuals (
    grid_id     text             NOT NULL,
    target_ts   timestamptz      NOT NULL,
    actual_mw   double precision NOT NULL,
    observed_at timestamptz      NOT NULL DEFAULT now(),
    source      text             NOT NULL DEFAULT 'sealed_test_replay',
    PRIMARY KEY (grid_id, target_ts)
);
"""


def run_psql(sql: str, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Execute SQL against the local AmpOps Postgres container."""
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "airflow",
        "-d",
        "ampops",
    ]

    if stdin is None:
        command.extend(["-c", sql])
        return subprocess.run(command, check=True, text=True)

    command.extend(["-c", sql])
    return subprocess.run(command, input=stdin, check=True, text=True)


def load_actuals(
    parquet: Path,
    target_date: str,
    grid_id: str,
) -> int:
    """Load one calendar day of actual demand into Postgres."""
    if not parquet.exists():
        raise SystemExit(
            f"Missing {parquet}. Run the data pipeline before loading actuals."
        )

    df = pd.read_parquet(parquet).sort_values(config.TIME_COL)

    start = pd.Timestamp(target_date)
    end = start + pd.Timedelta(days=1)

    actuals = df.loc[
        (df[config.TIME_COL] >= start) & (df[config.TIME_COL] < end),
        [config.TIME_COL, config.TARGET],
    ].copy()

    if actuals.empty:
        raise SystemExit(f"No actual rows found for {target_date} in {parquet}.")

    if len(actuals) != 24:
        raise SystemExit(
            f"Expected 24 hourly actuals for {target_date}; found {len(actuals)}."
        )

    actuals.insert(0, "grid_id", grid_id)
    actuals["source"] = "sealed_test_replay"

    run_psql(CREATE_TABLE_SQL)

    # Delete only this replay day's rows so rerunning the script is idempotent.
    delete_sql = f"""
    DELETE FROM actuals
    WHERE grid_id = '{grid_id}'
      AND target_ts >= '{start.isoformat()}'
      AND target_ts < '{end.isoformat()}';
    """
    run_psql(delete_sql)

    buffer = StringIO()
    actuals.to_csv(buffer, index=False, header=False)

    copy_sql = (
        r"\copy actuals(grid_id,target_ts,actual_mw,source) "
        r"FROM STDIN WITH (FORMAT csv)"
    )
    run_psql(copy_sql, stdin=buffer.getvalue())

    return len(actuals)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="2018-08-02",
        help="Replay target date to load, YYYY-MM-DD (default: 2018-08-02)",
    )
    parser.add_argument(
        "--parquet",
        default=str(config.TEST_PARQUET),
        help="Path to the sealed test parquet.",
    )
    parser.add_argument(
        "--grid-id",
        default="COMED",
        help="Grid identifier stored with the actuals.",
    )
    args = parser.parse_args(argv)

    written = load_actuals(
        parquet=Path(args.parquet),
        target_date=args.date,
        grid_id=args.grid_id,
    )

    print(f"Loaded {written} actual rows for {args.grid_id} on {args.date}.")

    verification_sql = f"""
    SELECT
        COUNT(*) AS matched_rows,
        ROUND(AVG(ABS(a.actual_mw - f.predicted_mw))::numeric, 2) AS mae_mw,
        ROUND((
            100.0 * AVG(
                ABS(a.actual_mw - f.predicted_mw)
                / NULLIF(ABS(a.actual_mw), 0)
            )
        )::numeric, 2) AS mape_pct
    FROM forecasts f
    JOIN actuals a
      ON f.grid_id = a.grid_id
     AND f.target_ts = a.target_ts
    WHERE f.grid_id = '{args.grid_id}'
      AND f.target_ts >= '{args.date} 00:00:00+00'
      AND f.target_ts < '{args.date} 00:00:00+00'::timestamptz
                            + interval '1 day';
    """
    run_psql(verification_sql)


if __name__ == "__main__":
    main()
