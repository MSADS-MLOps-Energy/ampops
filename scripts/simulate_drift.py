"""Apply or restore controlled feature drift in the Redis feature store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import redis

TEST_PATH = Path("data/processed/test.parquet")
BACKUP_PATH = Path("monitoring/drift_backup.json")

WINDOW_SIZE = 168
GRID_ID = "COMED"

SCENARIOS = {
    "temperature": {
        "field": "temperature_2m",
        "description": "+25 C temperature shift",
        "mode": "shift",
        "value": 25.0,
    },
    "humidity": {
        "field": "relative_humidity_2m",
        "description": "out-of-bounds humidity set to 150%",
        "mode": "set",
        "value": 150.0,
    },
}


def get_reference_window() -> pd.DataFrame:
    test = pd.read_parquet(TEST_PATH).sort_values("time")
    return test.iloc[:WINDOW_SIZE].copy()


def apply_drift(client: redis.Redis, scenario_name: str) -> None:
    if BACKUP_PATH.exists():
        raise RuntimeError(
            f"{BACKUP_PATH} already exists. "
            "Restore the previous simulation before applying another."
        )

    scenario = SCENARIOS[scenario_name]
    field = scenario["field"]
    window = get_reference_window()

    records = []

    # Collect all original values before modifying Redis.
    for ts in pd.to_datetime(window["time"]):
        epoch = int(pd.Timestamp(ts).timestamp())
        key = f"ampops:weather:{GRID_ID}:{epoch}"

        if not client.exists(key):
            raise RuntimeError(f"Missing Redis key: {key}")

        original = client.hget(key, field)
        if original is None:
            raise RuntimeError(f"Missing {field} in {key}")

        records.append(
            {
                "key": key,
                "timestamp": str(ts),
                "original_value": original,
            }
        )

    backup = {
        "scenario": scenario_name,
        "field": field,
        "records": records,
    }

    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_PATH.write_text(json.dumps(backup, indent=2))

    for record in records:
        original = float(record["original_value"])

        if scenario["mode"] == "shift":
            drifted = original + scenario["value"]
        else:
            drifted = scenario["value"]

        client.hset(record["key"], field, drifted)

    print(f"APPLIED: {scenario_name}")
    print(f"Feature: {field}")
    print(f"Simulation: {scenario['description']}")
    print(f"Redis records changed: {len(records)}")
    print(f"Backup written to: {BACKUP_PATH}")


def restore_drift(client: redis.Redis) -> None:
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(f"No drift backup found at {BACKUP_PATH}")

    backup = json.loads(BACKUP_PATH.read_text())
    field = backup["field"]

    for record in backup["records"]:
        client.hset(
            record["key"],
            field,
            record["original_value"],
        )

    restored = len(backup["records"])
    scenario = backup["scenario"]

    BACKUP_PATH.unlink()

    print(f"RESTORED: {restored} Redis records")
    print(f"Scenario restored: {scenario}")
    print(f"Feature restored: {field}")
    print("Temporary backup removed")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--apply",
        choices=SCENARIOS,
        metavar="{temperature,humidity}",
        help="Apply a controlled drift scenario.",
    )
    group.add_argument(
        "--restore",
        action="store_true",
        help="Restore original Redis values.",
    )

    args = parser.parse_args()

    client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True,
    )

    if args.apply:
        apply_drift(client, args.apply)
    else:
        restore_drift(client)


if __name__ == "__main__":
    main()
