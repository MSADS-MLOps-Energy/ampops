"""Build the clean reference window used for AmpOps drift simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = Path("data/processed/test.parquet")
OUTPUT_PATH = Path("monitoring/drift_baseline.json")
WINDOW_SIZE = 168

MONITORED_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "load_lag_24h",
    "load_lag_168h",
    "load_roll_mean_24h",
    "load_roll_std_24h",
]


def build_feature_baseline(series: pd.Series, bins: int = 10) -> dict:
    values = series.astype(float).to_numpy()

    edges = np.unique(
        np.quantile(values, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        edges = np.array([values.min(), values.max()])

    histogram_edges = edges.copy()
    histogram_edges[0] = -np.inf
    histogram_edges[-1] = np.inf

    counts = np.histogram(values, bins=histogram_edges)[0]
    proportions = counts / counts.sum()

    return {
        "bin_edges": [
            None if np.isinf(x) else float(x)
            for x in histogram_edges
        ],
        "reference_proportions": proportions.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def main() -> None:
    test = pd.read_parquet(REFERENCE_PATH).sort_values("time")
    reference = test.iloc[:WINDOW_SIZE].copy()

    baseline = {
        "reference_dataset": str(REFERENCE_PATH),
        "reference_rows": len(reference),
        "reference_start": str(reference["time"].min()),
        "reference_end": str(reference["time"].max()),
        "psi_warning_threshold": 0.10,
        "psi_alert_threshold": 0.25,
        "features": {},
    }

    for feature in MONITORED_FEATURES:
        baseline["features"][feature] = build_feature_baseline(
            reference[feature]
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(baseline, indent=2))

    print(f"Reference rows: {len(reference)}")
    print(f"Reference start: {reference['time'].min()}")
    print(f"Reference end: {reference['time'].max()}")
    print(f"Monitored features: {len(MONITORED_FEATURES)}")
    print(f"Baseline written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
