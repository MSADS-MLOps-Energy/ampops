
"""Statistical feature-drift detection for AmpOps using PSI."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

DEFAULT_BASELINE_PATH = Path("monitoring/drift_baseline.json")


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict:
    return json.loads(path.read_text())


def calculate_psi(series: pd.Series, feature_baseline: dict) -> float:
    values = series.astype(float).to_numpy()

    stored_edges = feature_baseline["bin_edges"]
    edges = np.array(
        [
            -np.inf if i == 0 and value is None
            else np.inf if i == len(stored_edges) - 1 and value is None
            else float(value)
            for i, value in enumerate(stored_edges)
        ]
    )

    current_counts = np.histogram(values, bins=edges)[0]
    current_proportions = current_counts / current_counts.sum()

    reference_proportions = np.array(
        feature_baseline["reference_proportions"],
        dtype=float,
    )

    eps = 1e-6
    ref = np.clip(reference_proportions, eps, None)
    cur = np.clip(current_proportions, eps, None)

    return float(np.sum((cur - ref) * np.log(cur / ref)))


def score_dataframe(
    current: pd.DataFrame,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> dict:
    baseline = load_baseline(baseline_path)

    warning_threshold = baseline["psi_warning_threshold"]
    alert_threshold = baseline["psi_alert_threshold"]

    results = {}

    for feature, feature_baseline in baseline["features"].items():
        if feature not in current.columns:
            results[feature] = {
                "psi": None,
                "status": "missing",
            }
            continue

        psi = calculate_psi(current[feature], feature_baseline)

        if psi >= alert_threshold:
            status = "alert"
        elif psi >= warning_threshold:
            status = "warning"
        else:
            status = "normal"

        results[feature] = {
            "psi": psi,
            "status": status,
        }

    alert_count = sum(
        item["status"] == "alert"
        for item in results.values()
    )
    warning_count = sum(
        item["status"] == "warning"
        for item in results.values()
    )
    missing_count = sum(
        item["status"] == "missing"
        for item in results.values()
    )

    available_psi = [
        item["psi"]
        for item in results.values()
        if item["psi"] is not None
    ]

    return {
        "rows": len(current),
        "alert_count": alert_count,
        "warning_count": warning_count,
        "missing_count": missing_count,
        "drift_detected": alert_count > 0 or missing_count > 0,
        "max_psi": max(available_psi) if available_psi else None,
        "features": results,
    }

class RollingDriftMonitor:
    """Accumulate recent inference features and score a rolling PSI window."""

    def __init__(
        self,
        baseline_path: Path = DEFAULT_BASELINE_PATH,
        window_size: int = 168,
    ) -> None:
        self.baseline_path = baseline_path
        self.baseline = load_baseline(baseline_path)
        self.features = list(self.baseline["features"])
        self.window_size = window_size

        self._rows: deque[dict] = deque(maxlen=window_size)
        self._lock = Lock()

    @property
    def rows_collected(self) -> int:
        with self._lock:
            return len(self._rows)

    def observe(self, frame: pd.DataFrame) -> dict:
        """Add inference rows and score once a full rolling window exists."""

        missing = [
            feature
            for feature in self.features
            if feature not in frame.columns
        ]
        if missing:
            return {
                "ready": False,
                "rows": self.rows_collected,
                "missing_features": missing,
            }

        records = frame[self.features].to_dict(orient="records")

        with self._lock:
            self._rows.extend(records)

            rows_collected = len(self._rows)

            if rows_collected < self.window_size:
                return {
                    "ready": False,
                    "rows": rows_collected,
                    "rows_required": self.window_size,
                    "missing_features": [],
                }

            current = pd.DataFrame(list(self._rows))

        result = score_dataframe(
            current,
            baseline_path=self.baseline_path,
        )

        result["ready"] = True
        result["rows_required"] = self.window_size
        result["missing_features"] = []

        return result
