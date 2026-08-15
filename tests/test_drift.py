"""Tests for AmpOps PSI drift monitoring."""

from __future__ import annotations

import json

import pandas as pd

from monitoring.drift import RollingDriftMonitor, score_dataframe


def _write_test_baseline(tmp_path):
    """Create a tiny deterministic PSI baseline for unit tests."""
    baseline = {
        "reference_rows": 4,
        "psi_warning_threshold": 0.10,
        "psi_alert_threshold": 0.25,
        "features": {
            "temperature_2m": {
                "bin_edges": [None, 5.0, None],
                "reference_proportions": [0.5, 0.5],
                "mean": 4.0,
                "std": 3.0,
                "min": 1.0,
                "max": 7.0,
                "n": 4,
            },
            "relative_humidity_2m": {
                "bin_edges": [None, 50.0, None],
                "reference_proportions": [0.5, 0.5],
                "mean": 50.0,
                "std": 20.0,
                "min": 30.0,
                "max": 70.0,
                "n": 4,
            },
        },
    }

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    return path


def _clean_frame():
    return pd.DataFrame(
        {
            "temperature_2m": [1.0, 2.0, 6.0, 7.0],
            "relative_humidity_2m": [30.0, 40.0, 60.0, 70.0],
        }
    )


def test_clean_distribution_does_not_alert(tmp_path):
    baseline_path = _write_test_baseline(tmp_path)

    result = score_dataframe(
        _clean_frame(),
        baseline_path=baseline_path,
    )

    assert result["drift_detected"] is False
    assert result["alert_count"] == 0
    assert result["warning_count"] == 0
    assert result["max_psi"] == 0.0


def test_shifted_temperature_triggers_alert(tmp_path):
    baseline_path = _write_test_baseline(tmp_path)

    drifted = _clean_frame()
    drifted["temperature_2m"] += 25.0

    result = score_dataframe(
        drifted,
        baseline_path=baseline_path,
    )

    assert result["drift_detected"] is True
    assert result["alert_count"] == 1
    assert result["features"]["temperature_2m"]["status"] == "alert"
    assert result["features"]["relative_humidity_2m"]["status"] == "normal"


def test_rolling_monitor_waits_for_full_window(tmp_path):
    baseline_path = _write_test_baseline(tmp_path)

    monitor = RollingDriftMonitor(
        baseline_path=baseline_path,
        window_size=4,
    )

    partial = monitor.observe(_clean_frame().iloc[:2])

    assert partial["ready"] is False
    assert partial["rows"] == 2
    assert partial["rows_required"] == 4

    complete = monitor.observe(_clean_frame().iloc[2:])

    assert complete["ready"] is True
    assert complete["rows"] == 4
    assert complete["drift_detected"] is False


def test_out_of_bounds_humidity_triggers_alert(tmp_path):
    baseline_path = _write_test_baseline(tmp_path)

    drifted = _clean_frame()
    drifted["relative_humidity_2m"] = 150.0

    result = score_dataframe(
        drifted,
        baseline_path=baseline_path,
    )

    assert result["drift_detected"] is True
    assert result["alert_count"] == 1
    assert result["features"]["relative_humidity_2m"]["status"] == "alert"
    assert result["features"]["temperature_2m"]["status"] == "normal"
