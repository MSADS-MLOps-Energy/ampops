"""Unit tests for shared training helpers (no live MLflow / H2O cluster)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ampops.training.bakeoff import evaluate


def test_evaluate_returns_headline_metrics():
    y = pd.Series([100.0, 200.0, 300.0])
    pred = np.array([110.0, 180.0, 330.0])
    metrics = evaluate(y, pred)
    assert set(metrics) == {"mape", "rmse", "mae"}
    assert metrics["mape"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_extract_leader_hyperparams_maps_h2o_aliases():
    pytest.importorskip("h2o", reason="hyperparam helper lives in automl module")
    from ampops.training.automl import extract_leader_hyperparams

    leader = SimpleNamespace(
        algo="gbm",
        model_id="GBM_1",
        actual_params={
            "ntrees": 120,
            "learn_rate": 0.08,
            "sample_rate": 0.75,
            "col_sample_rate": 0.9,
            "max_depth": 6,
            "min_rows": 10,
            "ignored_columns": ["a", "b"],
        },
    )
    payload = extract_leader_hyperparams(leader)
    assert payload["aliases"]["n_estimators"] == 120
    assert payload["aliases"]["learning_rate"] == 0.08
    assert payload["aliases"]["subsample"] == 0.75
    assert payload["aliases"]["colsample_bytree"] == 0.9
    assert payload["aliases"]["max_depth"] == 6
    assert payload["params"]["ntrees"] == 120
