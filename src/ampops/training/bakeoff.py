"""Shared training metrics.

The bake-off that used to live here (train each of linear/random-forest/
xgboost by hand, log each to MLflow) has been replaced by H2O AutoML — see
`ampops.training.automl`. This module now only provides `evaluate()`, the
MAPE/RMSE/MAE scorer both the old bake-off and the new AutoML path use, plus
the `VALIDATION_MONTHS` constant that defines the internal fit/validation
carve-out (still reused by `ampops.training.automl`).

**Model selection never touches the test set.** Candidates are scored on a
validation tail carved out of the *training* data; `data/processed/test.parquet`
stays sealed through the entire search and registration. Once a champion is
registered, `ampops.training.automl.evaluate_on_test` reloads that exact
version and scores it against `test.parquet` for reporting —
`ampops.training.registry.tag_test_metrics` then tags the result onto the
already-registered model version. That evaluation happens strictly after
selection and promotion are final; it never feeds back into which model wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Months of the training tail reserved for scoring candidates against each other.
VALIDATION_MONTHS = 3


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAPE (headline) and RMSE (secondary — penalizes the demand spikes that hurt)."""
    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
