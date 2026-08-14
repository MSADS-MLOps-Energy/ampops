"""Real (non-mocked) test of the H2O AutoML training step.

Guards on `h2o` being importable — this only checks the Python package is
installed, not that a JVM is reachable on PATH (Java itself isn't verified
here; that's a documented, accepted gap, not something this test covers).
Running it locally also requires Java on PATH, e.g.:
    export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h2o", reason="H2O only installed where the AutoML step runs")

from mlflow.tracking import MlflowClient  # noqa: E402

from ampops import config  # noqa: E402
from ampops.features.build import build_features  # noqa: E402
from ampops.training import automl, registry  # noqa: E402


@pytest.fixture
def synthetic_train_path(tmp_path) -> str:
    """A small, fast, feature-built frame shaped like `data/processed/train.parquet`.

    `run_h2o_automl` reads a parquet already run through
    `ampops.features.build.build_features` (calendar + lag/rolling columns
    present), then carves its own internal validation tail via `time_split`.
    That internal split defaults to `VALIDATION_MONTHS` (3 months), so the
    synthetic history needs to comfortably span more than that after the
    warm-up rows (168h rolling window + 24h horizon) are dropped.
    """
    times = pd.date_range("2015-01-01", periods=24 * 150, freq="h")  # ~5 months
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            config.TIME_COL: times,
            config.TARGET: 10_000
            + 500 * np.sin(np.arange(len(times)) / 24)
            + rng.normal(0, 10, len(times)),
            "temperature_2m": rng.normal(5, 3, len(times)),
        }
    )

    features = build_features(raw)
    path = tmp_path / "train.parquet"
    features.to_parquet(path)
    return str(path)


@pytest.fixture
def synthetic_test_path(tmp_path) -> str:
    """A small, later time slice standing in for the sealed test holdout.

    Built the same way as `synthetic_train_path` but from a distinct, later
    date range and a different RNG seed, so it exercises `evaluate_on_test`
    against a genuinely separate frame rather than reusing training rows.
    """
    times = pd.date_range("2015-08-01", periods=24 * 30, freq="h")  # ~1 month
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(
        {
            config.TIME_COL: times,
            config.TARGET: 10_000
            + 500 * np.sin(np.arange(len(times)) / 24)
            + rng.normal(0, 10, len(times)),
            "temperature_2m": rng.normal(5, 3, len(times)),
        }
    )

    features = build_features(raw)
    path = tmp_path / "test.parquet"
    features.to_parquet(path)
    return str(path)


def test_run_h2o_automl_returns_a_valid_champion_scorecard(
    synthetic_train_path, synthetic_test_path, tmp_path, monkeypatch
):
    # Keep the search tiny so the test runs fast — real overrides the function
    # reads live, not hardcoded, per `automl.run_h2o_automl`'s use of `config.*`.
    monkeypatch.setattr(config, "AUTOML_MAX_MODELS", 2)
    monkeypatch.setattr(config, "AUTOML_MAX_RUNTIME_SECS", 30)

    # Isolate MLflow from the Docker `mlflow` service / config.MLFLOW_TRACKING_URI.
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    # Local .venv's mlflow (3.15.1) refuses a bare file-store tracking URI
    # without this env var; Build hit the same thing in their smoke test.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    # config.MLFLOW_REGISTRY_URI defaults to "databricks-uc" independent of
    # tracking_uri (see _configure_mlflow) -- without clearing it here,
    # register_champion below would try to register this unsigned synthetic
    # model against real Unity Catalog instead of the isolated file store.
    # Clear the env var too: mlflow resolves MLFLOW_REGISTRY_URI from the
    # environment on its own whenever our code doesn't explicitly call
    # mlflow.set_registry_uri(), so a caller with .env already sourced (e.g.
    # a real Databricks token exported for other tooling) would otherwise
    # leak through regardless of the config.py patch above.
    monkeypatch.setattr(config, "MLFLOW_REGISTRY_URI", "")
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    # Same leakage risk for the UC prefix: resolve_registered_model_name()
    # would otherwise prefix this plain test name with AMPOPS_UC_MODEL_PREFIX
    # whenever a caller has .env sourced.
    monkeypatch.setattr(config, "UC_MODEL_PREFIX", "")

    result = automl.run_h2o_automl(
        synthetic_train_path,
        tracking_uri=tracking_uri,
        experiment="test-automl",
    )

    # --- shape and validity of the returned dict ---
    expected_keys = {"model_name", "run_id", "mape", "rmse", "mae", "n_train", "n_valid"}
    assert expected_keys <= result.keys()

    assert isinstance(result["model_name"], str) and result["model_name"]
    assert isinstance(result["run_id"], str) and result["run_id"]

    for metric in ("mape", "rmse", "mae"):
        assert isinstance(result[metric], float)
        assert result[metric] >= 0

    assert isinstance(result["n_train"], int) and result["n_train"] > 0
    assert isinstance(result["n_valid"], int) and result["n_valid"] > 0

    # --- the dict is a valid input to register_champion() (stronger check:
    # actually call it against the same isolated tracking URI) ---
    registration = registry.register_champion(
        result, model_name="test-ampops-model", tracking_uri=tracking_uri
    )

    assert registration["registered_model"] == "test-ampops-model"
    assert registration["run_id"] == result["run_id"]
    assert registration["algorithm"] == result["model_name"]
    assert registration["model_uri"] == "models:/test-ampops-model@champion"

    # --- evaluate the already-registered champion on the sealed test holdout,
    # then tag the result onto that same registered version (never before
    # registration, never influencing which model was chosen) ---
    versioned_uri = f"models:/{registration['registered_model']}/{registration['version']}"
    test_metrics = automl.evaluate_on_test(
        versioned_uri, synthetic_test_path, tracking_uri=tracking_uri
    )

    for metric in ("test_mape", "test_rmse", "test_mae"):
        assert isinstance(test_metrics[metric], float)
        assert test_metrics[metric] >= 0
    assert isinstance(test_metrics["n_test"], int) and test_metrics["n_test"] > 0

    tagged = registry.tag_test_metrics(registration, test_metrics, tracking_uri=tracking_uri)
    assert tagged["test_mape"] == test_metrics["test_mape"]

    version_info = MlflowClient(tracking_uri=tracking_uri).get_model_version(
        registration["registered_model"], registration["version"]
    )
    assert "test_mape" in version_info.tags
    assert "test_rmse" in version_info.tags
