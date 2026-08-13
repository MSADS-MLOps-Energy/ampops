"""Integration tests for the FastAPI serving layer, against a real H2O model.

These exercise the actual HTTP surface — routing, request/response shapes,
error codes, and the cache write-through path — with a real (but
deliberately tiny) H2O model loaded via the exact `mlflow.h2o.load_model`
path docs/serving_contract.md §3 requires. No mocking: the app under test is
a real FastAPI app with a real lifespan-loaded model, a real Parquet-backed
feature store, and a real in-process forecast cache.

The one expensive step — training the model — happens in a single
session-scoped fixture reused by every test that needs a loaded model,
mirroring tests/test_automl.py. `/health` and the 503 `/ready` path are
tested through a separate, cheap fixture that never trains anything, so the
module stays fast even when a caller only wants that subset.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("h2o", reason="H2O only installed where the serving API runs")

from fastapi.testclient import TestClient  # noqa: E402

from ampops import config  # noqa: E402
from ampops.features.build import build_features  # noqa: E402
from ampops.training import automl, registry  # noqa: E402

if not config.JOINED_PARQUET.exists():
    pytest.skip(
        "data/processed/joined_hourly.parquet is a gitignored data artifact "
        "and is not present in this environment",
        allow_module_level=True,
    )

from app.main import create_app  # noqa: E402

# Single-grid dataset; see test_serving_features.py for why "COMED" is used.
GRID_ID = "COMED"


@pytest.fixture(scope="session")
def session_env() -> pytest.MonkeyPatch:
    """A `MonkeyPatch` that lives for the whole test session.

    Plain `monkeypatch` is function-scoped and cannot be requested from a
    session-scoped fixture; `pytest.MonkeyPatch()` used directly (undone
    manually at teardown) is the documented workaround.
    """
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def champion(tmp_path_factory, session_env: pytest.MonkeyPatch) -> dict[str, str]:
    """Train a deliberately tiny H2O model once and register it locally.

    Mirrors tests/test_automl.py's real (non-mocked) training call, with the
    search budget shrunk to `AUTOML_MAX_MODELS=1` / `AUTOML_MAX_RUNTIME_SECS=20`
    per serving_contract.md §7's cost controls, and logs to a throwaway local
    MLflow file store rather than Databricks.
    """
    tmp_path = tmp_path_factory.mktemp("serving-api-model")
    tracking_uri = f"file:{tmp_path / 'mlruns'}"

    session_env.setattr(config, "AUTOML_MAX_MODELS", 1)
    session_env.setattr(config, "AUTOML_MAX_RUNTIME_SECS", 20)
    # The module default (MLFLOW_REGISTRY_URI="databricks-uc") makes
    # registration soft-fail outside a real Databricks workspace; a local
    # file-store tracking URI has its own built-in registry.
    session_env.setattr(config, "MLFLOW_REGISTRY_URI", "")
    session_env.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    joined = pd.read_parquet(config.JOINED_PARQUET)
    train_df = build_features(joined)
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path)

    result = automl.run_h2o_automl(
        str(train_path), tracking_uri=tracking_uri, experiment="serving-api-tests"
    )
    registration = registry.register_champion(
        result, model_name="ampops-serving-test-model", tracking_uri=tracking_uri
    )
    assert not registration["skipped"], registration.get("registry_error")

    # Point the serving app's own env-driven settings at this model for the
    # rest of the session — `create_app()` below reads these fresh.
    session_env.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    session_env.setenv("AMPOPS_SERVING_MODEL_URI", registration["model_uri"])
    session_env.setenv("AMPOPS_PARQUET_PATH", str(config.JOINED_PARQUET))
    session_env.setenv("AMPOPS_GRID_ID", GRID_ID)

    return {"model_uri": registration["model_uri"], "tracking_uri": tracking_uri}


@pytest.fixture(scope="session")
def model_client(champion: dict[str, str]) -> TestClient:
    """A `TestClient` whose lifespan has loaded the trivial champion above.

    Session-scoped and reused across every test below that needs a working
    model, per serving_contract.md §7's cost-control requirement.
    """
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def unloaded_client(tmp_path_factory) -> TestClient:
    """A `TestClient` pointed at a model URI that cannot possibly resolve.

    Deliberately independent of `champion`/`model_client` — no AutoML search
    — so `/health` and the 503 `/ready` path stay cheap to exercise even in
    isolation, per the assignment's cost-control note.
    """
    tmp_path = tmp_path_factory.mktemp("serving-api-no-model")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
        mp.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        mp.setenv(
            "AMPOPS_SERVING_MODEL_URI", "models:/ampops-serving-test-model-missing@champion"
        )
        mp.setenv("AMPOPS_PARQUET_PATH", str(config.JOINED_PARQUET))
        mp.setenv("AMPOPS_GRID_ID", GRID_ID)

        app = create_app()
        with TestClient(app) as client:
            yield client


class TestHealthAndReadiness:
    def test_health_returns_200_even_when_no_model_is_loaded(self, unloaded_client):
        response = unloaded_client.get("/health")

        assert response.status_code == 200

    def test_ready_returns_503_when_the_model_failed_to_load(self, unloaded_client):
        response = unloaded_client.get("/ready")

        assert response.status_code == 503

    def test_ready_returns_200_once_the_champion_is_loaded(self, model_client):
        response = model_client.get("/ready")

        assert response.status_code == 200

    def test_health_and_ready_genuinely_differ_when_the_model_is_missing(self, unloaded_client):
        """serving_contract.md §5: '/ready failing while /health passes is the
        diagnostic signal that the champion did not load.'
        """
        assert unloaded_client.get("/health").status_code == 200
        assert unloaded_client.get("/ready").status_code == 503


class TestPredict:
    def test_predict_returns_the_documented_response_shape(self, model_client):
        response = model_client.post(
            "/predict", json={"grid_id": GRID_ID, "timestamp": "2015-07-15T14:00:00"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["grid_id"] == GRID_ID
        assert body["timestamp"] == "2015-07-15T14:00:00"
        assert isinstance(body["predicted_mw"], float)
        assert body["model_version"]
        assert body["source"] in {"cache", "live"}
        assert isinstance(body["latency_ms"], float)

    def test_a_fresh_timestamp_with_nothing_cached_reports_source_live(self, model_client):
        response = model_client.post(
            "/predict", json={"grid_id": GRID_ID, "timestamp": "2015-09-01T03:00:00"}
        )

        assert response.status_code == 200
        assert response.json()["source"] == "live"

    def test_a_timestamp_previously_written_by_batch_reports_source_cache(self, model_client):
        ts = "2015-09-02T00:00:00"
        model_client.post("/predict/batch", json={"grid_id": GRID_ID, "timestamps": [ts]})

        response = model_client.post("/predict", json={"grid_id": GRID_ID, "timestamp": ts})

        assert response.status_code == 200
        assert response.json()["source"] == "cache"

    def test_a_live_miss_does_not_populate_the_cache(self, model_client):
        """serving_contract.md §5: only `/predict/batch` writes through — a
        live `/predict` miss must stay ephemeral.
        """
        ts = "2015-09-03T06:00:00"
        first = model_client.post("/predict", json={"grid_id": GRID_ID, "timestamp": ts})
        assert first.json()["source"] == "live"

        second = model_client.post("/predict", json={"grid_id": GRID_ID, "timestamp": ts})
        assert second.json()["source"] == "live"

    def test_an_out_of_range_timestamp_returns_422(self, model_client):
        response = model_client.post(
            "/predict", json={"grid_id": GRID_ID, "timestamp": "2011-01-01T02:00:00"}
        )

        assert response.status_code == 422


class TestPredictBatch:
    def test_predict_batch_returns_one_prediction_per_requested_timestamp(self, model_client):
        timestamps = [
            ts.strftime("%Y-%m-%dT%H:%M:%S")
            for ts in pd.date_range("2015-10-01T00:00:00", periods=24, freq="h")
        ]

        response = model_client.post(
            "/predict/batch", json={"grid_id": GRID_ID, "timestamps": timestamps}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["grid_id"] == GRID_ID
        assert body["model_version"]
        assert [p["timestamp"] for p in body["predictions"]] == timestamps
        assert all(isinstance(p["predicted_mw"], float) for p in body["predictions"])

    def test_predict_batch_writes_through_to_the_forecast_cache(self, model_client):
        ts = "2015-10-05T12:00:00"
        model_client.post("/predict/batch", json={"grid_id": GRID_ID, "timestamps": [ts]})

        cached = model_client.post("/predict", json={"grid_id": GRID_ID, "timestamp": ts})

        assert cached.json()["source"] == "cache"


class TestForecast:
    def test_forecast_returns_404_when_nothing_is_cached_for_that_date(self, model_client):
        response = model_client.get(
            "/forecast", params={"grid_id": GRID_ID, "date": "2016-02-02"}
        )

        assert response.status_code == 404

    def test_forecast_returns_every_hour_written_by_a_prior_batch_call(self, model_client):
        # Must be a day whose own 24 hours AND whose full 192h history are all
        # present in joined_hourly.parquet. Ten timestamps are genuinely absent
        # from the source (all DST fall-back artifacts -- see CLAUDE.md and
        # AmpOps_Project_Context.md 2.3), so a fall-back date such as
        # 2015-11-01 makes /predict/batch 422 for the whole day, and a date one
        # week later still reads the gap through its own load_lag_168h.
        day = "2015-12-01"
        timestamps = [
            ts.strftime("%Y-%m-%dT%H:%M:%S")
            for ts in pd.date_range(f"{day}T00:00:00", periods=24, freq="h")
        ]
        model_client.post("/predict/batch", json={"grid_id": GRID_ID, "timestamps": timestamps})

        response = model_client.get("/forecast", params={"grid_id": GRID_ID, "date": day})

        assert response.status_code == 200
        body = response.json()
        assert body["grid_id"] == GRID_ID
        assert len(body["predictions"]) == 24
        assert {p["timestamp"] for p in body["predictions"]} == set(timestamps)


class TestMetrics:
    def test_metrics_exposes_prometheus_text(self, model_client):
        response = model_client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "# HELP" in response.text
