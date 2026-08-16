"""The AmpOps serving API — the one place a prediction is ever produced.

`create_app()` is a factory rather than a module-level `app` object because the
backend pair (§4d) and the model URI are read from the environment at
construction time, and the test suite builds several apps in one process with
different environments. `app = create_app()` at the bottom is what
`uvicorn app.main:app` imports.

**Single uvicorn worker, always.** Each worker would start its own JVM and race
on the H2O port. See `app.model` for the rest of the cluster lifecycle rules.

Endpoint semantics worth stating once:

* `/predict` reads the forecast cache first and, on a miss, runs live inference
  *without* writing back. Only `/predict/batch` writes through, which keeps the
  cache meaning "the committed operational forecast" rather than "whatever
  someone happened to ask for."
* `/predict/batch` builds **one** window and issues **one** `H2OFrame` round
  trip for the entire batch. Looping the single-prediction path would pay the
  JVM crossing N times and recompute 192 hours of lags N times over.
* `/health` never touches H2O, Redis, or MLflow; `/ready` touches all three.
  `/ready` failing while `/health` passes is the diagnostic signal that the
  champion did not load.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import date as date_type

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from ampops.utils.io import get_logger
from app.config import Settings, get_settings
from app.features import build_inference_frame
from app.model import ChampionModel, ModelNotLoadedError
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
)
from app.store import (
    InsufficientHistoryError,
    build_feature_store,
    build_forecast_cache,
)
from monitoring.drift import RollingDriftMonitor

logger = get_logger(__name__)

# Metrics live at module scope, created exactly once per process. The default
# prometheus_client registry is a global: re-creating these per app would raise
# "Duplicated timeseries in CollectorRegistry" the moment a second app is built
# in the same process, which is precisely what the test suite does.
PREDICTION_LATENCY = Histogram(
    "ampops_prediction_latency_seconds",
    "End-to-end latency of a single prediction, by how it was served",
    labelnames=("source",),
)
PREDICTION_VALUE = Histogram(
    "ampops_prediction_mw",
    "Distribution of predicted load in MW (input to future drift detection)",
    buckets=(6_000, 9_000, 12_000, 15_000, 18_000, 21_000, 24_000, float("inf")),
)
CACHE_EVENTS = Counter(
    "ampops_forecast_cache_events_total",
    "Forecast-cache lookups on /predict",
    labelnames=("result",),
)
DRIFT_PSI = Gauge(
    "ampops_drift_psi",
    "PSI drift score for each monitored inference feature",
    labelnames=("feature",),
)

DRIFT_ALERT = Gauge(
    "ampops_drift_alert",
    "Whether feature drift is currently detected: 1=yes, 0=no",
)

DRIFT_ALERT_FEATURES = Gauge(
    "ampops_drift_alert_features",
    "Number of monitored features currently in drift alert state",
)

DRIFT_WINDOW_ROWS = Gauge(
    "ampops_drift_window_rows",
    "Number of inference rows currently collected for drift monitoring",
)
# One instrumentator for the process, matching the metrics above. The library
# tolerates a duplicate registration by dropping the second app's default
# metrics, so sharing the instance is the only way every app instruments the
# same collectors rather than silently instrumenting none.
INSTRUMENTATOR = Instrumentator()


def create_app() -> FastAPI:
    """Build a fully wired app. Backend selection happens here and nowhere else."""
    settings = get_settings()
    champion = ChampionModel()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.champion = champion
        app.state.store = build_feature_store(
            settings.store_backend, settings.redis_url, settings.parquet_path
        )
        app.state.cache = build_forecast_cache(settings.store_backend, settings.redis_url)
        app.state.drift_monitor = RollingDriftMonitor(window_size=168)
        # A JVM that will not start is reported through /ready, not by dying at
        # boot: the model load below fails on its own in that case and records
        # why, and a crashed container tells an operator far less than a 503
        # naming the failed check.
        try:
            champion.start_cluster()
        except Exception as exc:  # noqa: BLE001 — surfaced via /ready 503
            logger.error("H2O cluster failed to start: %s", exc)

        champion.load(settings.model_uri, tracking_uri=settings.mlflow_tracking_uri)
        _warm_up(app)

        yield

        champion.shutdown_cluster()

    app = FastAPI(
        title="AmpOps demand forecasting API",
        description="Short-term electricity load forecasting for the ComEd zone.",
        lifespan=lifespan,
    )

    _register_error_handlers(app)
    _register_routes(app)
    INSTRUMENTATOR.instrument(app).expose(app, include_in_schema=False)
    return app


def _observe_drift(app: FastAPI, frame: pd.DataFrame) -> None:
    """Record inference features and publish rolling drift state to Prometheus."""
    result = app.state.drift_monitor.observe(frame)

    DRIFT_WINDOW_ROWS.set(result["rows"])

    if not result["ready"]:
        return

    for feature, info in result["features"].items():
        if info["psi"] is not None:
            DRIFT_PSI.labels(feature=feature).set(info["psi"])

    DRIFT_ALERT.set(1 if result["drift_detected"] else 0)
    DRIFT_ALERT_FEATURES.set(result["alert_count"])


def _warm_up(app: FastAPI) -> None:
    """Pay the one-time `holidays` and H2O costs before the first real request.

    The first `add_calendar_features` call in a cold process loads the holiday
    tables, which has been observed taking tens of seconds on a cold filesystem
    cache (serving_contract.md §8); the first `H2OFrame` pays the JVM
    handshake. Doing both here keeps request #1 from being an outlier. Purely
    an optimization — any failure is logged and ignored.
    """
    store = app.state.store
    settings: Settings = app.state.settings
    champion: ChampionModel = app.state.champion
    try:
        latest = store.latest_timestamp(settings.grid_id)
        if latest is None:
            logger.warning("Feature store is empty; skipping warm-up")
            return
        frame = build_inference_frame(store, settings.grid_id, [latest])
        if champion.is_loaded:
            champion.predict(frame)
        logger.info("Warm-up inference frame built for %s", latest)
    except Exception as exc:  # noqa: BLE001 — warm-up must never block startup
        logger.warning("Warm-up skipped: %s", exc)


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InsufficientHistoryError)
    async def _insufficient_history(request: Request, exc: InsufficientHistoryError):
        # A client error, not a server fault: the requested hour is outside the
        # seeded range, so there is nothing the API could do differently.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})


def _settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness only. Deliberately touches nothing external."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(request: Request):
        champion: ChampionModel = request.app.state.champion
        model_loaded = champion.is_loaded
        cluster_ok = champion.cluster_is_running()
        store_ok = request.app.state.store.health()

        failed = [
            name
            for name, ok in (
                ("model", model_loaded),
                ("h2o_cluster", cluster_ok),
                ("feature_store", store_ok),
            )
            if not ok
        ]
        detail = None
        if failed:
            detail = f"failed checks: {', '.join(failed)}"
            if champion.load_error:
                detail = f"{detail} ({champion.load_error})"

        body = ReadinessResponse(
            ready=not failed,
            model_loaded=model_loaded,
            h2o_cluster=cluster_ok,
            feature_store=store_ok,
            model_uri=champion.model_uri,
            model_version=champion.version,
            detail=detail,
        )
        if failed:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        """One hour, cache-first. A live miss is answered but not written back.

        Only `/predict/batch` writes through, so the cache keeps meaning "the
        committed operational forecast" rather than a log of whatever anyone
        happened to explore.
        """
        started = time.perf_counter()
        ts = pd.Timestamp(payload.timestamp)
        cache = app.state.cache

        cached = cache.get(payload.grid_id, ts)
        if cached is not None:
            source = "cache"
            value = float(cached)
            version = cache.get_model_version(payload.grid_id, ts)
            model_version = version or app.state.champion.version or "unknown"
            CACHE_EVENTS.labels(result="hit").inc()
        else:
            source = "live"
            champion: ChampionModel = app.state.champion
            frame = build_inference_frame(app.state.store, payload.grid_id, [ts])
            _observe_drift(app, frame)
            value = float(champion.predict(frame)[0])
            model_version = champion.version or "unknown"
            CACHE_EVENTS.labels(result="miss").inc()

        elapsed = time.perf_counter() - started
        PREDICTION_LATENCY.labels(source=source).observe(elapsed)
        PREDICTION_VALUE.observe(value)

        return PredictResponse(
            grid_id=payload.grid_id,
            timestamp=payload.timestamp,
            predicted_mw=value,
            model_version=model_version,
            source=source,
            latency_ms=elapsed * 1_000,
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        """The whole batch in one window and one JVM round trip, then write through."""
        started = time.perf_counter()
        champion: ChampionModel = app.state.champion
        targets = [pd.Timestamp(ts) for ts in payload.timestamps]

        # One window, one H2OFrame round trip for the whole batch.
        frame = build_inference_frame(app.state.store, payload.grid_id, targets)
        _observe_drift(app, frame)
        values = [float(v) for v in champion.predict(frame)]

        model_version = champion.version or "unknown"
        app.state.cache.put_many(
            payload.grid_id, dict(zip(targets, values, strict=True)), model_version
        )

        elapsed = time.perf_counter() - started
        PREDICTION_LATENCY.labels(source="live").observe(elapsed / max(len(values), 1))
        for value in values:
            PREDICTION_VALUE.observe(value)

        return BatchPredictResponse(
            grid_id=payload.grid_id,
            model_version=model_version,
            predictions=[
                Prediction(timestamp=ts, predicted_mw=value)
                for ts, value in zip(payload.timestamps, values, strict=True)
            ],
        )

    @app.get("/forecast", response_model=BatchPredictResponse)
    def forecast(
        date: date_type = Query(...),
        grid_id: str | None = Query(default=None),
        settings: Settings = Depends(_settings_dep),
    ) -> BatchPredictResponse:
        """Read-only view of the committed forecast. Never runs inference."""
        resolved_grid = grid_id or settings.grid_id
        cache = app.state.cache
        cached = cache.get_day(resolved_grid, pd.Timestamp(date))
        if not cached:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no cached forecast for {resolved_grid} on {date}; "
                    "run POST /predict/batch first"
                ),
            )

        ordered = sorted(cached.items())
        model_version = (
            cache.get_model_version(resolved_grid, ordered[0][0])
            or app.state.champion.version
            or "unknown"
        )
        return BatchPredictResponse(
            grid_id=resolved_grid,
            model_version=model_version,
            predictions=[
                Prediction(timestamp=ts, predicted_mw=value) for ts, value in ordered
            ],
        )


app = create_app()
