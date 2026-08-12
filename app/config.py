"""Environment-driven settings for the serving layer.

Deliberately *not* memoized. `get_settings()` re-reads the environment on every
call, and `create_app()` calls it once per app. The test suite builds several
apps in one process with different `AMPOPS_*` values (a loaded champion, then a
deliberately-missing one), and an `lru_cache` here would silently serve the
first process's configuration to the second — the app would look healthy while
pointing at the wrong model. See docs/serving_contract.md §4d.

Path/model defaults come from `ampops.config` so the API and the training
pipeline cannot disagree about where data lives or what the registry model is
called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ampops import config

# 192h of history, not 168h. `load_roll_mean_168h` at t rolls 168 periods over
# `target.shift(24)`, so it reads back to t-191h; 191h is the true minimum and
# 192h adds one hour of margin. Verified empirically (serving_contract.md §8):
# 191h yields 0 null features, 168h yields 2. Do not reduce this.
HISTORY_HOURS = 192

# Backend pairs selected by AMPOPS_STORE_BACKEND (serving_contract.md §4d).
BACKEND_REDIS = "redis"
BACKEND_PARQUET = "parquet"


@dataclass(frozen=True)
class Settings:
    """One immutable snapshot of the environment, taken at app-construction time."""

    redis_url: str
    store_backend: str
    parquet_path: Path
    model_uri: str
    grid_id: str
    history_hours: int
    mlflow_tracking_uri: str


def resolve_model_uri() -> str:
    """Explicit override wins; otherwise the registry's `@champion` alias.

    `@champion` is not guaranteed to resolve: `registry.register_champion()`
    soft-fails on workspaces without a usable Unity Catalog and returns a
    `runs:/…` URI instead. Startup tolerates that (see `app.model`), so this
    only ever has to name the *preferred* URI.
    """
    override = os.getenv("AMPOPS_SERVING_MODEL_URI", "").strip()
    if override:
        return override
    return f"models:/{config.resolve_registered_model_name()}@champion"


def resolve_store_backend() -> str:
    """Pick the backend pair: explicit setting first, then the local-mode signal.

    `AMPOPS_STORE_BACKEND` wins whenever it is set, and docker-compose sets it
    to `redis` explicitly, so the deployed service is never affected by the
    fallback below.

    When it is *unset*, an explicitly-provided `AMPOPS_PARQUET_PATH` selects the
    parquet pair. That path only means anything in local mode, so setting it is
    already a statement of intent — and it is how the test suite asks for the
    Redis-free pair without a `conftest.py`. Defaulting to `redis` regardless
    would leave a host-side run pointed at a Redis that is not there, failing at
    the first request instead of at startup.
    """
    explicit = os.getenv("AMPOPS_STORE_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("AMPOPS_PARQUET_PATH", "").strip():
        return BACKEND_PARQUET
    return BACKEND_REDIS


def get_settings() -> Settings:
    """Read serving configuration fresh from the environment. Never cached."""
    return Settings(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        store_backend=resolve_store_backend(),
        parquet_path=Path(os.getenv("AMPOPS_PARQUET_PATH", str(config.JOINED_PARQUET))),
        model_uri=resolve_model_uri(),
        grid_id=os.getenv("AMPOPS_GRID_ID", "COMED"),
        history_hours=HISTORY_HOURS,
        # `ampops.config` reads this at import time, which is too early for a
        # caller that sets it afterwards — so read the env first and fall back.
        # There is deliberately no registry-URI setting: MLflow resolves that
        # from `MLFLOW_REGISTRY_URI` itself, and pinning it would mutate global
        # process state (see `app.model.ChampionModel.load`).
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI),
    )
