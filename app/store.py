"""Feature store and forecast cache — the two storage surfaces the API owns.

Two backend pairs, chosen once in `create_app()` (serving_contract.md §4d):

* `redis`   — `RedisFeatureStore` + `RedisForecastCache`, the compose/production
              path, seeded by `scripts/seed_redis.py`.
* `parquet` — `ParquetFeatureStore` + `InMemoryForecastCache`, which is what
              keeps the test suite free of Redis and Docker while still
              exercising real cache semantics.

**The dtype map is the point of this module.** `mlflow.h2o.log_model` was called
without a signature, so nothing between here and the JVM validates a column
type: an int64 weather column arriving as float64 does not raise, it just makes
H2O reinterpret the column and quietly degrade predictions. Redis has no types
at all — every value comes back as a string — so `_apply_weather_dtypes` on the
read path is the only thing standing between a Redis round trip and silent
corruption. `ParquetFeatureStore` applies the same map so the two backends are
interchangeable by construction rather than by hope.

Timestamps cross the Redis boundary as epoch seconds. `pd.Timestamp.timestamp()`
treats a naive timestamp as UTC (unlike `datetime.timestamp()`, which uses the
host's local zone), so the encoding is deterministic regardless of where the
container runs — which matters, because the whole dataset lives on a naive
fixed-UTC-5 grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date as date_type
from pathlib import Path
from typing import Any

import pandas as pd

from ampops import config
from ampops.utils.io import get_logger

logger = get_logger(__name__)

# serving_contract.md §1a. Ordered exactly as in joined_hourly.parquet: the
# frame handed to `feature_columns()` must come out in the same column order
# training produced, and that order is the joined file's.
WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "snow_depth",
    "snowfall",
    "rain",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
)

# The 8 columns that are int64 in training. Everything else in WEATHER_COLUMNS
# is float64. Getting one of these wrong is the single most likely silent
# corruption bug in the serving layer, hence the explicit set.
WEATHER_INT_COLUMNS: frozenset[str] = frozenset(
    {
        "relative_humidity_2m",
        "weather_code",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_direction_10m",
        "wind_direction_100m",
    }
)

WEATHER_DTYPES: dict[str, str] = {
    col: ("int64" if col in WEATHER_INT_COLUMNS else "float64") for col in WEATHER_COLUMNS
}

# Column order of the window every store returns: exactly joined_hourly.parquet.
WINDOW_COLUMNS: tuple[str, ...] = (config.TIME_COL, config.TARGET, *WEATHER_COLUMNS)

FORECAST_KEY = "ampops:forecast:{grid_id}"
FORECAST_META_KEY = "ampops:forecast_meta:{grid_id}"
LOAD_KEY = "ampops:load:{grid_id}"
WEATHER_KEY = "ampops:weather:{grid_id}:{epoch}"


class InsufficientHistoryError(Exception):
    """The store cannot cover `t-history_hours … t` for every requested target.

    A client error (HTTP 422), not a server fault: asking for a timestamp
    outside the seeded range is a bad request, and answering it with
    partially-null lag features would be far worse than refusing.
    """


def _to_epoch(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _from_epoch(epoch: int | str) -> pd.Timestamp:
    return pd.Timestamp(int(epoch), unit="s")


def _apply_weather_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force the §1a dtype map onto a window frame. See the module docstring."""
    out = df.copy()
    out[config.TIME_COL] = pd.to_datetime(out[config.TIME_COL])
    out[config.TARGET] = out[config.TARGET].astype("float64")
    for col, dtype in WEATHER_DTYPES.items():
        # float first: int64 cannot parse "97.0", which is how a whole number
        # comes back out of Redis after a str() round trip.
        out[col] = pd.to_numeric(out[col]).astype("float64").astype(dtype)
    return out[list(WINDOW_COLUMNS)]


def _window_bounds(
    timestamps: Sequence[pd.Timestamp], history_hours: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """One window spans `min(targets) - history … max(targets)` for the whole batch."""
    targets = [pd.Timestamp(ts) for ts in timestamps]
    if not targets:
        raise InsufficientHistoryError("no timestamps requested")
    return min(targets) - pd.Timedelta(hours=history_hours), max(targets)


def _check_coverage(
    window: pd.DataFrame,
    timestamps: Sequence[pd.Timestamp],
    start: pd.Timestamp,
) -> None:
    """Refuse a window that cannot produce complete features.

    Two distinct failures, both 422: the history runs out before `start` (the
    lag/rolling columns would be null), or a requested target simply is not in
    the store (there is nothing to predict *for*).
    """
    if window.empty:
        raise InsufficientHistoryError(
            f"no data available in {start} … {max(timestamps)}; requested timestamps "
            "are outside the seeded range"
        )

    earliest = window[config.TIME_COL].min()
    if earliest > start:
        raise InsufficientHistoryError(
            f"need history back to {start} but the store starts at {earliest}"
        )

    available = set(window[config.TIME_COL])
    missing = [ts for ts in timestamps if pd.Timestamp(ts) not in available]
    if missing:
        raise InsufficientHistoryError(
            f"no stored observation for {len(missing)} requested timestamp(s), "
            f"first: {missing[0]}"
        )


class FeatureStore(ABC):
    """Source of the raw `time + COMED_MW + 30 weather` window the API features on."""

    @abstractmethod
    def get_window(
        self, grid_id: str, timestamps: Sequence[pd.Timestamp], history_hours: int
    ) -> pd.DataFrame:
        """Rows covering `min(timestamps) - history_hours … max(timestamps)`."""

    @abstractmethod
    def health(self) -> bool:
        """True when the store is reachable and populated."""

    @abstractmethod
    def latest_timestamp(self, grid_id: str) -> pd.Timestamp | None:
        """Newest seeded hour, used for the startup warm-up frame."""


class ParquetFeatureStore(FeatureStore):
    """Feature store backed by an in-memory copy of `joined_hourly.parquet`.

    The dataset is single-grid, so `grid_id` is accepted for interface parity
    with the Redis backend and otherwise ignored — there is no grid column in
    the joined file to filter on.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = _apply_weather_dtypes(
            df.sort_values(config.TIME_COL).reset_index(drop=True)
        )

    @classmethod
    def from_path(cls, path: str | Path) -> ParquetFeatureStore:
        return cls(pd.read_parquet(path))

    def get_window(
        self, grid_id: str, timestamps: Sequence[pd.Timestamp], history_hours: int
    ) -> pd.DataFrame:
        start, end = _window_bounds(timestamps, history_hours)
        times = self._df[config.TIME_COL]
        window = self._df[(times >= start) & (times <= end)].reset_index(drop=True)
        _check_coverage(window, timestamps, start)
        return window

    def health(self) -> bool:
        return not self._df.empty

    def latest_timestamp(self, grid_id: str) -> pd.Timestamp | None:
        if self._df.empty:
            return None
        return self._df[config.TIME_COL].max()


class RedisFeatureStore(FeatureStore):
    """Feature store backed by the Redis hashes `scripts/seed_redis.py` writes.

    Two round trips per window, regardless of its length: one `HMGET` for the
    load hash, one pipelined `HGETALL` batch for the per-hour weather hashes.
    """

    def __init__(self, redis_url: str) -> None:
        import redis  # imported lazily so the parquet backend needs no redis-py

        self._url = redis_url
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get_window(
        self, grid_id: str, timestamps: Sequence[pd.Timestamp], history_hours: int
    ) -> pd.DataFrame:
        start, end = _window_bounds(timestamps, history_hours)
        hours = pd.date_range(start, end, freq="h")
        epochs = [_to_epoch(ts) for ts in hours]

        loads = self._client.hmget(LOAD_KEY.format(grid_id=grid_id), [str(e) for e in epochs])

        pipe = self._client.pipeline()
        for epoch in epochs:
            pipe.hgetall(WEATHER_KEY.format(grid_id=grid_id, epoch=epoch))
        weather = pipe.execute()

        rows: list[dict[str, Any]] = []
        for ts, load, values in zip(hours, loads, weather, strict=True):
            # A gap in the source data is legitimate (joined_hourly has ~11
            # missing hours); `add_lag_features` reindexes onto a complete
            # hourly grid, so a hole here is handled downstream. A *missing
            # target* is not, and `_check_coverage` rejects that below.
            if not values:
                continue
            row: dict[str, Any] = {config.TIME_COL: ts, config.TARGET: load}
            row.update({col: values.get(col) for col in WEATHER_COLUMNS})
            rows.append(row)

        if not rows:
            raise InsufficientHistoryError(
                f"redis has no weather for {grid_id} in {start} … {end}"
            )

        window = _apply_weather_dtypes(pd.DataFrame(rows))
        _check_coverage(window, timestamps, start)
        return window

    def health(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:  # noqa: BLE001 — any Redis failure is "unhealthy"
            logger.warning("Redis feature store unhealthy: %s", exc)
            return False

    def latest_timestamp(self, grid_id: str) -> pd.Timestamp | None:
        fields = self._client.hkeys(LOAD_KEY.format(grid_id=grid_id))
        if not fields:
            return None
        return _from_epoch(max(int(f) for f in fields))


class ForecastCache(ABC):
    """The committed operational forecast, written through by `/predict/batch`.

    Every prediction is stored alongside the model version that produced it.
    Without that pairing the monitoring phase cannot attribute an error back to
    a champion, which is the whole reason the retrain webhook exists.
    """

    @abstractmethod
    def get(self, grid_id: str, ts: pd.Timestamp) -> float | None: ...

    @abstractmethod
    def get_day(self, grid_id: str, date: Any) -> dict[pd.Timestamp, float]: ...

    @abstractmethod
    def put_many(
        self, grid_id: str, preds: dict[pd.Timestamp, float], model_version: str
    ) -> None: ...

    @abstractmethod
    def get_model_version(self, grid_id: str, ts: pd.Timestamp) -> str | None:
        """Model version credited with the cached prediction at `ts`, if any."""


def _same_day(ts: pd.Timestamp, day: Any) -> bool:
    target = pd.Timestamp(day).normalize()
    return pd.Timestamp(ts).normalize() == target


class InMemoryForecastCache(ForecastCache):
    """Process-local cache with the same semantics as the Redis one.

    Used by `make run-api` on the host and by the test suite, so cache
    hit/miss behaviour is exercised for real without a Redis dependency.
    """

    def __init__(self) -> None:
        self._values: dict[str, dict[pd.Timestamp, float]] = {}
        self._versions: dict[str, dict[pd.Timestamp, str]] = {}

    def get(self, grid_id: str, ts: pd.Timestamp) -> float | None:
        return self._values.get(grid_id, {}).get(pd.Timestamp(ts))

    def get_day(
        self, grid_id: str, date: date_type | pd.Timestamp | str
    ) -> dict[pd.Timestamp, float]:
        return {
            ts: value
            for ts, value in self._values.get(grid_id, {}).items()
            if _same_day(ts, date)
        }

    def put_many(
        self, grid_id: str, preds: dict[pd.Timestamp, float], model_version: str
    ) -> None:
        values = self._values.setdefault(grid_id, {})
        versions = self._versions.setdefault(grid_id, {})
        for ts, value in preds.items():
            key = pd.Timestamp(ts)
            values[key] = float(value)
            versions[key] = model_version

    def get_model_version(self, grid_id: str, ts: pd.Timestamp) -> str | None:
        return self._versions.get(grid_id, {}).get(pd.Timestamp(ts))


class RedisForecastCache(ForecastCache):
    """Redis-backed forecast cache: one hash for values, one for model versions.

    No TTL is set. These are the operational forecasts the monitoring layer
    scores against, so expiring them would delete the evidence.
    """

    def __init__(self, redis_url: str) -> None:
        import redis  # lazily imported, as in RedisFeatureStore

        self._url = redis_url
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, grid_id: str, ts: pd.Timestamp) -> float | None:
        raw = self._client.hget(FORECAST_KEY.format(grid_id=grid_id), str(_to_epoch(ts)))
        return None if raw is None else float(raw)

    def get_day(
        self, grid_id: str, date: date_type | pd.Timestamp | str
    ) -> dict[pd.Timestamp, float]:
        raw = self._client.hgetall(FORECAST_KEY.format(grid_id=grid_id))
        out: dict[pd.Timestamp, float] = {}
        for epoch, value in raw.items():
            ts = _from_epoch(epoch)
            if _same_day(ts, date):
                out[ts] = float(value)
        return out

    def put_many(
        self, grid_id: str, preds: dict[pd.Timestamp, float], model_version: str
    ) -> None:
        if not preds:
            return
        values = {str(_to_epoch(ts)): float(value) for ts, value in preds.items()}
        versions = {str(_to_epoch(ts)): model_version for ts in preds}

        pipe = self._client.pipeline()
        pipe.hset(FORECAST_KEY.format(grid_id=grid_id), mapping=values)
        pipe.hset(FORECAST_META_KEY.format(grid_id=grid_id), mapping=versions)
        pipe.execute()

    def get_model_version(self, grid_id: str, ts: pd.Timestamp) -> str | None:
        return self._client.hget(
            FORECAST_META_KEY.format(grid_id=grid_id), str(_to_epoch(ts))
        )


def build_feature_store(backend: str, redis_url: str, parquet_path: str | Path) -> FeatureStore:
    """Backend selection lives here and in `build_forecast_cache` — nowhere else."""
    if backend == "parquet":
        return ParquetFeatureStore.from_path(parquet_path)
    return RedisFeatureStore(redis_url)


def build_forecast_cache(backend: str, redis_url: str) -> ForecastCache:
    if backend == "parquet":
        return InMemoryForecastCache()
    return RedisForecastCache(redis_url)
