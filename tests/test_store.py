"""Tests for the feature-store and forecast-cache storage layer.

Two failure modes matter here, per docs/serving_contract.md §4a and §7:

1. `ParquetFeatureStore` (the Redis-free path used by every other serving
   test) must reproduce the exact window shape and dtypes a Redis-backed
   store would — including the 8 int64 weather columns, since the model's
   missing MLflow signature means a wrong dtype fails silently at predict
   time, not loudly at load time.
2. Redis itself has no types: everything round-trips through it as bytes.
   `RedisFeatureStore` must apply serving_contract.md §1a's dtype map on
   *read*, or those same 8 columns silently become float64. This is called
   out in the contract as "the single most likely silent-corruption bug in
   the whole serving layer," so it gets a dedicated round-trip assertion
   below rather than just a smoke test.

The Redis-backed tests skip via a live connection probe rather than
`importorskip`, since `redis-py` being installed says nothing about whether a
Redis server is actually reachable from this environment.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from ampops import config

if not config.JOINED_PARQUET.exists():
    pytest.skip(
        "data/processed/joined_hourly.parquet is a gitignored data artifact "
        "and is not present in this environment",
        allow_module_level=True,
    )

from app.store import (  # noqa: E402
    ForecastCache,
    InMemoryForecastCache,
    InsufficientHistoryError,
    ParquetFeatureStore,
)

# Single-grid dataset; see test_serving_features.py for why "COMED" is used.
GRID_ID = "COMED"
HISTORY_HOURS = 192

# serving_contract.md §1a — these 8 columns are the dtype trap. Redis returns
# strings for everything; deserializing these as float64 is the specific
# silent-corruption bug this file exists to catch.
WEATHER_INT64_COLUMNS = [
    "relative_humidity_2m",
    "weather_code",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_direction_10m",
    "wind_direction_100m",
]
WEATHER_FLOAT64_COLUMNS = [
    "temperature_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "snow_depth",
    "snowfall",
    "rain",
    "pressure_msl",
    "surface_pressure",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]


@pytest.fixture(scope="module")
def joined() -> pd.DataFrame:
    return pd.read_parquet(config.JOINED_PARQUET)


@pytest.fixture(scope="module")
def store(joined) -> ParquetFeatureStore:
    return ParquetFeatureStore(joined)


class TestParquetFeatureStoreWindow:
    def test_get_window_spans_exactly_history_hours_plus_the_target(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        window = store.get_window(GRID_ID, [ts], history_hours=HISTORY_HOURS)

        assert window[config.TIME_COL].min() == ts - pd.Timedelta(hours=HISTORY_HOURS)
        assert window[config.TIME_COL].max() == ts
        assert len(window) == HISTORY_HOURS + 1

    def test_get_window_spans_min_to_max_target_for_a_batch(self, store):
        timestamps = list(pd.date_range("2015-06-01 00:00:00", periods=24, freq="h"))
        window = store.get_window(GRID_ID, timestamps, history_hours=HISTORY_HOURS)

        assert window[config.TIME_COL].min() == min(timestamps) - pd.Timedelta(
            hours=HISTORY_HOURS
        )
        assert window[config.TIME_COL].max() == max(timestamps)

    def test_get_window_columns_match_the_joined_source(self, store, joined):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        window = store.get_window(GRID_ID, [ts], history_hours=HISTORY_HOURS)

        assert set(window.columns) == set(joined.columns)

    def test_int64_weather_columns_stay_int64(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        window = store.get_window(GRID_ID, [ts], history_hours=HISTORY_HOURS)

        for col in WEATHER_INT64_COLUMNS:
            assert window[col].dtype == "int64", f"{col} came back as {window[col].dtype}"

    def test_float64_weather_columns_stay_float64(self, store):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        window = store.get_window(GRID_ID, [ts], history_hours=HISTORY_HOURS)

        for col in WEATHER_FLOAT64_COLUMNS:
            assert window[col].dtype == "float64", f"{col} came back as {window[col].dtype}"

    def test_get_window_raises_outside_the_seeded_range(self, store, joined):
        too_early = joined[config.TIME_COL].min() + pd.Timedelta(hours=10)

        with pytest.raises(InsufficientHistoryError):
            store.get_window(GRID_ID, [too_early], history_hours=HISTORY_HOURS)

    def test_get_window_raises_for_a_timestamp_entirely_outside_the_data(self, store):
        with pytest.raises(InsufficientHistoryError):
            store.get_window(
                GRID_ID, [pd.Timestamp("2030-01-01 00:00:00")], history_hours=HISTORY_HOURS
            )

    def test_health_reports_true_for_a_populated_store(self, store):
        assert store.health() is True


class TestInMemoryForecastCache:
    def test_put_many_then_get_returns_the_stored_prediction(self):
        cache: ForecastCache = InMemoryForecastCache()
        ts = pd.Timestamp("2015-07-15 14:00:00")

        cache.put_many("COMED", {ts: 18432.7}, model_version="3")

        assert cache.get("COMED", ts) == pytest.approx(18432.7)

    def test_get_misses_return_none(self):
        cache = InMemoryForecastCache()

        assert cache.get("COMED", pd.Timestamp("2015-07-15 14:00:00")) is None

    def test_get_day_returns_every_cached_hour_for_that_date(self):
        cache = InMemoryForecastCache()
        day = pd.Timestamp("2015-07-15")
        preds = {day + pd.Timedelta(hours=h): float(10_000 + h) for h in range(24)}

        cache.put_many("COMED", preds, model_version="3")
        out = cache.get_day("COMED", day)

        assert len(out) == 24
        assert out[day + pd.Timedelta(hours=5)] == pytest.approx(10_005.0)

    def test_get_day_is_empty_for_a_date_with_nothing_cached(self):
        cache = InMemoryForecastCache()

        assert cache.get_day("COMED", pd.Timestamp("2020-01-01")) == {}

    def test_different_grid_ids_do_not_share_a_cache_entry(self):
        cache = InMemoryForecastCache()
        ts = pd.Timestamp("2015-07-15 14:00:00")

        cache.put_many("COMED", {ts: 18432.7}, model_version="3")

        assert cache.get("OTHER_GRID", ts) is None


# --- Redis-backed store/cache: real Redis, skipped when none is reachable ---

redis = pytest.importorskip("redis", reason="redis-py only needed for the Redis-backed tests")

from app.store import RedisFeatureStore, RedisForecastCache  # noqa: E402

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TEST_GRID = "TEST_GRID_DO_NOT_USE_IN_PROD"


def _redis_reachable() -> bool:
    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5).ping()
        return True
    except redis.exceptions.RedisError:
        return False


@pytest.fixture(scope="module")
def redis_client():
    if not _redis_reachable():
        pytest.skip(f"no Redis reachable at {REDIS_URL}")

    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    yield client

    for key in client.scan_iter(f"ampops:*:{REDIS_TEST_GRID}*"):
        client.delete(key)


class TestRedisFeatureStoreDtypeRoundTrip:
    """The dtype map surviving the string round trip is the crown-jewel
    assertion for this file — see the module docstring.
    """

    def test_int64_and_float64_weather_columns_survive_the_redis_round_trip(
        self, redis_client, joined
    ):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        history_hours = 3
        window_source = joined[
            (joined[config.TIME_COL] >= ts - pd.Timedelta(hours=history_hours))
            & (joined[config.TIME_COL] <= ts)
        ].sort_values(config.TIME_COL)
        assert len(window_source) == history_hours + 1

        weather_cols = WEATHER_INT64_COLUMNS + WEATHER_FLOAT64_COLUMNS
        load_key = f"ampops:load:{REDIS_TEST_GRID}"
        weather_keys = []
        try:
            for _, row in window_source.iterrows():
                epoch = int(row[config.TIME_COL].timestamp())
                redis_client.hset(load_key, epoch, str(row[config.TARGET]))

                weather_key = f"ampops:weather:{REDIS_TEST_GRID}:{epoch}"
                redis_client.hset(weather_key, mapping={c: str(row[c]) for c in weather_cols})
                weather_keys.append(weather_key)

            store = RedisFeatureStore(REDIS_URL)
            window = store.get_window(REDIS_TEST_GRID, [ts], history_hours=history_hours)

            for col in WEATHER_INT64_COLUMNS:
                assert window[col].dtype == "int64", f"{col} came back as {window[col].dtype}"
            for col in WEATHER_FLOAT64_COLUMNS:
                assert window[col].dtype == "float64", f"{col} came back as {window[col].dtype}"

            expected_row = window_source[window_source[config.TIME_COL] == ts].iloc[0]
            got_row = window[window[config.TIME_COL] == ts].iloc[0]
            for col in weather_cols:
                assert got_row[col] == pytest.approx(expected_row[col]), col
        finally:
            redis_client.delete(load_key)
            for key in weather_keys:
                redis_client.delete(key)


class TestRedisForecastCache:
    def test_put_many_then_get_round_trips_through_redis(self, redis_client):
        ts = pd.Timestamp("2015-07-15 14:00:00")
        cache = RedisForecastCache(REDIS_URL)

        try:
            cache.put_many(REDIS_TEST_GRID, {ts: 18432.7}, model_version="3")

            assert cache.get(REDIS_TEST_GRID, ts) == pytest.approx(18432.7)

            forecast_key = f"ampops:forecast:{REDIS_TEST_GRID}"
            meta_key = f"ampops:forecast_meta:{REDIS_TEST_GRID}"
            assert redis_client.hget(forecast_key, str(int(ts.timestamp()))) is not None
            assert redis_client.hget(meta_key, str(int(ts.timestamp()))) == "3"
        finally:
            redis_client.delete(f"ampops:forecast:{REDIS_TEST_GRID}")
            redis_client.delete(f"ampops:forecast_meta:{REDIS_TEST_GRID}")

    def test_get_day_returns_every_cached_hour_for_that_date(self, redis_client):
        day = pd.Timestamp("2015-08-01")
        preds = {day + pd.Timedelta(hours=h): float(10_000 + h) for h in range(24)}
        cache = RedisForecastCache(REDIS_URL)

        try:
            cache.put_many(REDIS_TEST_GRID, preds, model_version="3")
            out = cache.get_day(REDIS_TEST_GRID, day)

            assert len(out) == 24
            assert out[day + pd.Timedelta(hours=5)] == pytest.approx(10_005.0)
        finally:
            redis_client.delete(f"ampops:forecast:{REDIS_TEST_GRID}")
            redis_client.delete(f"ampops:forecast_meta:{REDIS_TEST_GRID}")
