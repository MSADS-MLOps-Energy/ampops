# AmpOps Serving Contract

**Status:** authoritative spec for the FastAPI serving layer.
**Audience:** anyone implementing, testing, or documenting `app/`,
`dags/ampops_daily_forecast.py`, or `scripts/seed_redis.py`.

This document is the single source of truth for the serving layer. It exists so that
nobody has to re-derive the feature schema, the H2O loading pattern, or the storage key
layout by reading `src/ampops/` again. **Read this first; treat it as binding.** If
implementation and this document disagree, that is a bug in one of them — resolve it
explicitly rather than silently diverging.

Everything in §1 and §8 was verified empirically against the real artifacts
(`data/processed/train.parquet`, `joined_hourly.parquet`), not inferred from code.

---

## 1. The model input contract

The champion consumes **exactly 49 feature columns**. There is **no MLflow signature**
on the logged model (`mlflow.h2o.log_model(leader, artifact_path="model")` in
`src/ampops/training/automl.py:311` passes neither `signature=` nor `input_example=`).

**Consequence, and the single most important fact in this document:** MLflow and H2O
perform *zero* column validation and *zero* dtype coercion at predict time. A wrong
dtype does not raise — H2O silently infers a different column type (a numeric column
arriving as a string becomes an `enum`) and predictions degrade in a way that looks
plausible. **The API owns 100% of schema correctness.**

The column set is defined by `ampops.features.build.feature_columns(df)`, which is
derived, not hardcoded: *everything except `time` and `COMED_MW`*.

### 1a. Weather block — 30 columns, sourced from the feature store

Read verbatim from `joined_hourly.parquet`. Column names come from
`clean.clean_col` (strip `(units)`, lowercase, snake_case).

`float64` (22):
`temperature_2m`, `dew_point_2m`, `apparent_temperature`, `precipitation`,
`snow_depth`, `snowfall`, `rain`, `pressure_msl`, `surface_pressure`,
`et0_fao_evapotranspiration`, `vapour_pressure_deficit`, `wind_speed_10m`,
`wind_speed_100m`, `wind_gusts_10m`, `soil_temperature_0_to_7cm`,
`soil_temperature_7_to_28cm`, `soil_temperature_28_to_100cm`,
`soil_temperature_100_to_255cm`, `soil_moisture_0_to_7cm`, `soil_moisture_7_to_28cm`,
`soil_moisture_28_to_100cm`, `soil_moisture_100_to_255cm`

**`int64` (8) — these are the dtype trap.** Redis returns strings; deserializing these
as `float64` is the most likely silent-corruption bug in the whole serving layer:
`relative_humidity_2m`, `weather_code`, `cloud_cover`, `cloud_cover_low`,
`cloud_cover_mid`, `cloud_cover_high`, `wind_direction_10m`, `wind_direction_100m`

### 1b. Derived block — 19 columns, computed at request time

Never store these; always recompute. Produced by `add_calendar_features` +
`add_lag_features`.

| Columns | dtype | Source |
|---|---|---|
| `hour`, `dayofweek`, `month`, `dayofyear`, `year` | **`int32`** | `add_calendar_features` |
| `is_weekend`, `is_holiday` | `int64` | `add_calendar_features` |
| `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos` | `float64` | `add_calendar_features` |
| `load_lag_24h`, `load_lag_168h` | `float64` | `add_lag_features` |
| `load_roll_mean_24h`, `load_roll_std_24h`, `load_roll_mean_168h`, `load_roll_std_168h` | `float64` | `add_lag_features` |

`int32` vs `int64` is not cosmetic — it is what `pandas` produces from
`.dt.hour` and what training fed H2O. Reproduce it by using the real functions (§2),
not by constructing columns by hand.

---

## 2. Building the inference frame — reuse, never reimplement

**Rule: `app/features.py` must not contain any feature arithmetic.** It assembles a
window and delegates to the training functions. This makes training/serving skew
structurally impossible: if `build.py` changes, serving changes with it.

```python
from ampops import config
from ampops.features.build import add_calendar_features, add_lag_features, feature_columns

# window: DataFrame[time, COMED_MW, <30 weather>] covering t-192h .. t,
#         with COMED_MW set to NaN at every target timestamp.
frame = add_lag_features(add_calendar_features(window))
rows  = frame[frame[config.TIME_COL].isin(target_times)]
X     = rows[feature_columns(frame)]          # the 49 columns, correct dtypes
```

Three rules that are easy to get wrong:

1. **Do NOT call `build_features()`.** Its trailing
   `dropna(subset=[c for c in cols if c.startswith("load_")])` would discard the very
   row you are predicting.
2. **`COMED_MW` at the target timestamp must be `NaN`, and that is correct** — not a
   workaround. Every lag references `t-24` or older, so no feature reads it. Setting it
   to `0.0` or omitting the row would corrupt the lag grid.
3. **Cast the target column to `float64` after blanking.** `df.loc[mask, col] = pd.NA`
   on a float column can flip it to `object`, which breaks `add_lag_features`' shifts.

### History window

**192 hours** (`t-192h … t`, 193 rows for a single prediction).

Derivation: `load_roll_mean_168h` at `t` rolls 168 periods over
`horizon_safe = target.shift(24)`, so it reads `target[t-191h … t-24h]`. Verified
empirically: 191h of history yields 0 null features, 168h yields 2 nulls. 192h is the
minimum plus one hour of margin — do not reduce it.

For a batch of N timestamps, build **one** window spanning
`min(targets) - 192h … max(targets)` and blank `COMED_MW` at all N targets. Do not loop.

### Insufficient history

Raise a typed `InsufficientHistoryError` → HTTP **422**. A timestamp outside the seeded
range is a client error, not a server fault.

### Missing timestamps are a real, permanent feature of this dataset

`joined_hourly.parquet` is missing **exactly 10 timestamps**, every one a DST fall-back
artifact (documented in `CLAUDE.md` and `AmpOps_Project_Context.md` §2.3):

```
2011-11-06 00:00 / 02:00   2012-11-04 00:00 / 02:00   2013-11-03 00:00 / 02:00
2014-11-02 00:00           2015-11-01 00:00           2016-11-06 00:00
2017-11-05 00:00
```

Two consequences that have already caused one bug:

1. **Requesting a missing hour 422s the entire batch.** This is deliberate. Serving it
   would require inserting a weather-less row, and because pandas dtypes are per-column,
   one such row demotes all 8 `int64` weather columns to `float64` **for the whole
   batch** — trading the exact silent corruption §1a exists to prevent for one hour that
   has no weather observation anyway.
2. **A gap inside the 192h history window silently produces `NaN` features**, since
   `add_lag_features` reindexes onto a gap-free grid. A date one week after a fall-back
   hour still reads that gap through its own `load_lag_168h`.

**When choosing dates for tests or demos, verify the full span `[t-192h, t+23h]` is
present**, not just the target day.

---

## 3. Loading and calling the model

```python
import h2o, mlflow.h2o

h2o.init(port=_free_port(), start_h2o=True)      # ONCE, in FastAPI lifespan startup
model = mlflow.h2o.load_model(model_uri)         # requires a live cluster
preds = model.predict(h2o.H2OFrame(X)).as_data_frame()["predict"].to_numpy()
```

**Use `mlflow.h2o.load_model`, not `mlflow.pyfunc.load_model`.** The pyfunc wrapper
auto-calls `h2o.init()` with whatever is in the artifact's `h2o.yaml` — typically the
default port 54321, with no port control. That is precisely the stale-cluster hazard
`automl.py`'s module docstring warns about, and it bites when the container restarts.

Lifecycle rules:

- **Init once in the lifespan startup handler; shut down in lifespan shutdown.**
- **Never call `h2o.cluster().shutdown()` in a request handler.** `automl.py` does this
  in a `finally` block, which is correct for a batch Airflow task and fatal for a
  long-lived server. Do not copy that pattern.
- Mirror `automl._free_port()` and `automl._ensure_java_home()`. `_ensure_java_home` is
  a macOS Homebrew heuristic and is a no-op in the Linux container — harmless, keep it
  for host dev.
- **Pin uvicorn to a single worker.** Multiple workers would race on the JVM port. This
  is a hard constraint; document it wherever the service is started.
- Training always passed the target column *into* `predict`; serving will not. H2O
  supports this, but it is not exercised upstream — the parity test (§7) is what proves
  it is safe.

### Model URI resolution

```
AMPOPS_SERVING_MODEL_URI    (explicit override; wins if set)
  else  models:/{config.resolve_registered_model_name()}@champion
```

**`@champion` is not guaranteed to resolve.** `registry.register_champion()` soft-fails
when the workspace has no UC catalog, returning `model_uri = "runs:/<run_id>/model"` and
`skipped=True`. On load failure the process **must still start** — log the error, leave
the model unloaded, and report it via `/ready` (503). Crashing on startup would make the
container unrecoverable in exactly the configuration the training pipeline tolerates.

`model_version`, echoed in every response, is the registry version string when available,
else the run id. It is never invented.

### Java

The model cannot load without a JRE. `openjdk-17-jre-headless` must be in the serving
image (H2O 3.46.x supports Java 8–17; **21 is unsupported**). `h2o==3.46.0.11` is
already pinned in `requirements.txt`.

---

## 4. Storage layout

### 4a. Redis — feature store (read path, seeded by `scripts/seed_redis.py`)

| Key | Type | Contents |
|---|---|---|
| `ampops:load:{grid_id}` | hash | field = epoch-hour (int seconds), value = `COMED_MW` |
| `ampops:weather:{grid_id}:{epoch}` | hash | the 30 weather columns for that hour |

Read pattern: one `HMGET` for the 192 load fields, one pipelined `HGETALL` batch for
weather. Two round trips total.

**Apply the §1a dtype map on deserialization.** Redis has no types; everything comes
back as `bytes`/`str`.

### 4b. Redis — forecast cache (write-through, owned by the API)

| Key | Type | Contents |
|---|---|---|
| `ampops:forecast:{grid_id}` | hash | field = epoch-hour, value = predicted MW |
| `ampops:forecast_meta:{grid_id}` | hash | field = epoch-hour, value = `model_version` |

Written by `POST /predict/batch` as it computes. **No aggressive TTL** — long or none.
The meta hash exists so a cached prediction is attributable to a model version;
without it the monitoring phase cannot tell which model produced what.

### 4c. Postgres — the system of record

Database `ampops` on the existing `ampops-postgres` instance (Airflow's `airflow`
database is untouched). Written **only** by the forecast DAG.

```sql
CREATE TABLE IF NOT EXISTS forecasts (
  grid_id        text             NOT NULL,
  target_ts      timestamptz      NOT NULL,
  predicted_mw   double precision NOT NULL,
  model_version  text             NOT NULL,
  generated_at   timestamptz      NOT NULL DEFAULT now(),
  run_id         text,
  PRIMARY KEY (grid_id, target_ts, model_version)
);
```

The composite PK makes DAG re-runs idempotent via `ON CONFLICT ... DO UPDATE`, and lets
two model versions forecast the same hour — which champion/challenger comparison needs.

**Only scheduled batches are recorded.** Ad-hoc `/predict` traffic stays ephemeral;
mixing exploratory calls into the record would pollute the accuracy metric the retrain
webhook keys off.

> **initdb ordering gotcha.** `postgres:15` runs `/docker-entrypoint-initdb.d/*` only on
> *first* initialization of the `postgres-data` volume. Anyone with an existing volume
> will not get the `ampops` database. Hence three mitigations:
> `docker/postgres/init-ampops-db.sh` (fresh volumes), `make ampops-db-init`
> (existing stacks, idempotent), and the DAG catching "database does not exist" to
> re-raise it pointing at that make target.

### 4d. Backend selection — Redis mode vs local mode

The service runs against two interchangeable backend pairs, chosen by
`AMPOPS_STORE_BACKEND`:

| Value | FeatureStore | ForecastCache | Used by |
|---|---|---|---|
| `redis` *(default)* | `RedisFeatureStore` | `RedisForecastCache` | Docker compose, production |
| `parquet` | `ParquetFeatureStore` | `InMemoryForecastCache` | tests, `make run-api` on the host |

Local mode is what keeps the test suite free of Redis and Docker while still exercising
real cache semantics. `create_app()` performs this wiring; nothing else in the app may
branch on the backend.

Resolution order (`resolve_store_backend()`):
1. An explicitly set `AMPOPS_STORE_BACKEND` always wins.
2. Otherwise, an explicitly set `AMPOPS_PARQUET_PATH` selects the parquet pair.
3. Otherwise, `redis`.

Step 2 exists because the test suite has no `conftest.py` and never sets the backend
variable; a strict `redis` default would point every test at an unreachable server.
Compose sets `AMPOPS_STORE_BACKEND=redis` explicitly, so production never relies on
inference.

**Pinned constructor signatures** (the tests import these directly, so they are part of
the contract, not an implementation detail):

```python
ParquetFeatureStore(df: pd.DataFrame)                  # already-loaded frame
ParquetFeatureStore.from_path(path: str | Path)        # convenience loader
RedisFeatureStore(redis_url: str)
RedisForecastCache(redis_url: str)
InMemoryForecastCache()                                 # no args

FeatureStore.get_window(grid_id: str, timestamps: list[pd.Timestamp],
                        history_hours: int) -> pd.DataFrame
FeatureStore.health() -> bool
ForecastCache.get(grid_id: str, ts: pd.Timestamp) -> float | None
ForecastCache.get_day(grid_id: str, date) -> dict[pd.Timestamp, float]
ForecastCache.put_many(grid_id: str, preds: dict[pd.Timestamp, float],
                       model_version: str) -> None
```

**`get_settings()` must read the environment fresh on every call — do not memoize it**
(no `lru_cache`). Tests construct multiple apps in one process with different env, and a
cached settings object would silently leak the first configuration into the second.

---

## 5. HTTP API

All request/response bodies are JSON. Timestamps are ISO-8601 **naive** strings on the
fixed UTC-5 grid — the same convention as `joined_hourly.parquet`'s `time` column. Do
not accept or emit tz-aware timestamps; that grid is an internal invariant.

### `POST /predict`

```jsonc
// request
{ "grid_id": "COMED", "timestamp": "2018-07-15T14:00:00" }

// 200
{ "grid_id": "COMED", "timestamp": "2018-07-15T14:00:00",
  "predicted_mw": 18432.7, "model_version": "3",
  "source": "cache",           // "cache" | "live"
  "latency_ms": 1.4 }
```

Checks the forecast cache first; on a miss runs live inference and returns
`source: "live"`. A live miss does **not** populate the cache (only `/predict/batch`
writes through) — this keeps the cache meaning "the committed operational forecast".

### `POST /predict/batch`

```jsonc
// request
{ "grid_id": "COMED", "timestamps": ["2018-07-15T00:00:00", "..."] }

// 200
{ "grid_id": "COMED", "model_version": "3",
  "predictions": [ { "timestamp": "...", "predicted_mw": 18432.7 } ] }
```

**Must build one window and issue one `H2OFrame` round trip for the whole batch.**
Looping `/predict` internally would pay the JVM round trip N times and defeat the point.
Writes through to the forecast cache (§4b).

### `GET /forecast?grid_id=COMED&date=2018-07-15`

Read-only. Returns the cached forecasts for that date, or **404** if none. Shape mirrors
`/predict/batch` for consistency:

```jsonc
{ "grid_id": "COMED", "model_version": "3",
  "predictions": [ { "timestamp": "2018-07-15T00:00:00", "predicted_mw": 17204.1 } ] }
```

### `GET /health`

Liveness. Always **200** if the process is up. Must not touch H2O, Redis, or MLflow.

### `GET /ready`

**200** only if the model is loaded **and** the H2O cluster is reachable **and** the
feature store is healthy. Otherwise **503** with a body naming which check failed.

`/health` and `/ready` must genuinely differ: `/ready` failing while `/health` passes is
the diagnostic signal that the champion did not load.

### `GET /metrics`

`prometheus-fastapi-instrumentator` defaults, plus:
- `ampops_prediction_latency_seconds` — latency histogram, labelled by `source`
  (`cache` vs `live`), so the ~1.3ms and ~300–530ms paths stay separable
- `ampops_prediction_mw` — prediction-value histogram, 8 explicit buckets from
  6,000 to 24,000 MW
- `ampops_forecast_cache_events_total` — cache hit/miss counter

All three are **implemented and scraped** (`monitoring/prometheus.yml`, 15s interval).
Note what they are and are not: these describe the *output* and the *service*, never the
*input*. `ampops_prediction_mw` supports prediction-drift monitoring on its own, but
input/feature drift needs the 49-column frame persisted somewhere, and nothing persists
it today. Grafana is running but unprovisioned — no datasource, no dashboards.

### Error codes

| Code | Condition |
|---|---|
| 422 | `InsufficientHistoryError`, or a timestamp outside the seeded range |
| 404 | `/forecast` with no cached entries for that date |
| 503 | model not loaded / H2O down / feature store unreachable |
| 400 | malformed body (FastAPI/Pydantic default) |

---

## 6. Forecast DAG — `dags/ampops_daily_forecast.py`

Orchestration only, mirroring `dags/ampops_training_pipeline.py`'s established style
(thin tasks, paths/values through XCom, no business logic in the DAG).

```
resolve_horizon()  -> 24 timestamps for the next operating day
request_forecast() -> POST {AMPOPS_API_URL}/predict/batch
persist()          -> CREATE TABLE IF NOT EXISTS + upsert 24 rows (ON CONFLICT DO UPDATE)
verify_cached()    -> GET /forecast, assert all 24 landed, else fail the run
```

- Imports allowed: `requests`, `PostgresHook`, stdlib, `pendulum`.
  **Forbidden: `h2o`, `mlflow`, `ampops.features`, `ampops.training`.** The whole point
  is one inference code path.
- `schedule="@daily"`, `catchup=False`, `max_active_runs=1`.
- Connection via `AIRFLOW_CONN_AMPOPS_DB` env var — no manual Airflow-UI setup.
- The `api` service lives behind the `serving` compose profile, so it may legitimately
  be down. **Fail fast and loudly** with a message naming the profile; do not hang on a
  connection timeout.

### Replay mode

The model needs weather for the **target** hour. In production that is a weather
*forecast*; this repo has only historical Open-Meteo *observations*. So the DAG runs in
replay: `AMPOPS_SIMULATED_TODAY` (default: the latest seeded date) defines "today", and
the DAG forecasts the following 24 hours from seeded observations.

**State this plainly in user-facing docs.** A real deployment needs Open-Meteo's
*forecast* API, not the archive API — `openmeteo-requests` is already in
`requirements.txt`, so that path is open. Do not present replay as live forecasting.

---

## 7. Testing requirements

House style is non-negotiable and derived from the existing suite:

- Module docstring explaining *why* the tests exist and which failure mode they catch.
- `from __future__ import annotations` immediately after the docstring.
- **No mocking.** The suite uses real objects; `monkeypatch` only shrinks config or
  redirects MLflow. The lone exception in-repo is a `SimpleNamespace` H2O stand-in.
- Test names are full sentences (`test_no_feature_sees_load_newer_than_horizon`).
- `pytest.importorskip(...)` at module level for heavy deps; never `pytest.mark.skipif`.
- **No `conftest.py` exists — keep it that way.** Fixtures are per-file.
- Ruff: line-length 100, `E,F,I,W,UP`, `E501` ignored.

### Cost controls — the suite must stay fast

- One **session-scoped** fixture trains a deliberately trivial H2O model
  (`AUTOML_MAX_MODELS=1`, `AUTOML_MAX_RUNTIME_SECS=20` via `monkeypatch.setattr` on
  `config`, which the modules read live) and logs it to `file:{tmp_path}/mlruns` with
  `MLFLOW_ALLOW_FILE_STORE=true`. Reuse it across every API test.
- Unit tests use `ParquetFeatureStore` — no Redis, no Docker.
- `RedisFeatureStore` tests skip via a connection probe when Redis is not up.
- Never run a full AutoML search in the serving tests.

### The parity test is the crown jewel

```python
# tests/test_serving_features.py
pd.testing.assert_frame_equal(api_frame[FEATURES], training_row[FEATURES])
```

Take real rows from `train.parquet`, seed a `ParquetFeatureStore` from
`joined_hourly.parquet`, ask the API's frame builder for those timestamps, and assert
all 49 features match exactly — dtypes included. Skip when the parquets are absent
(they are gitignored). Cover: a summer timestamp, a winter timestamp (different DST
offset), a US holiday, and a 24h batch.

If this passes, training/serving skew is ruled out empirically rather than by
inspection. **Do not weaken it** — it and the Redis dtype map are the only defenses
against the missing MLflow signature.

---

## 8. Verified facts

Measured against the real artifacts on 2026-08-12. Trust these over re-derivation.

| Fact | Value |
|---|---|
| `train.parquet` | 56,529 × 51 → 49 features |
| `joined_hourly.parquet` | 66,493 × 32 (`time` + `COMED_MW` + 30 weather) |
| Feature split | 30 weather + 13 calendar + 6 lag/rolling |
| `joined` time range | 2011-01-01 01:00 → 2018-08-02 23:00 |
| `train` time range | 2011-01-09 00:00 → 2017-08-02 23:00 |
| Minimum history for 0 nulls | **191h** (192h used, 1h margin); 168h leaves 2 nulls |
| Window parity | Verified PASS for summer, winter, July-4 holiday, and a 24h batch — 49/49 features equal, dtypes equal |
| `add_lag_features` cost | ~0.01s for a 193-row window |
| `add_calendar_features` cost | ~0.04s warm |
| `import h2o` cost | **0.4s** warm |

**Cold-start caveat, measured twice:** the *first* execution of a fresh Python process on
this machine can pay a large one-time cost (a 193-row `add_calendar_features` was once
observed at 44s, and `import h2o` was once reported as "several minutes"). Both are ~0.04s
and ~0.4s on every subsequent run. This is an OS/filesystem cold-cache artifact, **not** a
real cost in either the API or the test suite — do not design around it, and do not report
a slow first run as a performance defect without re-running it warm.

**Startup warm-up:** because that first `holidays` load is one-time and can be slow on a
cold process, the lifespan startup handler should build one throwaway inference frame
after loading the model. This warms both the `holidays` cache and the H2O connection so
the first real request is not an outlier.

---

## 9. Environment variables

| Var | Default | Purpose |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Feature store + forecast cache |
| `AMPOPS_STORE_BACKEND` | `redis` | `redis` \| `parquet` — selects the backend pair (§4d) |
| `AMPOPS_PARQUET_PATH` | `config.JOINED_PARQUET` | Source for `ParquetFeatureStore` in local mode |
| `AMPOPS_SERVING_MODEL_URI` | *(unset)* | Explicit model URI override |
| `AMPOPS_MODEL_NAME` | `ampops-demand-forecaster` | Registry name (existing) |
| `AMPOPS_GRID_ID` | `COMED` | Default grid id |
| `AMPOPS_API_URL` | `http://api:8000` | Where the forecast DAG calls |
| `AMPOPS_SIMULATED_TODAY` | latest seeded date | Replay-mode "today" |
| `AIRFLOW_CONN_AMPOPS_DB` | `postgresql://airflow:airflow@postgres/ampops` | DAG's Postgres connection |
| `MLFLOW_TRACKING_URI` | `databricks` | Existing |
| `MLFLOW_REGISTRY_URI` | `databricks-uc` | Existing |

---

## 10. Durability boundary

`make airflow-reset` is `docker compose down -v` and destroys `postgres-data`,
`redis-data`, and every named volume. Postgres is therefore durable against **restarts**,
not against a reset. `make forecast-export` produces the only host-side copy that
survives one. Replay mode makes forecasts deterministically re-derivable, so this is an
acceptable boundary — it just must not be mistaken for permanence.
