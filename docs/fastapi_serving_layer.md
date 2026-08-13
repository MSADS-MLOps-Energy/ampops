# FastAPI serving layer

**Status: built and validated end-to-end in the local Docker stack (2026-08-12).**
Champion `ampops-demand-forecaster` v5 (`drf`) loads via `@champion`, the daily
forecast DAG runs and persists to Postgres, and the host test suite passes
106/5 skipped, ruff clean.

**Audience:** anyone running, extending, or debugging `app/`,
`dags/ampops_daily_forecast.py`, or `scripts/seed_redis.py`.

This document explains the *why* behind the implementation — architecture,
lifecycle rules, and traps hit while validating it. The binding spec — the
49-column feature schema, endpoint contracts, storage key layout — lives in
`docs/serving_contract.md`; this document references it rather than repeating
it. If the two ever disagree, `serving_contract.md` wins and the code (or this
doc) is wrong.

---

## Why this exists, and the one design decision everything follows from

Training produces a champion model whose only contract with the world is
"49 columns in, one MW value out," with **no MLflow signature** enforcing
that contract (`docs/serving_contract.md` §1). Every design choice in `app/`
exists to keep serving from silently drifting from what training actually fed
the model — because nothing downstream of the model would ever raise if it
did; predictions would simply degrade in a way that looks plausible.

That produces the anti-skew design: **`app/features.py` computes zero
features.** It fetches a raw window from a feature store and hands it straight
to `ampops.features.build.add_calendar_features` / `add_lag_features` /
`feature_columns` — the exact functions `scripts/run_training.py` used to
build `train.parquet`. If a feature definition changes upstream, serving
changes with it automatically; there is no second implementation to remember
to update. `tests/test_serving_features.py` is the proof: it takes real rows
out of `train.parquet`, seeds a `ParquetFeatureStore` from
`joined_hourly.parquet`, asks the API to build the same rows, and asserts all
49 columns match byte-for-byte, dtypes included.

## Request flow

```
client
  │
  ▼
POST /predict  or  POST /predict/batch
  │
  ├─ /predict: check forecast cache (Redis hash) → hit → return, ~1.3ms
  │             miss → live path below, do NOT write back
  │
  └─ /predict/batch: always live, always writes through
      │
      ▼
  app.features.build_inference_frame(store, grid_id, targets)
      │  1. store.get_window(): one window spanning min(targets)-192h .. max(targets)
      │     (Redis: 1 HMGET for load + 1 pipelined HGETALL batch for weather —
      │      two round trips regardless of window length)
      │  2. blank COMED_MW at every target timestamp (NaN, not 0 — see below)
      │  3. ampops.features.build.add_calendar_features + add_lag_features
      │  4. slice down to feature_columns() = the 49 training columns, in order
      ▼
  app.model.ChampionModel.predict(frame)
      │  h2o.H2OFrame(frame)  ← the pandas→JVM round trip that costs ~300ms
      │  model.predict(...).as_data_frame()["predict"]
      ▼
  /predict/batch also: app.state.cache.put_many(...) — write-through
  response: {grid_id, timestamp(s), predicted_mw, model_version, source, latency_ms}
```

`/predict/batch` builds **one** window and issues **one** `H2OFrame` round
trip for the whole batch — looping the single-prediction path internally
would pay the JVM crossing N times and recompute 192 hours of lag/rolling
features N times over. The forecast DAG relies on this: 24 hourly timestamps
become one batch call, not 24 sequential ones.

## Anti-skew design in `app/features.py`

Two rules that read like bugs at first glance and are not (see the module
docstring in `app/features.py` for the full reasoning):

1. **`build_features()` (the training-time convenience wrapper) is never
   called.** Its trailing `dropna(subset=[...load_*])` would discard exactly
   the row being predicted, since that row's target is blank by construction.
2. **`COMED_MW` is `NaN` at every target timestamp, and that's correct.**
   Every feature the model uses references `t-24` or older (the horizon rule
   `ampops.features.build` enforces), so nothing ever reads the blanked value
   — but the row must still occupy its slot in the hourly grid, or the
   lag/rolling shifts land one hour off for every row after it. The frame is
   cast back to `float64` after blanking because `df.loc[mask, col] = pd.NA`
   on a float column can silently flip it to `object`, which breaks the
   shifts inside `add_lag_features`.

## H2O lifecycle: why single-worker is mandatory

`app/model.py::ChampionModel` owns the entire JVM lifecycle for the process:

- `h2o.init()` runs **once**, in the FastAPI lifespan startup handler
  (`create_app()`'s `lifespan`), on a fresh OS-assigned free port
  (`_free_port()`, imported — not copied — from `ampops.training.automl`).
- The cluster is shut down in lifespan shutdown, **never** in a request
  handler. `automl.py` shuts down in a `finally` block, which is correct for
  a one-shot Airflow task and would be fatal for a server that has to answer
  the next request.
- The model is loaded with `mlflow.h2o.load_model`, **not**
  `mlflow.pyfunc.load_model`. The pyfunc wrapper calls `h2o.init()` itself
  using whatever port the artifact's `h2o.yaml` recorded — typically the
  default 54321, with no override — which is exactly the stale-cluster
  hazard `automl.py`'s docstring warns about, and it would bite on every
  container restart.
- **uvicorn is pinned to a single worker** (`--workers 1` in the Dockerfile
  CMD, and `make run-api` relies on `--reload` implying one worker). This is
  a hard constraint, not a tuning knob: every worker would start its own JVM
  and race for the same port. Scale by running more container replicas
  behind a load balancer, never by adding uvicorn workers.
- A failed model load does **not** crash the process. `register_champion()`
  soft-fails on workspaces without a usable Unity Catalog, so `@champion`
  legitimately may not resolve in some environments. On load failure the
  process starts anyway, `/health` stays green, and `/ready` reports 503
  naming the failed check plus the underlying exception — a crashed
  container gives an operator far less signal than that.
- The lifespan handler also calls `_warm_up()` after loading the model: it
  builds one throwaway inference frame against the latest seeded timestamp.
  This pays the one-time `holidays` table load and the first `H2OFrame`
  handshake before any real client request arrives, so request #1 isn't an
  outlier (`serving_contract.md` §8 documents a cold-process `holidays` load
  once observed at 44s vs ~0.04s warm).

## The two storage tiers

| Tier | Backend | Role | Written by |
|---|---|---|---|
| Feature store | Redis (`RedisFeatureStore`) or in-process parquet (`ParquetFeatureStore`) | Read-only source of the raw `time + COMED_MW + 30 weather` window | `scripts/seed_redis.py` (Redis) / loaded once at startup (parquet) |
| Forecast cache | Redis (`RedisForecastCache`) or in-memory (`InMemoryForecastCache`) | The committed operational forecast — write-through | `POST /predict/batch`, read by `/predict` and `GET /forecast` |
| System of record | Postgres `ampops.forecasts` | Durable, queryable history of every scheduled batch | `dags/ampops_daily_forecast.py`'s `persist` task, only |

Backend selection (`app/config.py::resolve_store_backend`) happens once, in
`create_app()`, and nowhere else in the app branches on it — see
`serving_contract.md` §4d for the exact resolution order. Locally
(`make run-api`) it's the parquet pair, so the test suite and host dev need
neither Redis nor Docker. Compose sets `AMPOPS_STORE_BACKEND=redis` explicitly.

Independence between the three tiers was verified directly, not assumed:
deleting only the Redis forecast-cache keys made `/forecast` return 404 while
Postgres still held all 24 rows and the feature store was untouched. A
subsequent live recomputation for one of those hours returned a value
byte-identical to what had been cached before deletion — confirming no
cache/live drift, i.e. the write-through path and the live-inference path
compute the same thing.

Redis has no types — everything round-trips as a string — so `app/store.py`'s
`_apply_weather_dtypes` (the §1a dtype map in the contract) is the only thing
standing between a Redis read and H2O silently reinterpreting an `int64`
weather column as `float64` (or worse, a numeric column as `enum`). Both
`RedisFeatureStore` and `ParquetFeatureStore` apply the same map, which is
what makes the two backends interchangeable by construction rather than by
hope.

## The daily forecast DAG and replay mode

`dags/ampops_daily_forecast.py`: `resolve_horizon → request_forecast →
persist → verify_cached`, mirroring the thin-task style of
`dags/ampops_training_pipeline.py`. It talks to the serving API over HTTP and
to Postgres over SQL — **it must never import `h2o`, `mlflow`,
`ampops.features`, or `ampops.training`**, because inference happens in
exactly one place (`POST /predict/batch`) and a second in-DAG model load
would be a parallel code path free to drift from what the API actually
serves. `tests/test_forecast_dag.py` enforces this by inspecting the DAG
file's AST, not just by convention.

**Replay mode.** The model needs weather for the *target* hour; this repo
only has historical Open-Meteo *observations*, not forecasts. So
`AMPOPS_SIMULATED_TODAY` (default: the day before the dataset's last seeded
hour) stands in for "today," and the DAG requests the following 24 hours from
already-seeded observations. This is a faithful rehearsal of the operational
loop but **is not live forecasting** — a real deployment needs Open-Meteo's
forecast API (`openmeteo-requests` is already a dependency) feeding the
feature store with actual forecasts, not archive data.

Idempotency and versioning both come from the Postgres composite primary key
`(grid_id, target_ts, model_version)` with `ON CONFLICT ... DO UPDATE`:
re-running the DAG for the same day updates the same 24 rows rather than
duplicating them (verified: a repeat run left 24 rows, not 48), and two
different model versions can hold forecasts for the same hour, which is what
a future champion/challenger comparison needs.

Validated end-to-end: all 4 DAG tasks succeeded, `/forecast` returned 24
entries for `2018-08-02`, and Postgres held 24 matching rows.

## Endpoints

Full request/response shapes are in `serving_contract.md` §5. Summary:

| Endpoint | Purpose | Notes |
|---|---|---|
| `POST /predict` | Single-hour forecast | Cache-first; a live miss is answered but not written back |
| `POST /predict/batch` | Multi-hour forecast | One window, one JVM round trip, writes through to the cache |
| `GET /forecast?grid_id=&date=` | Read the committed forecast for a day | Read-only, never runs inference; 404 if nothing cached |
| `GET /health` | Liveness | Touches nothing external; always 200 if the process is up |
| `GET /ready` | Readiness | 200 only if model + H2O cluster + feature store are all healthy, else 503 naming the failure |
| `GET /metrics` | Prometheus scrape target | Default instrumentator metrics + prediction-latency histogram (by source), prediction-value histogram, cache hit/miss counter |

`/health` vs `/ready` is a deliberate split: `/health` passing while `/ready`
fails is the diagnostic signal that the champion did not load.

## Running it

### Locally (no Docker, no Redis)

```bash
make run-api
```

Uses `AMPOPS_STORE_BACKEND=parquet` against `data/processed/joined_hourly.parquet`
directly — no seeding step needed, no Redis. `--reload` implies a single
uvicorn worker, satisfying the H2O constraint above. Needs the same local
Java 8–17 setup as AutoML training (`brew install openjdk@17` + `PATH`, see
`CLAUDE.md`).

### In Docker (the validated path)

The serving stack sits behind the `serving` compose profile so the training
stack (postgres/mlflow/airflow) can come up without paying for a JVM, a
Redis, and two dashboards:

```bash
make docker-up          # docker compose --profile serving up --build
make seed-redis          # loads joined_hourly.parquet into Redis; required once —
                          # an unseeded store 422s every request
make forecast-trigger    # runs the daily forecast DAG against the live API
make forecast-export     # Postgres forecasts -> data/processed/forecasts.csv
                          # (the only copy that survives `make airflow-reset`)
```

`docker-up` builds `ampops-api:latest` (4.07GB — Java 17.0.20 + h2o 3.46.0.11
+ the full requirements.txt) and starts `api`, `redis`, `prometheus`,
`grafana` alongside `postgres`/`mlflow`/Airflow. `GET /ready` on a healthy
stack:

```json
{"ready":true,"model_loaded":true,"h2o_cluster":true,"feature_store":true,
 "model_uri":"models:/ampops-demand-forecaster@champion","model_version":"5"}
```

### Measured latency

From the local Docker stack, not asserted:

- **Cached read (`/predict` hit, or `/forecast`): ~1.3 ms.**
- **Live inference (`/predict` miss, `/predict/batch`): ~296–530 ms**, 852 ms
  observed on the first cold call before warm-up. The live path's cost is the
  pandas→`H2OFrame` conversion — a REST round trip into the JVM — which is
  exactly why the precomputed forecast cache exists: the daily DAG pays this
  cost once for 24 hours, and every subsequent read of that day is the ~1.3ms
  path.

Prediction sanity check: 2018-07-15 14:00 predicted 15,792.4 MW against an
actual of 16,558.0 MW (APE 4.62%), consistent with the champion's validation
MAPE of 0.042276.

## Troubleshooting

**`/ready` returns 503.** Read the `detail` field — it names which of
`model`, `h2o_cluster`, `feature_store` failed, plus the underlying exception
for a model load failure. Common causes:

- `@champion` doesn't resolve → check `MLFLOW_REGISTRY_URI` is actually set
  (see the trap below — `:-databricks-uc` silently wins if the `.env` line is
  merely commented out).
- Feature store unhealthy in Redis mode → `make seed-redis` hasn't been run,
  or Redis isn't up (`docker compose --profile serving up -d redis`).
- H2O cluster failed to start → check for a Java version mismatch (needs
  8–17, not 21) or a lingering JVM occupying the port.

**A request 422s.** Either an `InsufficientHistoryError` (the requested
timestamp, or the 192h window behind it, falls outside the seeded range —
`serving_contract.md` §2), or the timestamp lands on one of the dataset's 10
missing hours (all DST fall-back artifacts, listed in the contract §2).
Requesting a missing hour intentionally fails the *whole* batch rather than
silently degrading every other row's weather dtypes.

**Forecast DAG fails to reach the API.** The `api` service lives behind the
`serving` compose profile and may legitimately not be running — the DAG's
`request_forecast`/`verify_cached` tasks fail fast with a message naming the
profile rather than hanging out a connection timeout. Start it with
`docker compose --profile serving up -d api redis`.

**Known traps hit during validation** (see also `CLAUDE.md`'s decision log):

1. `python:3.11-slim` now resolves to Debian trixie, which dropped OpenJDK 17
   (only 21/25 available). H2O 3.46.x tops out at Java 17, and 21 builds
   cleanly but fails at model *load* time, not build time — a slow, confusing
   failure to debug from the symptom alone. Fixed by pinning
   `python:3.11-slim-bookworm` in the root `Dockerfile`.
2. `MLFLOW_REGISTRY_URI` silently reverts to `databricks-uc` if merely
   commented out in `.env` — `docker-compose.yml` uses
   `${MLFLOW_REGISTRY_URI:-databricks-uc}`, and `:-` substitutes the default
   for both "unset" and "empty." Set it explicitly for local serving:
   `MLFLOW_REGISTRY_URI=http://mlflow:5000`.
3. The VirtioFS `errno 35` deadlock (`docs/virtiofs_errno35_deadlock.md`) is
   broader than its original data-only writeup: during validation it also
   hit `app/`+`src/` bind mounts (a Python import failing in the api
   container) and `monitoring/prometheus.yml`. Workaround: host-side
   `cp -p f f.new && mv -f f.new f` to mint a fresh inode for the poisoned
   path.
4. Test dates must be checked across the *whole* `[t-192h, t+23h]` span, not
   just the target day — a date that looks fine can still read a DST gap
   through its own `load_lag_168h` a week later.

## Known gaps / future work

- **Monitoring is half-built, and the halves are worth separating.** *Built:*
  `prometheus-fastapi-instrumentator` defaults plus three purpose-chosen series
  (`ampops_prediction_latency_seconds` labelled by `source`,
  `ampops_prediction_mw`, `ampops_forecast_cache_events_total`), scraped every
  15s by `monitoring/prometheus.yml`, with Prometheus and Grafana both in the
  `serving` profile. *Not built:* Grafana ships no datasource or dashboard JSON
  (a fresh container opens empty), and there is no drift detection, no actuals
  source, no prediction-vs-actual scoring job, and no retrain trigger.
- **No inference inputs are persisted anywhere, which blocks input-drift work
  entirely.** `build_inference_frame` assembles 49 columns in memory, hands them
  to H2O, and drops them; the only trace a request leaves is a bucketed
  observation in `ampops_prediction_mw` (the *output*) and, for batch calls, the
  predicted value in `ampops.forecasts`. Any drift tooling — Evidently, a
  hand-rolled PSI job, or Grafana panels over per-feature histograms — needs the
  input distribution, so a feature-logging sink is the prerequisite for all
  three, not a detail of one.
- Ad-hoc `/predict` calls are deliberately not persisted to Postgres — only
  scheduled batches are, to keep the accuracy metric the retrain webhook will
  key off unpolluted by exploratory traffic.
- Replay mode is a stand-in for a real weather-forecast feed; switching to
  live forecasting means pointing `openmeteo-requests` at Open-Meteo's
  forecast API instead of the archive API and feeding results into the
  feature store on a schedule.
- Databricks-backed end-to-end validation (as opposed to local compose) is
  still pending a workspace URL; the `.env` switch is already staged and
  commented out.
