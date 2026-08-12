# AmpOps

**An automated, reproducible short-term load forecasting platform for smart grid reliability.**

Predicting electricity demand before the grid has to guess.

---

## Overview

Modern electric grids operate under strict frequency balance requirements, and system operators must continuously adjust supply to meet volatile demand. Over-estimating demand forces grids to run expensive, carbon-intensive "peaker" plants; under-estimating demand risks frequency dropouts and blackouts.

AmpOps automates the end-to-end machine learning lifecycle for short-term load forecasting: it ingests raw smart meter and weather telemetry, versions feature data, tracks challenger models, serves day-ahead forecasts from a containerized FastAPI service, and continuously monitors for sensor failure or data drift.

Serving latency is measured, not asserted: a scheduled daily batch precomputes the next operating day's 24 hourly forecasts, so cached reads return in roughly **1 ms**, while an on-demand prediction that misses the cache runs live H2O inference at roughly **300 ms** (a pandas frame has to cross into the JVM and back). Both numbers are from the local Docker stack — see `docs/serving_contract.md`.

## Data Sources

- **[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)** — 10+ years of hourly electricity consumption (MW) across major US regional transmission grids. AmpOps uses the ComEd zone (`COMED_hourly.csv`).
- **[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)** — hourly weather for the Chicago point (41.86°N, −87.65°W), 2010–2019.

## Architecture

```
Upstream Raw Data Ingestion (PJM Grid + Open-Meteo Weather API)
              │
              ▼
   1. Ingest / clean / join / features   (src/ampops)
              │
              ▼
   2. Airflow DAG or local scripts       (orchestration)
              │
              ▼
   3. H2O AutoML search + MLflow         (Databricks or local tracking)
              │
              ▼
   4. Model Registry (@champion)         (Unity Catalog when configured)
              │
              ▼
   5. FastAPI + daily forecast DAG       (implemented)
              │
              ▼
   6. Monitoring / drift / retrain       (stub)
```

**Pipeline stages (implemented):**

1. **Data Ingestion & Features** — Raw CSVs in `data/raw/` are cleaned (DST realignment, dedup), joined, and feature-engineered (calendar + horizon-safe lags). Chronological split: final **12 months** sealed as `test.parquet`.
2. **AutoML & Experiment Tracking** — Airflow (or `make train`) runs an **H2O AutoML** search over algorithms and hyperparameters, logs the leader to MLflow, and scores it on a validation tail. Selection uses `nfolds=0` with an explicit `validation_frame` rather than k-fold, because k-fold would shuffle time-ordered rows and leak future information. Tracking can target **Databricks MLflow** or a local compose MLflow server. Requires Java — see Prerequisites.
3. **Registry & Holdout** — Champion promotion to `models:/…@champion` (Databricks Unity Catalog when a catalog is configured; soft-fails to a `runs:/` URI otherwise so training still completes), followed by a post-registration evaluation against the sealed test set, tagged onto the registered version.
4. **Serving** — FastAPI + H2O inference behind a Redis-backed forecast cache and feature store, orchestrated by a daily forecast DAG that precomputes the next operating day and persists to Postgres. Implemented and validated end-to-end in the local Docker stack — see [Serving](#serving) below. Monitoring (Evidently drift detection, actuals-vs-prediction scoring, retrain webhook) is still a stub under `monitoring/`.

Full Databricks wiring notes: [`docs/databricks_experiment_tracking.md`](docs/databricks_experiment_tracking.md). Full serving write-up: [`docs/fastapi_serving_layer.md`](docs/fastapi_serving_layer.md); binding serving spec: [`docs/serving_contract.md`](docs/serving_contract.md).

## Evaluation

| Split | Purpose | Primary metric |
|---|---|---|
| Validation tail (last 3 months of **train**) | AutoML leader selection | **MAPE** (RMSE, MAE secondary) |
| Sealed test (final 12 months) | Holdout scored after registration | **MAPE** |

Each MLflow run also logs the leader's hyperparameters (`hp.*` params plus a `hyperparams.json` artifact) and wall-clock timing so runs can be compared and re-executed. Test-set metrics are written back onto the registered model version as `test_mape` / `test_rmse` / `test_mae` tags.

## Tech Stack

`Python 3.11` · `H2O AutoML` (Java 17) · `scikit-learn` · `Airflow` · `MLflow` (+ Databricks) · `FastAPI` · `Redis` · `Postgres` · `Prometheus` · `Docker` · `Evidently` (planned) · `Ruff` · `pytest`

## Getting Started

### Prerequisites

- **Python 3.11** (conda env `ampops` or `make setup` → `.venv`)
- **Java 17** — H2O spawns a JVM, so `make train` and the serving API both need it. macOS: `brew install openjdk@17` (keg-only; the Makefile adds it to `PATH` automatically). H2O 3.46.x supports Java 8–17 — **21 is not supported**.
- Docker & Docker Compose (optional — for the full Airflow stack)
- Databricks workspace + PAT (optional — for cloud experiment tracking)

### Getting the data

Datasets are not committed to git. Place both files in `data/raw/` (filenames matter):

| File | Source | Notes |
|---|---|---|
| `COMED_hourly.csv` | [Kaggle: Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | 66,497 rows, 2011-01-01 → 2018-08-03 |
| `open-meteo-41.86N87.65W179m.csv` | [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | Chicago 41.86N/−87.65W, hourly, 2010–2019, fixed UTC−5. Keep the 3-line preamble — `ingest` skips exactly 3 rows. |

Everything under `data/interim/` and `data/processed/` is generated — never commit it.

### Configure tracking

```bash
cp .env.example .env
```

**Databricks (recommended for the course demo):**

```env
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT=/Users/you@uchicago.edu/Ampops
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<pat>
MLFLOW_REGISTRY_URI=databricks-uc
AMPOPS_MODEL_NAME=ampops-demand-forecaster
# AMPOPS_UC_MODEL_PREFIX=catalog.schema   # if your UC catalog is not main.default
```

**Local compose MLflow:** leave `MLFLOW_TRACKING_URI=http://mlflow:5000` (Airflow) / use `http://localhost:5050` in the browser.

### Fast path: local data + train → Databricks

```bash
conda activate ampops          # or: make setup && source .venv/bin/activate
set -a && source .env && set +a

make pipeline-local            # ingest → join → features → train/test split
make train                     # H2O AutoML → register (best-effort) → sealed holdout
```

Useful variants:

```bash
python scripts/run_training.py --skip-register          # search + score, no registry write
python scripts/run_training.py --skip-holdout           # stop after registration
python scripts/run_training.py --train-path <path> --test-path <path>
```

Search budget is env-driven rather than per-flag: `AMPOPS_AUTOML_MAX_RUNTIME_SECS` (default 300) and `AMPOPS_AUTOML_MAX_MODELS` (default 10).

### Airflow stack (Docker)

```bash
cp .env.example .env           # include Databricks vars if Airflow should log there
make airflow-up                # postgres, MLflow, Airflow
```

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | `admin` / `admin` |
| Local MLflow UI | http://localhost:5050 | — (unused if tracking URI is `databricks`) |

Trigger **`ampops_training_pipeline`** in the UI or:

```bash
make dag-trigger
```

DAG shape:

```
ingest_raw → validate_raw → clean_and_join → validate_joined
  → build_features → split_train_test
  → run_automl → register → evaluate_test
```

`make dag-test` runs the DAG in one process (stops the scheduler first to avoid duplicate task execution).

Serving is implemented — see [Serving](#serving) below for endpoints and how to run it (`make docker-up`).

### Airflow without Docker (fallback)

```bash
make setup
make setup-airflow-local
make mlflow-local           # second shell — :5000
make dag-test-local
```

## Serving

The FastAPI serving layer (`app/`) loads the registered champion via
`mlflow.h2o.load_model` and answers day-ahead demand forecasts. Two latency
paths, both measured against the local Docker stack (see the Overview above
and `docs/serving_contract.md` §8 for the full numbers):

- **Cached** (`/predict` hit, or `GET /forecast`) — reads a precomputed value
  out of Redis, ~1 ms.
- **Live** (`/predict` miss, or `/predict/batch`) — runs H2O inference
  directly, ~300 ms; this is the pandas→JVM round trip, which is exactly why
  the daily forecast DAG exists: it pays that cost once for 24 hours so every
  later read of that day is the cached path.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Single-hour forecast, cache-first |
| `POST /predict/batch` | Multi-hour forecast, one JVM round trip, writes through to the cache |
| `GET /forecast?grid_id=&date=` | Read-only view of the committed forecast for a day |
| `GET /health` / `GET /ready` | Liveness / readiness (model + H2O cluster + feature store) |
| `GET /metrics` | Prometheus scrape target |

```bash
make run-api            # host, no Redis/Docker needed (parquet feature-store backend)

make docker-up          # docker compose --profile serving up --build
make seed-redis          # populate the Redis feature store (required once before /predict works)
make forecast-trigger    # run the daily forecast DAG against the live API
make forecast-export     # Postgres forecasts -> data/processed/forecasts.csv
```

Full architecture, the H2O lifecycle rules, the two storage tiers, replay
mode, and troubleshooting: [`docs/fastapi_serving_layer.md`](docs/fastapi_serving_layer.md).
Binding spec (49-column feature schema, storage key layout, endpoint
contracts): [`docs/serving_contract.md`](docs/serving_contract.md).

### Development

```bash
make lint    # Ruff
make test    # pytest
```

## Project Structure

```
ampops/
├── dags/
│   ├── ampops_training_pipeline.py   # ingest -> ... -> register -> evaluate_test
│   └── ampops_daily_forecast.py      # resolve_horizon -> request_forecast -> persist -> verify_cached
├── src/ampops/
│   ├── config.py              # paths, data contract, AutoML + MLflow settings
│   ├── data/                  # ingest, clean (DST), join, validate
│   ├── features/              # calendar + lag features, time split
│   ├── training/              # automl (search + test eval), registry, shared metrics
│   └── utils/
├── app/                        # FastAPI serving layer (implemented)
│   ├── main.py                 # endpoints, lifespan (H2O cluster start/stop)
│   ├── model.py                 # ChampionModel: H2O load + predict lifecycle
│   ├── features.py             # delegates to ampops.features.build — no feature arithmetic here
│   ├── store.py                 # FeatureStore / ForecastCache (Redis + parquet/in-memory backends)
│   ├── schemas.py               # request/response Pydantic models
│   └── config.py                # env-driven Settings, not memoized
├── monitoring/
│   └── prometheus.yml           # scrape config for the api service
├── scripts/
│   ├── run_pipeline_local.py  # data stages without Airflow
│   ├── run_training.py        # AutoML + holdout CLI (make train)
│   └── seed_redis.py           # load joined_hourly.parquet into the Redis feature store
├── notebooks/                 # EDA
├── tests/
├── docs/
│   ├── AmpOps_Project_Context.md
│   ├── timezone_alignment_finding.md
│   ├── data_cleaning_plan.md
│   ├── automl_implementation.md
│   ├── databricks_experiment_tracking.md
│   ├── serving_contract.md            # binding serving spec
│   ├── fastapi_serving_layer.md       # serving architecture + lifecycle + traps
│   └── virtiofs_errno35_deadlock.md
├── data/{raw,interim,processed}/
├── docker-compose.yml
├── requirements.txt · requirements-airflow.txt
└── Makefile
```

## Engineering Notes

Two decisions in the pipeline are worth reading the code for:

**DST realignment** (`src/ampops/data/clean.py`). The weather export is stamped
at a fixed UTC-5 offset that never observes daylight saving; the COMED load
series is DST-aware. Joining them on the raw timestamps misaligns the two
sources by one hour for the whole summer — silently, with no error and no null.

Establishing the *direction* of that offset took real work, and the first
attempt got it backwards. The two datasets are one hour apart in summer and
aligned in winter, so the correction shifts summer rows −1h. Two mechanisms
explain this and both predict the same shift (PJM publishes every zone on
Eastern time, and/or the stamps are hour-ending), so they need not be
adjudicated. The evidence: summer load-vs-temperature coupling peaks at −1h
(0.7925 vs 0.7497 uncorrected), and adopting the correction improved *every*
model in the then-current bake-off (since replaced by the AutoML search). Full write-up in `docs/timezone_alignment_finding.md`.

`validate.check_dst_alignment` guards the transformation on every run by
comparing hour-of-day load profiles against an uncorrected control. Worth
knowing its limit: it proves the transformation ran as intended, not that the
intent was right — the superseded interpretation passed its own version of the
same check.

**No-leakage feature engineering** (`src/ampops/features/build.py`). Forecasts
are day-ahead, so no feature may reference load newer than `t-24`. That rules out
a `t-1` lag and requires every rolling window to be horizon-shifted before it is
computed. `tests/test_features.py` enforces this empirically: it perturbs the
target inside the forbidden window and asserts no feature value moves.

## License

Distributed under the MIT License. See `LICENSE` for details.
