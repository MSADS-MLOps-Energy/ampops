# AmpOps

**An automated, reproducible short-term load forecasting platform for smart grid reliability.**

Predicting electricity demand before the grid has to guess.

---

## Overview

Modern electric grids operate under strict frequency balance requirements, and system operators must continuously adjust supply to meet volatile demand. Over-estimating demand forces grids to run expensive, carbon-intensive "peaker" plants; under-estimating demand risks frequency dropouts and blackouts.

AmpOps automates the end-to-end machine learning lifecycle for short-term load forecasting: it ingests raw smart meter and weather telemetry, versions feature data, tracks challenger models, deploys low-latency containerized inference, and continuously monitors for sensor failure or data drift.

## Data Sources

- **[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)** — 10+ years of hourly electricity consumption (MW) across major US regional transmission grids. AmpOps uses the ComEd zone (`COMED_hourly.csv`).
- **[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)** — hourly weather for the Chicago point (41.86°N, −87.65°W), 2010–2019.

## Architecture

```
Upstream Raw Data (COMED + Open-Meteo)
              │
              ▼
   1. Ingest / clean / join / features   (src/ampops)
              │
              ▼
   2. Airflow DAG or local scripts       (orchestration)
              │
              ▼
   3. MLflow bake-off + holdout          (Databricks or local tracking)
              │
              ▼
   4. Model Registry (@champion)         (Unity Catalog when configured)
              │
              ▼
   5. FastAPI + monitoring               (deployment / Week 3)
```

**Pipeline stages (implemented):**

1. **Data Ingestion & Features** — Raw CSVs in `data/raw/` are cleaned (DST realignment, dedup), joined, and feature-engineered (calendar + horizon-safe lags). Chronological split: final **12 months** sealed as `test.parquet`.
2. **Bake-off & Experiment Tracking** — Airflow (or `make train`) trains **linear → random forest → XGBoost**, logs each as an MLflow run (`eval_name=validation_tail`), picks the lowest-MAPE champion, then scores the sealed holdout (`eval_name=test_holdout`). Tracking can target **Databricks MLflow** or a local compose MLflow server.
3. **Registry** — Champion promotion to `models:/…@champion` (Databricks Unity Catalog when a catalog is configured; soft-fails otherwise so training/holdout still complete).
4. **Serving & Monitoring** — FastAPI / Evidently / drift injection (deployment and monitoring workstreams; stubs under `app/` and `monitoring/`).

Full Databricks wiring notes: [`docs/databricks_experiment_tracking.md`](docs/databricks_experiment_tracking.md).

## Evaluation

| Split | Purpose | Primary metric |
|---|---|---|
| Validation tail (last 3 months of **train**) | Model selection in the bake-off | **MAPE** (RMSE, MAE secondary) |
| Sealed test (final 12 months) | Holdout after champion selection | **MAPE** |

Each MLflow run also logs hyperparameters (`n_estimators`, … plus `hp.*` copies and a `hyperparams.json` artifact) and wall-clock timing so runs can be compared and re-executed.

## Tech Stack

`Python 3.11` · `XGBoost` · `scikit-learn` · `Airflow` · `MLflow` (+ Databricks) · `FastAPI` · `Docker` · `Evidently` (planned) · `Ruff` · `pytest`

## Getting Started

### Prerequisites

- **Python 3.11** (conda env `ampops` or `make setup` → `.venv`)
- Docker & Docker Compose (optional — for the full Airflow stack)
- Databricks workspace + PAT (optional — for cloud experiment tracking)

### Getting the data

Datasets are not committed to git. Place both files in `data/raw/` (filenames matter):

| File | Source | Notes |
|---|---|---|
| `COMED_hourly.csv` | [Kaggle: Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | 66,497 rows, 2011-01-01 → 2018-08-03 |
| `open-meteo-41.86N87.65W179m.csv` | Open-Meteo archive, or regenerate locally | Chicago 41.86N/−87.65W, hourly, 2010–2019, fixed UTC−5 |

```bash
# Optional: regenerate the weather file (writes the 3-line preamble ingest expects)
python scripts/download_open_meteo.py
```

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
AMPOPS_MODEL_NAME=ampops_demand_forecaster
# AMPOPS_UC_MODEL_PREFIX=catalog.schema   # if your UC catalog is not main.default
```

**Local compose MLflow:** leave `MLFLOW_TRACKING_URI=http://mlflow:5000` (Airflow) / use `http://localhost:5050` in the browser.

### Fast path: local data + train → Databricks

```bash
conda activate ampops          # or: make setup && source .venv/bin/activate
set -a && source .env && set +a

make pipeline-local            # ingest → join → features → train/test split
make train                     # bake-off → register (best-effort) → sealed holdout
```

Useful variants:

```bash
python scripts/run_training.py --models xgboost
python scripts/run_training.py --skip-register
python scripts/run_training.py --from-run <run_id> --skip-register
python scripts/run_training.py --models xgboost --param n_estimators=300
```

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
  → train[linear | random_forest | xgboost]   (dynamically mapped)
  → choose_champion → register → score_holdout
```

`make dag-test` runs the DAG in one process (stops the scheduler first to avoid duplicate task execution).

Serving / dashboards (when those workstreams land):

```bash
make docker-up          # docker compose --profile serving up
```

### Airflow without Docker (fallback)

```bash
make setup
make setup-airflow-local
make mlflow-local           # second shell — :5000
make dag-test-local
```

### Development

```bash
make lint    # Ruff
make test    # pytest
```

## Project Structure

```
ampops/
├── dags/                      # Airflow DAG (orchestration only)
├── src/ampops/
│   ├── config.py              # paths, MODEL_CONFIGS, metrics
│   ├── data/                  # ingest, clean (DST), join, validate
│   ├── features/              # calendar + lag features, time split
│   ├── training/              # bake-off, holdout eval, registry, pipeline
│   └── utils/
├── scripts/
│   ├── run_pipeline_local.py  # data stages without Airflow
│   ├── run_training.py        # bake-off + holdout CLI (make train)
│   └── download_open_meteo.py
├── notebooks/                 # EDA
├── tests/
├── docs/
│   ├── AmpOps_Project_Context.md
│   ├── timezone_alignment_finding.md
│   └── databricks_experiment_tracking.md
├── app/ · monitoring/         # Week 3 stubs
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
model in the bake-off. Full write-up in `docs/timezone_alignment_finding.md`.

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
