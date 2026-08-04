# AmpOps

**An automated, reproducible short-term load forecasting platform for smart grid reliability.**

Predicting electricity demand before the grid has to guess.

---

## Overview

Modern electric grids operate under strict frequency balance requirements, and system operators must continuously adjust supply to meet volatile demand. Over-estimating demand forces grids to run expensive, carbon-intensive "peaker" plants; under-estimating demand risks frequency dropouts and blackouts.

AmpOps automates the end-to-end machine learning lifecycle for short-term load forecasting: it ingests raw smart meter and weather telemetry, versions feature data, tracks challenger models, deploys low-latency containerized inference, and continuously monitors for sensor failure or data drift.

## Data Sources

- **[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)** — 10+ years of hourly electricity consumption (MW) across major US regional transmission grids.
- **[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)** — hourly ambient temperature, relative humidity, dew point, and precipitation, mapped to grid hubs (e.g., ComEd/Chicago, PJM East/Philadelphia).

## Architecture

```
Upstream Raw Data Ingestion (PJM Grid + Open-Meteo Weather API)
              │
              ▼
   1. Data Tracking & DVC        (raw v1 → curated v2 matrices)
              │
              ▼  (triggers automated DAG)
   2. Airflow / Prefect Pipeline (lags & rolling averages)
              │
              ▼  (logs params, hashes, metrics)
   3. MLflow Model Registry      (champion/challenger metadata)
              │
              ▼  (shadow-deploys containerized candidate)
   4. FastAPI + Redis Cache      (sub-100ms inference SLA)
              │
        ┌─────┴─────────────────┐
        ▼                       ▼
   5. Prometheus/Grafana   6. Retrain Webhook Loop
      (telemetry)             (fires on MAPE/drift > threshold)
```

**Pipeline stages:**

1. **Data Ingestion & Versioning** — Raw hourly telemetry lands in `data/raw/`. DVC tracks two dataset versions: `v1` (uncurated) and `v2` (deduplicated, DST-corrected, with engineered lag features `t-1`, `t-24`, `t-168` and 24-hour rolling means).
2. **Pipeline Automation & Tracking** — An Airflow/Prefect DAG enforces chronological train/holdout splits (final 12 months held out). MLflow logs an XGBoost baseline against a PyTorch LSTM, then registers the winner with semantic versioning.
3. **Containerization & Deployment** — The champion model is served via a Dockerized FastAPI app backed by a Redis feature cache. Clients pass only a `Grid_ID` and timestamp; the API resolves lag vectors from Redis. New challengers are deployed via shadow routing — evaluated in parallel but not yet acted on.
4. **Monitoring & Drift Engineering** — Prometheus scrapes endpoint metrics into Grafana. Simulated drift events (unit-mismatch sensor corruption, decoupled temperature/demand vectors) validate resilience. Sustained MAPE > 5% over a 6-hour window fires a webhook that triggers automated retraining.

## Evaluation

- **RMSE** — penalizes large prediction misses, mirroring the high cost of peak-demand surprises.
- **MAPE**, tracked specifically during **top-10th-percentile peak loads** (e.g., hottest summer afternoons) — the business KPI that matters most, since blackout risk is highest exactly when demand peaks.

## Tech Stack

`Python` · `XGBoost` · `PyTorch` · `Airflow` / `Prefect` · `DVC` · `MLflow` · `FastAPI` · `Redis` · `Docker` · `Prometheus` · `Grafana` · `Ruff`

## Getting Started

### Prerequisites
- Docker & Docker Compose (required — the pipeline runs in containers)
- Python 3.11 (optional, for running tests and the data stages on the host)

### Getting the data

Datasets are not committed to git. Download both files into `data/raw/` before
running the pipeline — the filenames matter, since `src/ampops/config.py`
resolves them by name:

| File | Source | Notes |
|---|---|---|
| `COMED_hourly.csv` | [Kaggle: Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | Take `COMED_hourly.csv` from the archive. 66,497 rows, 2011-01-01 → 2018-08-03. |
| `open-meteo-41.86N87.65W179m.csv` | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Chicago point 41.86N, -87.65W. Hourly variables, 2010-01-01 → 2019-12-31. |

The pipeline validates both on ingest, so a wrong file or truncated download
fails at `validate_raw` with a specific message rather than silently producing a
bad model.

Everything under `data/interim/` and `data/processed/` is generated — never
commit it, and never hand-edit it.

### Reproducing the training pipeline

The whole pipeline runs in Docker. From a fresh clone, with the two raw files in
place:

```bash
git clone https://github.com/<your-org>/ampops.git
cd ampops

cp .env.example .env    # Airflow credentials, MLflow URI
make airflow-up         # builds the image and starts postgres, MLflow, Airflow
```

First build takes a few minutes. When it finishes:

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | `admin` / `admin` |
| MLflow UI | http://localhost:5050 | — |

Open the Airflow UI, find **`ampops_training_pipeline`**, and trigger it with the
▶ button. It runs:

```
ingest_raw → validate_raw → clean_and_join → validate_joined
  → build_features → split_train_test
  → train[linear | random_forest | xgboost]   (dynamically mapped)
  → choose_champion → register
```

On success you'll have `data/processed/{joined_hourly,features,train,test}.parquet`
on the host, three runs in MLflow, and `ampops-demand-forecaster` registered with
the `@champion` alias.

Or trigger the same run from the command line:

```bash
make dag-trigger
```

For DAG development there is also `make dag-test`, which runs the whole DAG in a
single process. It stops the scheduler first, deliberately: `airflow dags test`
writes into the same metadata database the scheduler polls, so a running
scheduler races it and executes every task twice — visible only as duplicate
MLflow runs, never as an error.

Serving and dashboard services (FastAPI, Redis, Prometheus, Grafana) sit behind a
compose profile because they depend on files the deployment and monitoring
workstreams have not written yet:

```bash
make docker-up          # docker compose --profile serving up
```

### Running Airflow without Docker (fallback)

The identical DAG runs in a local Python 3.11 venv — useful if Docker is
unavailable or slow to pull base images:

```bash
make setup                  # venv + dependencies
make setup-airflow-local    # Airflow 2.9.3 under the official constraints
make mlflow-local           # in a second shell — tracking server on :5000
make dag-test-local         # runs the full DAG in one process
```

### Running the data stages without Docker

```bash
make setup                     # Python 3.11 venv + dependencies
make pipeline-local            # ingest → validate → join → features → split
```

This skips training (which needs the MLflow server) but regenerates every
processed dataset, and is the quickest way to check a change to `src/ampops/`.

### Development

```bash
make lint    # Ruff
make test    # pytest
```

## Project Structure

```
ampops/
├── dags/               # Airflow DAG (orchestration only, no business logic)
├── src/ampops/         # the pipeline package
│   ├── config.py       # paths, constants, model configs — single source of truth
│   ├── data/           # ingest, clean (DST realignment), join, validate
│   ├── features/       # calendar + lag feature engineering, chronological split
│   ├── training/       # model bake-off, champion selection, registry promotion
│   └── utils/          # parquet IO, logging
├── scripts/            # run_pipeline_local.py
├── notebooks/          # 01_join_and_eda.ipynb (EDA deliverable)
├── tests/              # pytest: cleaning, features, split, gates, DAG structure
├── app/                # FastAPI service (deployment workstream)
├── monitoring/         # Prometheus/Grafana config (monitoring workstream)
├── data/
│   ├── raw/            # source CSVs
│   ├── interim/        # per-stage intermediates
│   └── processed/      # joined, features, train, test
├── docker/airflow/     # Airflow image
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