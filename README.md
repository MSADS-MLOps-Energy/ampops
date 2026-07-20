# AmpOps

**An automated, reproducible short-term load forecasting platform for smart grid reliability.**

Predicting electricity demand before the grid has to guess.

---

## Overview

Modern electric grids operate under strict frequency balance requirements, and system operators must continuously adjust supply to meet volatile demand. Over-estimating demand forces grids to run expensive, carbon-intensive "peaker" plants; under-estimating demand risks frequency dropouts and blackouts.

AmpOps automates the end-to-end machine learning lifecycle for short-term load forecasting: it ingests raw smart meter and weather telemetry, versions feature data, tracks challenger models, deploys low-latency containerized inference, and continuously monitors for sensor failure or data drift.

## Team

| Role | Owner | Responsibilities |
|---|---|---|
| Data & Orchestration Lead | Sachin Patel | Data ingestion, DVC tracking, feature engineering, Airflow/Prefect DAGs |
| Model Infrastructure Lead | Minhae Park | Baseline vs. deep learning models, hyperparameter tuning, MLflow tracking/registry |
| Deployment Engineer | Miguel Roca Garcia | Docker packaging, FastAPI endpoint, SLA testing, shadow/canary routing |
| Monitoring & Reliability Lead | Collin Kim | Prometheus/Grafana, drift simulation, continuous training webhook |

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
- Python 3.11+
- Docker & Docker Compose
- An AWS S3 bucket (or other DVC-supported remote) for data versioning

### Setup

```bash
git clone https://github.com/<your-org>/ampops.git
cd ampops

# create env and install dependencies
make setup

# configure environment variables
cp .env.example .env
# fill in DVC remote, MLflow URI, etc.

# initialize data versioning
make dvc-init
```

### Running locally

```bash
# spin up API, Redis, MLflow, Postgres, Prometheus, Grafana
make docker-up

# or run the API alone, outside Docker
make run-api
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Development

```bash
make lint    # Ruff
make test    # pytest with coverage
```

## Project Structure

```
ampops/
├── app/                # FastAPI service
├── dags/               # Airflow/Prefect DAGs
├── data/
│   ├── raw/            # DVC-tracked raw telemetry
│   └── processed/      # DVC-tracked curated features
├── monitoring/         # Prometheus/Grafana config
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── Makefile
```

## License

Distributed under the MIT License. See `LICENSE` for details.