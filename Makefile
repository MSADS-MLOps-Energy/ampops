.PHONY: setup lint test run-api dvc-init docker-up docker-down \
        airflow-up airflow-down airflow-logs airflow-reset dag-test pipeline-local \
        data-export data-import train \
        seed-redis ampops-db-init forecast-trigger forecast-export

# Prefer repo .venv when present; otherwise use whatever `python` is active
# (e.g. conda env ampops). Override with: make train PYTHON=/path/to/python
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

# Homebrew OpenJDK is keg-only; without this, H2O finds the macOS /usr/bin/java stub.
ifneq ($(wildcard /opt/homebrew/opt/openjdk@17/bin/java),)
  export JAVA_HOME := /opt/homebrew/opt/openjdk@17
  export PATH := $(JAVA_HOME)/bin:$(PATH)
else ifneq ($(wildcard /usr/local/opt/openjdk@17/bin/java),)
  export JAVA_HOME := /usr/local/opt/openjdk@17
  export PATH := $(JAVA_HOME)/bin:$(PATH)
endif

# --- Local development ------------------------------------------------------

setup:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -v --cov=app

# Run the data stages outside Airflow — the fastest way to check a change to
# the ampops package without waiting on the scheduler.
pipeline-local:
	$(PYTHON) scripts/run_pipeline_local.py

# Host-side H2O AutoML → Databricks MLflow (needs Java 8–17 + h2o + .env creds).
# Loads `.env` when present so DATABRICKS_* / MLFLOW_* are set for the child process.
train:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	$(PYTHON) scripts/run_training.py

# --- Airflow stack ----------------------------------------------------------

airflow-up:
	docker compose up -d --build postgres mlflow airflow-init airflow-scheduler airflow-webserver
	@echo "Airflow UI -> http://localhost:8080  (admin/admin)"
	@echo "MLflow UI  -> http://localhost:5050"

airflow-down:
	docker compose down

# Wipes the Airflow metadata DB, all MLflow runs, every task log, and the
# generated parquets in data/interim + data/processed (all named volumes now).
# Destructive. Raw CSVs survive — data/raw is a host bind mount.
airflow-reset:
	docker compose down -v

airflow-logs:
	docker compose logs -f airflow-scheduler

# --- Moving data in and out of the volumes -----------------------------------
#
# data/interim and data/processed live in named volumes rather than on the host,
# to keep the DAG's fixed rewrite paths off the VirtioFS bind mount (see
# docs/virtiofs_errno35_deadlock.md). These targets move files across that
# boundary with `docker compose cp`, which streams over the Docker API and never
# touches VirtioFS. data/raw needs no such step — it is still bind-mounted.

# Pull the generated parquets out for notebooks and inspection.
# Overwrites whatever is in ./data/processed on the host.
data-export:
	docker compose cp airflow-scheduler:/opt/airflow/data/processed/. ./data/processed/
	@echo "Exported -> ./data/processed"

# Push host parquets in — e.g. to run the training tasks against an existing
# train.parquet without re-running the data stages first.
data-import:
	docker compose cp ./data/processed/. airflow-scheduler:/opt/airflow/data/processed/
	@echo "Imported -> ampops-data-processed volume"

# Trigger a real run through the scheduler. This is what the ▶ button in the
# UI does, and the right way to produce a run for the demo.
dag-trigger:
	docker compose exec airflow-scheduler \
		airflow dags trigger ampops_training_pipeline

# Run the whole DAG in one process. Faster inner loop for DAG development.
#
# WARNING: stop the scheduler first (`docker compose stop airflow-scheduler`).
# `dags test` writes task instances into the same metadata DB the scheduler is
# polling, so a live scheduler races it and executes every task a second time —
# which shows up as duplicate MLflow runs, not as an obvious error.
dag-test:
	docker compose stop airflow-scheduler
	docker compose run --rm airflow-scheduler \
		airflow dags test ampops_training_pipeline $(shell date +%Y-%m-%d)
	docker compose start airflow-scheduler

# --- Airflow without Docker (fallback) --------------------------------------
#
# Runs the identical DAG in the local venv. Useful when Docker is unavailable
# or slow to pull images. Needs `make setup-airflow-local` once, and an MLflow
# server on :5000 (`make mlflow-local` in another shell).

AIRFLOW_LOCAL_ENV = AIRFLOW_HOME=$(PWD)/.airflow \
	PYTHONPATH=$(PWD)/src \
	AIRFLOW__CORE__DAGS_FOLDER=$(PWD)/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000

setup-airflow-local:
	. .venv/bin/activate && pip install "apache-airflow==2.9.3" \
		--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
	$(AIRFLOW_LOCAL_ENV) .venv/bin/airflow db migrate

mlflow-local:
	.venv/bin/mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///$(PWD)/.mlflow/mlflow.db \
		--artifacts-destination $(PWD)/.mlflow/artifacts --serve-artifacts

dag-test-local:
	$(AIRFLOW_LOCAL_ENV) .venv/bin/airflow dags test \
		ampops_training_pipeline $(shell date +%Y-%m-%d)

# --- Serving stack (deployment workstream) ----------------------------------

# --reload implies a single worker, which is what the API requires anyway: each
# worker starts its own H2O JVM and they would race on the port (app/model.py).
# AMPOPS_STORE_BACKEND=parquet is the Redis-free host pair.
run-api:
	AMPOPS_STORE_BACKEND=parquet PYTHONPATH=$(PWD)/src \
		$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose --profile serving up --build

docker-down:
	docker compose --profile serving down -v

# Load joined_hourly.parquet into the Redis feature store. Required before the
# API can answer anything — an unseeded store returns 422 for every timestamp.
# Runs on the host against the published Redis port.
seed-redis:
	REDIS_URL=$${REDIS_URL:-redis://localhost:6379/0} \
		$(PYTHON) scripts/seed_redis.py

# Create the `ampops` database on an EXISTING postgres-data volume. Idempotent.
# docker/postgres/init-ampops-db.sh covers the fresh-volume case; postgres:15
# only runs initdb scripts the first time a volume is initialized, which is why
# both exist.
ampops-db-init:
	docker compose exec -T postgres sh -c \
		"psql -U airflow -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname='ampops'\" | grep -q 1 \
		 || psql -U airflow -d postgres -c 'CREATE DATABASE ampops'"
	@echo "ampops database ready"

# Trigger the daily forecast DAG. Needs the serving profile up (`make docker-up`)
# — the DAG fails fast and says so if the api service is not reachable.
forecast-trigger:
	docker compose exec airflow-scheduler \
		airflow dags trigger ampops_daily_forecast

# Postgres lives in a named volume, so `make airflow-reset` (compose down -v)
# destroys it. This CSV is the only host-side copy that survives one.
forecast-export:
	@mkdir -p data/processed
	docker compose exec -T postgres \
		psql -U airflow -d ampops -c \
		"COPY (SELECT * FROM forecasts ORDER BY target_ts) TO STDOUT WITH CSV HEADER" \
		> data/processed/forecasts.csv
	@echo "Exported -> data/processed/forecasts.csv"

dvc-init:
	dvc init
	dvc remote add -d storage $$DVC_REMOTE_URL
