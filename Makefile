.PHONY: setup lint test run-api dvc-init docker-up docker-down \
        airflow-up airflow-down airflow-logs airflow-reset dag-test \
        pipeline-local train

# --- Local development ------------------------------------------------------

setup:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	ruff check .

# --cov=app dropped: the app/ package does not exist yet, and pytest-cov errors
# on a missing source. Re-add it when the serving workstream lands.
test:
	pytest -v

# Run the data stages outside Airflow — the fastest way to check a change to
# the ampops package without waiting on the scheduler.
# Uses the active interpreter (conda activate ampops, or .venv after make setup).
pipeline-local:
	python scripts/run_pipeline_local.py

# Bake-off + optional registry + sealed test holdout. Logs eval_name, metrics,
# and duration to whatever MLFLOW_TRACKING_URI points at (Databricks when
# MLFLOW_TRACKING_URI=databricks in .env). Requires `make pipeline-local` first.
train:
	python scripts/run_training.py

# --- Airflow stack ----------------------------------------------------------

airflow-up:
	docker compose up -d --build postgres mlflow airflow-init airflow-scheduler airflow-webserver
	@echo "Airflow UI -> http://localhost:8080  (admin/admin)"
	@echo "MLflow UI  -> http://localhost:5050"

airflow-down:
	docker compose down

# Wipes the Airflow metadata DB and all MLflow runs. Destructive.
airflow-reset:
	docker compose down -v

airflow-logs:
	docker compose logs -f airflow-scheduler

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

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose --profile serving up --build

docker-down:
	docker compose --profile serving down -v

dvc-init:
	dvc init
	dvc remote add -d storage $$DVC_REMOTE_URL
