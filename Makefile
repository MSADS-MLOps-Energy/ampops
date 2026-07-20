.PHONY: setup lint test run-api dvc-init docker-up docker-down

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -v --cov=app

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dvc-init:
	dvc init
	dvc remote add -d storage $$DVC_REMOTE_URL

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
