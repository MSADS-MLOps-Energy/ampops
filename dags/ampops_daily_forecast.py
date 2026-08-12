"""AmpOps daily forecast pipeline.

    resolve_horizon -> request_forecast -> persist -> verify_cached

Orchestration only, like `dags/ampops_training_pipeline.py`: every task is a
thin call out to something else — here, HTTP to the serving API and SQL to
Postgres.

**This DAG must never import `h2o`, `mlflow`, `ampops.features`, or
`ampops.training`.** Inference happens in exactly one place, behind
`POST /predict/batch`; a second in-DAG model load would be a parallel code path
free to drift from what the API actually serves, and no test would notice.
`tests/test_forecast_dag.py` enforces this by inspecting this file's AST.

**Replay mode.** The model needs weather for the *target* hour. In production
that is a weather forecast; this repo has only historical Open-Meteo
observations, so `AMPOPS_SIMULATED_TODAY` defines "today" and the DAG forecasts
the following 24 hours out of seeded observations. This is a faithful rehearsal
of the operational loop, **not** live forecasting — a real deployment needs
Open-Meteo's forecast API (`openmeteo-requests`, already a dependency) feeding
the feature store.

Postgres is reached through `AIRFLOW_CONN_AMPOPS_DB`, so there is no manual
Airflow-UI connection setup. The `ampops` database is created by
`docker/postgres/init-ampops-db.sh` on a fresh volume, or by `make
ampops-db-init` on an existing one.
"""

from __future__ import annotations

import os

import pendulum
import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.providers.postgres.hooks.postgres import PostgresHook

# The joined dataset ends 2018-08-02 23:00, so 2018-08-01 is the last "today"
# whose following 24 hours are fully seeded. Override with AMPOPS_SIMULATED_TODAY.
DEFAULT_SIMULATED_TODAY = "2018-08-01"

FORECAST_HOURS = 24
API_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
POSTGRES_CONN_ID = "ampops_db"

# The api service sits behind the "serving" compose profile, so it may
# legitimately not be running. Fail fast and name the profile rather than
# waiting out a connect timeout with an opaque error.
REQUEST_TIMEOUT_SECONDS = 120

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
  grid_id        text             NOT NULL,
  target_ts      timestamptz      NOT NULL,
  predicted_mw   double precision NOT NULL,
  model_version  text             NOT NULL,
  generated_at   timestamptz      NOT NULL DEFAULT now(),
  run_id         text,
  PRIMARY KEY (grid_id, target_ts, model_version)
);
"""

# The composite primary key is what makes a re-run idempotent, and what lets
# two model versions hold a forecast for the same hour — which is exactly the
# comparison champion/challenger evaluation needs.
UPSERT_SQL = """
INSERT INTO forecasts (grid_id, target_ts, predicted_mw, model_version, run_id)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (grid_id, target_ts, model_version)
DO UPDATE SET predicted_mw = EXCLUDED.predicted_mw,
              generated_at = now(),
              run_id       = EXCLUDED.run_id;
"""

DEFAULT_ARGS = {
    "owner": "miguel",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=2),
}


def _api_url() -> str:
    return os.getenv("AMPOPS_API_URL", "http://api:8000").rstrip("/")


def _grid_id() -> str:
    return os.getenv("AMPOPS_GRID_ID", "COMED")


@dag(
    dag_id="ampops_daily_forecast",
    description="Request tomorrow's 24 hourly load forecasts from the serving API and record them",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ampops", "serving", "forecast"],
)
def ampops_daily_forecast():
    @task
    def resolve_horizon() -> dict:
        """The 24 hourly timestamps of the next operating day, in replay time."""
        today = pendulum.parse(os.getenv("AMPOPS_SIMULATED_TODAY", DEFAULT_SIMULATED_TODAY))
        target_day = today.add(days=1).start_of("day")
        timestamps = [
            target_day.add(hours=hour).format("YYYY-MM-DDTHH:mm:ss")
            for hour in range(FORECAST_HOURS)
        ]
        return {
            "grid_id": _grid_id(),
            "date": target_day.format("YYYY-MM-DD"),
            "timestamps": timestamps,
        }

    @task
    def request_forecast(horizon: dict) -> dict:
        """POST the whole horizon to the API in one call — one window, one JVM trip."""
        url = f"{_api_url()}/predict/batch"
        payload = {"grid_id": horizon["grid_id"], "timestamps": horizon["timestamps"]}
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            raise AirflowFailException(
                f"Could not reach the serving API at {url}: {exc}. The api service is "
                "behind the 'serving' compose profile — start it with "
                "`docker compose --profile serving up -d api redis`."
            ) from exc

        if response.status_code >= 400:
            raise AirflowFailException(
                f"{url} returned {response.status_code}: {response.text[:500]}"
            )

        body = response.json()
        return {"date": horizon["date"], **body}

    @task
    def persist(forecast: dict, **context) -> dict:
        """Upsert the 24 rows into the `ampops` database — the system of record.

        Only scheduled batches land here. Ad-hoc `/predict` traffic stays
        ephemeral, so the accuracy metric the retrain webhook keys off is never
        polluted by exploratory calls.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        rows = [
            (
                forecast["grid_id"],
                prediction["timestamp"],
                float(prediction["predicted_mw"]),
                forecast["model_version"],
                context["run_id"],
            )
            for prediction in forecast["predictions"]
        ]

        try:
            with hook.get_conn() as conn, conn.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
                cursor.executemany(UPSERT_SQL, rows)
                conn.commit()
        except Exception as exc:
            # postgres:15 runs /docker-entrypoint-initdb.d/* only on the FIRST
            # initialization of its data volume, so anyone with a pre-existing
            # stack never got the `ampops` database created.
            if "does not exist" in str(exc).lower():
                raise AirflowFailException(
                    f"The `ampops` database is missing ({exc}). Run `make ampops-db-init` "
                    "— docker/postgres/init-ampops-db.sh only runs on a fresh "
                    "postgres-data volume."
                ) from exc
            raise

        return {
            "grid_id": forecast["grid_id"],
            "date": forecast["date"],
            "rows": len(rows),
        }

    @task
    def verify_cached(persisted: dict) -> dict:
        """Read the forecast back through the API and fail loudly if hours are missing.

        `/predict/batch` writes through to the forecast cache, so a short read
        here proves the operational artifact — not just the database row — is
        actually there for whoever queries it next.
        """
        url = f"{_api_url()}/forecast"
        params = {"grid_id": persisted["grid_id"], "date": persisted["date"]}
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            raise AirflowFailException(f"Could not reach {url}: {exc}") from exc

        if response.status_code >= 400:
            raise AirflowFailException(
                f"{url} returned {response.status_code}: {response.text[:500]}"
            )

        cached = response.json().get("predictions", [])
        if len(cached) != FORECAST_HOURS:
            raise AirflowFailException(
                f"expected {FORECAST_HOURS} cached hours for {persisted['date']}, "
                f"found {len(cached)}"
            )

        return {**persisted, "cached": len(cached)}

    verify_cached(persist(request_forecast(resolve_horizon())))


ampops_daily_forecast()
