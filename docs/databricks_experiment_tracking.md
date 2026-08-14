# Databricks Experiment Tracking (with H2O AutoML)

**Date:** 2026-08-11 (updated after AutoML merge)  
**Owner:** Collin (experiment-tracking workstream)

---

## Design

| Piece | Choice |
|---|---|
| Model search | **H2O AutoML** in-process (`ampops.training.automl`) |
| Tracking / registry | **Databricks MLflow** (`MLFLOW_TRACKING_URI=databricks`) |
| Orchestration | Airflow in Docker (Java 17 in image) or host `make train` |

Databricks **AutoML** is not used (Free Edition has no classic compute /
`databricks.automl`). VirtioFS named volumes for interim/processed/logs stay
as in `docs/virtiofs_errno35_deadlock.md` — do not re-bind-mount those paths.

---

## `.env`

```env
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT=/Users/you@uchicago.edu/Ampops
MLFLOW_REGISTRY_URI=databricks-uc
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<pat>
AMPOPS_MODEL_NAME=ampops-demand-forecaster
AMPOPS_UC_MODEL_PREFIX=workspace.default   # or your own catalog.schema
```

`workspace.default` is the schema every Databricks workspace auto-creates, so
it's a safe default with no provisioning step. `.env`/`.env.example` structure
this as two mutually-exclusive blocks (Databricks active / local commented
out, or vice versa) — see the comment header in either file before editing.

---

## Status: confirmed working end-to-end (2026-08-14)

`workspace.default.ampops-demand-forecaster` is registered in this
workspace's Unity Catalog with the `@champion` alias resolving to v1 — not a
soft-fail fallback. It was migrated in from the local-compose champion
(v5, `drf`) via `scripts/migrate_champion_to_databricks.py` rather than
retrained, so its metrics match the local run exactly (`mape=0.042276`,
`test_mape=0.030125`); the version tags include `source_run_id` and
`migrated_from=local-compose` for provenance. The FastAPI serving layer
(`docs/fastapi_serving_layer.md`) loads it directly from this workspace.

---

## Host run (needs Java)

Homebrew OpenJDK is keg-only. `make train` and `automl._ensure_java_home()`
prefer `/opt/homebrew/opt/openjdk@17`.

```bash
conda activate ampops
brew install openjdk@17   # once
make pipeline-local       # if train/test parquets missing
make train
```

Pipeline: AutoML → register (soft-fail OK) → sealed test via `model_uri`
(`models:/…@champion` or `runs:/<run_id>/model`) → log `test_*` metrics on the run.

---

## Registry soft-fail

Many course workspaces disable the legacy Model Registry and lack a default UC
catalog. Registration then warns and continues; experiment runs + holdout
metrics still land in Databricks. To enable `@champion`, set a real
`catalog.schema.model` name (or `AMPOPS_UC_MODEL_PREFIX`).

---

## Airflow

Compose forwards `MLFLOW_*` and `DATABRICKS_*` into Airflow. Rebuild after
pulling `databricks-sdk` in `requirements-airflow.txt`:

```bash
make airflow-up
make dag-trigger
```
