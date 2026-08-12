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
# AMPOPS_UC_MODEL_PREFIX=catalog.schema   # if name is not already 3-level
```

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
