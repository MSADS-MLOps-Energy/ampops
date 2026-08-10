# AutoML implementation: H2O AutoML (replacing the hardcoded bake-off)

**Status: implemented and validated, including the register → evaluate-on-test
→ tag flow, with a real registered model version live in MLflow.**
`ampops.training.automl.run_h2o_automl()` / `evaluate_on_test()` and
`ampops.training.registry.register_champion()` / `tag_test_metrics()` are
real, tested (`tests/test_automl.py`, real H2O runs, not mocked), wired into
`dags/ampops_training_pipeline.py`, and validated end-to-end against both
synthetic data and the real `data/processed/train.parquet`/`test.parquet` —
including an actual model registered in the running MLflow instance
(`ampops-demand-forecaster` v1, `@champion`). The full test suite (72 tests)
has been re-run inside the actual `docker compose` stack (Java 17 + H2O +
Airflow), not just a bare `.venv` — see "Validation performed" below. This
document describes what was actually built, not a plan.

## Summary

`docs/AmpOps_Project_Context.md` §2.3 originally satisfied the course's
AutoML mention with a hardcoded three-model bake-off
(Linear → RandomForest → XGBoost, fixed hyperparameters) instead of a real
AutoML framework, reasoning that the course lists AutoML as an example, not a
requirement. §9 of the same doc separately assigns a distinct Week 2
deliverable — "Set up AutoML for model building (Databricks) → Model Build" —
that the bake-off was never meant to satisfy. This document covers how that
deliverable was fulfilled for real: the `train`/`choose_champion` steps in
`dags/ampops_training_pipeline.py` were replaced with a single `run_automl`
task backed by genuine automated model search (H2O AutoML), and it also
records the investigation that determined which AutoML tool to use.

## Why not Databricks

The team stood up a real Databricks Free Edition workspace and ran a live
spike against it — not a simulation. OAuth and PAT authentication were both
tested and confirmed working, the Jobs API was reachable, and Unity Catalog
was enabled with 3 catalogs. Two separate, hard blockers were found:

1. **No classic compute.** The workspace only supports serverless compute.
   Submitting a job with a classic `new_cluster` spec was rejected outright
   by the platform with the error *"Only serverless compute is supported in
   the workspace."*
2. **`databricks.automl` is not importable in serverless.** A live test
   submitted a notebook job that attempted `from databricks import automl`
   and failed with `cannot import name 'automl' from 'databricks'`. The real
   AutoML Python API (`automl.regress(...)` etc.) has always required a
   classic Databricks Runtime ML cluster — something Free Edition's
   serverless-only tier cannot provision. There is also no REST-level escape
   hatch: the `databricks-sdk`'s `WorkspaceClient` was inspected directly and
   has no `automl` client namespace at all.

**Conclusion:** this is a product-tier limitation, not a configuration
mistake. Real Databricks AutoML cannot run on Databricks Free Edition as it
exists today, regardless of how the job or auth is configured. Databricks
credentials have since been removed from the local `.env`.

## The H2O approach (as built)

The team pivoted to the open-source `h2o` Python package. It runs in-process
inside the existing Airflow worker container, needs no external account, no
new credentials, and — unlike the Databricks design, which would have
required copying model artifacts from a Databricks-hosted MLflow tracking
server back to the project's local self-hosted one — no cross-service
artifact bridging.

### Architecture

`src/ampops/training/automl.py` (~150 lines) exposes:

```python
run_h2o_automl(train_path, tracking_uri=None, experiment=None) -> dict
```

It is called from a new `run_automl` Airflow task in
`dags/ampops_training_pipeline.py`, which replaces the old dynamically-mapped
`train` (per-algorithm bake-off) and `choose_champion` tasks. The DAG's final
wiring is now:

```
evaluate_test(register(run_automl(splits["train"])), splits["test"])
```

`register()` (in `src/ampops/training/registry.py`) was **not** modified —
the whole point of the AutoML swap was that `run_h2o_automl()`'s return shape
needed to be a drop-in replacement for what `select_champion()` used to
produce.

Step by step:

1. **Cluster startup with retry-safe port allocation.** Before every
   `h2o.init()`, the function grabs an OS-assigned free TCP port via
   `socket.bind(("127.0.0.1", 0))` and passes it explicitly to `h2o.init()`.
   This guarantees a fresh H2O cluster is started rather than potentially
   reconnecting to a stale JVM left behind by a killed prior Airflow task
   attempt — a real risk given the DAG's `retries: 1` combined with
   Airflow's LocalExecutor subprocess model, not a hypothetical one. Cluster
   shutdown happens in a `finally` block.

   **Residual risk (accepted, documented in the module's own docstring):** a
   `SIGKILL`'d attempt skips the `finally`, so a stray JVM could still linger
   in the long-lived Airflow scheduler container until it's restarted. The
   port-picking design ensures that orphan can never block or get adopted by
   a subsequent retry — but nothing in-process can clean it up after a hard
   kill.

2. **Split.** Loads `train_path` and reuses the existing
   `features.split.time_split(df, test_months=VALIDATION_MONTHS)` — the same
   3-month chronological carve-out `src/ampops/training/bakeoff.py` already
   used — to get a fit/validation split. Both are converted to `H2OFrame`.
   The sealed `test.parquet` is never touched by this step.

3. **Train.**
   ```python
   H2OAutoML(
       max_runtime_secs=config.AUTOML_MAX_RUNTIME_SECS,
       max_models=config.AUTOML_MAX_MODELS,
       seed=config.RANDOM_SEED,
       nfolds=0,
       sort_metric="RMSE",
   )
   aml.train(x=feature_columns(df), y=config.TARGET,
             training_frame=fit_h2o, validation_frame=valid_h2o)
   ```

   **`nfolds=0` with an explicit `validation_frame` is correctness-critical,
   not a style choice.** This is time-ordered data — lag/rolling features
   reference the past — so H2O's default k-fold cross-validation would
   shuffle rows across folds and leak future information into training. This
   is the same discipline the rest of the codebase already applies elsewhere
   (chronological splits, a sealed test set never touched during model
   selection).

   **Side effect (intentional, not a bug):** with `nfolds=0`, H2O skips
   building Stacked Ensemble meta-models, since those require
   cross-validated base-model predictions. Losing access to Stacked
   Ensembles is an accepted trade-off for correctness on time-ordered data.

4. **Evaluate.** Takes `aml.leader`, predicts on the validation frame, and
   computes metrics via the existing `bakeoff.evaluate()` helper — reused,
   not reimplemented. H2O AutoML's own leaderboard metric is RMSE (MAPE
   isn't a native H2O AutoML sort metric), so RMSE drives model selection,
   but MAPE — the project's headline metric — is still computed and reported
   the same way it always was.

5. **Log.** The winning model is logged via `mlflow.h2o.log_model()`
   directly to the project's existing local, self-hosted MLflow tracking
   server (`http://mlflow:5000` in Docker, experiment
   `ampops-demand-forecasting`) — no cross-service artifact bridging needed,
   unlike what the earlier, abandoned Databricks design would have required.

6. **Return.** Yields:
   ```python
   {
       "model_name": <algorithm string from aml.leader.algo>,
       "run_id": <the new local MLflow run's ID>,
       "mape": ...,
       "rmse": ...,
       "mae": ...,
       "n_train": ...,
       "n_valid": ...,
   }
   ```
   This is a valid input to the untouched `register_champion()`, verified by
   a real (not mocked) integration test, which sets the `@champion` alias on
   the `ampops-demand-forecaster` registered model exactly as it did before
   this change.

### Test-set evaluation: register → evaluate on test → tag

`run_h2o_automl()` only ever sees train/validate data — the sealed
`data/processed/test.parquet` holdout stays untouched through the whole
search and through registration. Closing the "train/validate/**test**" loop
was deliberately built as a separate, later step rather than folded into
`run_h2o_automl()` itself, so that model selection can never be influenced by
test-set performance and so the same evaluation logic can be pointed at *any*
registered model version later, not just one freshly produced by this run:

1. **`register_champion()`** (unchanged) promotes the AutoML leader to
   `@champion` exactly as before — this step doesn't know or care that a test
   evaluation is coming next.
2. **`automl.evaluate_on_test(model_uri, test_path, tracking_uri=None)`**
   (new) reloads the *exact* registered version via its
   `models:/<name>/<version>` URI — not the `@champion` alias, so a
   concurrent run moving the alias can't change what gets scored — using
   `mlflow.h2o.load_model()`, scores it against `test.parquet` with the same
   `bakeoff.evaluate()` helper used everywhere else, and returns
   `{"test_mape", "test_rmse", "test_mae", "n_test"}`. It manages its own
   fresh H2O cluster (same `_free_port()` + `finally`-shutdown discipline as
   `run_h2o_automl`), since it runs as an independent Airflow task/process
   with nothing to inherit from the training task.
3. **`registry.tag_test_metrics(registration, metrics, tracking_uri=None)`**
   (new) writes `test_mape` / `test_rmse` / `test_mae` onto the already-
   registered model version as MLflow model-version tags, and also logs them
   onto the original training run (`MlflowClient.log_metric`, since there's
   no active run context at this point) so validation and test metrics live
   side by side. It does not re-decide or re-promote anything — no
   pass/fail gate was added; this is reporting/tagging only, by design.

The DAG wires this as a new `evaluate_test` task immediately after
`register`, taking both the registration dict and `splits["test"]`:

```python
@task
def evaluate_test(registration: dict, test_path: str) -> dict:
    model_uri = f"models:/{registration['registered_model']}/{registration['version']}"
    metrics = automl.evaluate_on_test(model_uri, test_path)
    return registry.tag_test_metrics(registration, metrics)
```

`tests/test_automl.py` covers this live: after registering a synthetic
champion, it calls `evaluate_on_test()` against a distinct synthetic
test-holdout fixture, then `tag_test_metrics()`, then independently confirms
via `MlflowClient.get_model_version(...).tags` that the tags actually landed
on the registry — not just that the function returned without error.

### Config knobs

`src/ampops/config.py`:

- `MODEL_CONFIGS` (the old hardcoded bake-off hyperparameter list) was
  **removed**.
- `AUTOML_MAX_RUNTIME_SECS` — default 300 seconds, env-overridable via
  `AMPOPS_AUTOML_MAX_RUNTIME_SECS`.
- `AUTOML_MAX_MODELS` — default 10, env-overridable via
  `AMPOPS_AUTOML_MAX_MODELS`.

`src/ampops/training/bakeoff.py`: `build_estimator()`, `log_model()`, and
`train_candidate()` were removed (dead now that the manual bake-off is
gone). `evaluate()` was kept — it's reused by `automl.py` — along with the
`VALIDATION_MONTHS = 3` constant.

`src/ampops/training/registry.py`: `select_champion()` was removed (no
longer needed — AutoML already produces a single winner, there's nothing to
pick between). `register_champion()` is completely unchanged.

### New dependency: Java

H2O requires a JVM.

- `docker/airflow/Dockerfile` now installs `openjdk-17-jre-headless` (H2O
  3.46.x supports Java 8–17; 17 is the newest supported LTS available on the
  image's Debian bookworm base).
- `requirements-airflow.txt` now pins `h2o==3.46.0.11` — confirmed
  compatible with Airflow 2.9.3's official constraints file, which resolves
  `numpy==1.26.4` / `pandas==2.1.4` (H2O-3 requires `numpy<2`).

### Running it

**In Docker** (the primary path): no extra setup. `openjdk-17-jre-headless`
and `h2o==3.46.0.11` are baked into the Airflow worker image via
`docker/airflow/Dockerfile` / `requirements-airflow.txt`. Trigger the DAG
normally (`make dag-trigger` or via the Airflow UI) and `run_automl` runs
like any other task.

**Locally, outside Docker** (for running `run_h2o_automl()` directly, e.g.
via `make dag-test-local` or a script/notebook): H2O needs a local JVM plus
the `h2o` package, neither of which are part of the base `make setup` /
`requirements.txt` install.

```bash
brew install openjdk@17
# openjdk@17 installs keg-only on macOS — it will not be on PATH by default
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"

pip install h2o==3.46.0.11   # into the same .venv as the rest of the project
```

Without the `PATH` export, `h2o.init()` fails to find `java` even though
Homebrew installed it — this is the most common local setup snag.

### Validation performed

- A genuine (not mocked) H2O AutoML run against synthetic data, and
  separately a real smoke test against the actual
  `data/processed/train.parquet`: DRF (H2O's Distributed Random Forest) came
  out as leader, MAPE 0.0435 / RMSE 762.05 on real data. Both confirmed the
  full path works, including logging to MLflow and successful hand-off to
  `register_champion()`.
- New test file `tests/test_automl.py` runs a real H2O AutoML search (not
  mocked) against a small synthetic dataset, using an isolated
  `tmp_path`-based local MLflow tracking URI so it never depends on the
  `mlflow` Docker service being up.
- Local (non-Docker) `.venv` run: 68 passed, 1 skipped. The skip is
  `tests/test_dag.py`, expected locally since Airflow isn't installed
  outside Docker (guarded by `pytest.importorskip`).
- **Re-validated inside the real `docker compose` stack** (postgres +
  mlflow + airflow-scheduler/webserver, the actual Java 17 + `h2o==3.46.0.11`
  image), by copying `tests/` into the running `airflow-scheduler`
  container and running pytest there: **72 passed**, 0 skipped — this run
  additionally exercises `test_dag.py` for real (Airflow is installed) and
  confirms `test_automl.py`'s live H2O search + `register_champion()`
  hand-off succeed in the actual production-shaped environment, not just a
  bare `.venv`.
- Attempted a live end-to-end trigger of `ampops_training_pipeline` via
  `airflow dags trigger` against the running stack, twice (including once
  against a completely fresh Docker Desktop VM, restarted specifically to
  rule out session-accumulated state). Both times it failed at `clean_and_join`
  writing `joined_hourly.parquet` with `OSError: [Errno 35] Resource deadlock
  avoided` — the same error class as an earlier session's failure reading the
  weather CSV at `ingest_raw`. **This is an open, unresolved Docker
  Desktop/VirtioFS bug, not an AutoML or pipeline-code defect** — full
  root-cause investigation, evidence, and recommended fixes are in
  **`docs/virtiofs_errno35_deadlock.md`**; that document also corrects an
  earlier (incorrect) "Spotlight indexer" theory from a prior session. In
  short: a per-inode VirtioFS lock-state bug in Docker Desktop's own VM
  process, worked around (not fixed) this session by copying the affected
  files to fresh inodes from the host side.
- **Live validation, real data, registered for real**: with that workaround,
  ran `automl.run_h2o_automl()` → `registry.register_champion()` →
  `automl.evaluate_on_test()` → `registry.tag_test_metrics()` directly
  against the actual `data/processed/train.parquet` (54,321 rows) /
  `test.parquet` (2,208 train-internal-validation + 8,591 sealed-test rows)
  inside the real `airflow-scheduler` container (Java 17, `h2o==3.46.0.11`).
  Result: DRF leader, **validation MAPE 0.0425 / RMSE 745.2 MW**, **test MAPE
  0.0302 / RMSE 519.3 MW**. Registered as `ampops-demand-forecaster` v1,
  `@champion` alias confirmed resolving to v1, tags confirmed present via
  `MlflowClient`: `semantic_version=1.1.0`, `algorithm=drf`,
  `mape=0.042476`, `rmse=745.199651`, `test_mape=0.030194`,
  `test_rmse=519.344517`, `test_mae=351.253249`. Test MAPE beating
  validation MAPE here is incidental to this particular random split, not a
  general guarantee.

## Known limitations / forward-looking notes

- **Serving-side Java dependency.** The not-yet-built FastAPI serving layer
  will eventually need to load the registered champion via
  `mlflow.pyfunc.load_model(...)`. An H2O-logged model's pyfunc wrapper
  needs Java plus the `h2o` package installed wherever it's loaded — unlike
  the sklearn/xgboost models it replaces, which were lightweight to load.
  The root `Dockerfile` (for the future `api` service, currently
  `python:3.11-slim` with no Java) doesn't have this yet. This is not being
  fixed as part of this work — just documented so the future serving
  workstream isn't surprised by this new dependency when it lands.
