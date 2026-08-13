"""Structural tests for the daily forecast DAG (`dags/ampops_daily_forecast.py`).

Mirrors tests/test_dag.py: these only import the DAG module and inspect the
resulting `DagBag`/`DAG` objects, without a scheduler or metadata database, so
a broken import or a dropped task dependency is caught in a second instead of
in the Airflow UI.

One extra check beyond test_dag.py's pattern: docs/serving_contract.md §6
makes it a hard architectural rule that this DAG never imports `h2o`,
`mlflow`, `ampops.features`, or `ampops.training` — its whole job is to be a
thin HTTP client of the serving API's one inference code path
(`POST /predict/batch`), not a second, parallel model-loading path that could
silently diverge from what the API actually serves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="Airflow only installed in the pipeline image")

from airflow.models import DagBag  # noqa: E402

DAG_ID = "ampops_daily_forecast"
DAG_MODULE_PATH = Path("dags") / "ampops_daily_forecast.py"

# serving_contract.md §6: the forecast DAG must stay a thin HTTP client of the
# serving API — these are the only inference-adjacent modules it is forbidden
# from importing, because importing any of them would mean a second,
# divergence-prone inference code path.
FORBIDDEN_IMPORT_PREFIXES = ("h2o", "mlflow", "ampops.features", "ampops.training")


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder="dags", include_examples=False)


def _imported_module_names(source: str) -> set[str]:
    """All module names named in `import`/`from ... import` statements."""
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_dag_imports_without_errors(dagbag):
    assert not dagbag.import_errors, f"DAG import failed: {dagbag.import_errors}"


def test_dag_is_present(dagbag):
    assert DAG_ID in dagbag.dags


def test_task_ids_match_the_documented_pipeline(dagbag):
    dag = dagbag.dags[DAG_ID]
    expected = {"resolve_horizon", "request_forecast", "persist", "verify_cached"}

    assert expected <= set(dag.task_ids), f"missing tasks: {expected - set(dag.task_ids)}"


def test_task_dependencies_form_the_documented_chain(dagbag):
    """resolve_horizon -> request_forecast -> persist -> verify_cached.

    Each stage's output is what the next stage needs (timestamps, then the
    HTTP response, then the DB write) — an out-of-order edge here would mean
    e.g. verifying a Postgres write that hasn't happened yet.
    """
    dag = dagbag.dags[DAG_ID]

    assert "resolve_horizon" in {
        t.task_id for t in dag.get_task("request_forecast").upstream_list
    }
    assert "request_forecast" in {t.task_id for t in dag.get_task("persist").upstream_list}
    assert "persist" in {t.task_id for t in dag.get_task("verify_cached").upstream_list}


def test_catchup_is_disabled(dagbag):
    """Backfilling a daily forecast DAG would queue a run per missed day."""
    assert dagbag.dags[DAG_ID].catchup is False


def test_max_active_runs_is_one(dagbag):
    """Overlapping runs could double-write or race on the same target hour."""
    assert dagbag.dags[DAG_ID].max_active_runs == 1


def test_dag_module_does_not_import_h2o_mlflow_or_the_training_stack():
    source = DAG_MODULE_PATH.read_text(encoding="utf-8")
    modules = _imported_module_names(source)

    offending = {
        module
        for module in modules
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    }

    assert not offending, (
        f"forecast DAG imports {offending}, which serving_contract.md §6 forbids — "
        "inference must go through POST /predict/batch, not a second in-DAG model load"
    )
