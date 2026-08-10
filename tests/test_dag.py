"""DAG structural tests.

These run without a scheduler or a metadata database — they only import the DAG
module and inspect the resulting object. Catching an import error or a broken
dependency here takes a second; catching it in the Airflow UI takes a rebuild.

Skipped automatically when Airflow is not installed, so the suite still passes
in a plain venv. Inside the container (or after `pip install apache-airflow`)
they run for real.
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow only installed in the pipeline image")

from airflow.models import DagBag  # noqa: E402

DAG_ID = "ampops_training_pipeline"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder="dags", include_examples=False)


def test_dag_imports_without_errors(dagbag):
    assert not dagbag.import_errors, f"DAG import failed: {dagbag.import_errors}"


def test_dag_is_present(dagbag):
    assert DAG_ID in dagbag.dags


def test_task_dependencies_form_the_expected_chain(dagbag):
    dag = dagbag.dags[DAG_ID]
    ids = set(dag.task_ids)

    expected = {
        "ingest_raw",
        "validate_raw",
        "clean_and_join",
        "validate_joined",
        "build_features",
        "split_train_test",
        "run_automl",
        "register",
        "evaluate_test",
    }
    assert expected <= ids, f"missing tasks: {expected - ids}"

    # Validation must sit between ingestion and cleaning — if this edge is ever
    # dropped, bad data reaches training silently.
    assert "validate_raw" in {t.task_id for t in dag.get_task("clean_and_join").upstream_list}
    assert "validate_joined" in {t.task_id for t in dag.get_task("build_features").upstream_list}
    assert "run_automl" in {t.task_id for t in dag.get_task("register").upstream_list}
    # Test evaluation must happen after registration, never before — model
    # selection is decided on train/validate alone.
    assert "register" in {t.task_id for t in dag.get_task("evaluate_test").upstream_list}
    assert "split_train_test" in {t.task_id for t in dag.get_task("evaluate_test").upstream_list}


def test_catchup_is_disabled(dagbag):
    """Backfilling a retraining pipeline would queue a run per missed interval."""
    assert dagbag.dags[DAG_ID].catchup is False
