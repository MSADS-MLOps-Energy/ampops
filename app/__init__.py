"""AmpOps FastAPI serving layer.

The API is the *only* place inference happens. Everything else that needs a
prediction — the daily forecast DAG, a dashboard, a notebook — goes through
`POST /predict/batch` rather than loading the champion itself, so there is
exactly one code path that can drift from training.

`docs/serving_contract.md` is the binding spec for this package.
"""

from __future__ import annotations
