"""Champion model loading and the H2O cluster lifecycle for a long-lived server.

`mlflow.h2o.load_model`, never `mlflow.pyfunc.load_model`. The pyfunc wrapper
calls `h2o.init()` itself using whatever port the artifact's `h2o.yaml`
recorded — usually the default 54321, with no way to override it. That is
exactly the stale-cluster hazard `ampops.training.automl`'s module docstring
describes, and it bites on container restart. Loading the native flavour keeps
port selection here, where a fresh free port is picked once per process.

Lifecycle, per serving_contract.md §3:

* `h2o.init()` runs **once**, in the FastAPI lifespan startup handler.
* The cluster is shut down in lifespan shutdown — *never* in a request handler.
  `automl.py` shuts down in a `finally` block, which is right for a batch
  Airflow task and fatal for a server that has to answer the next request.
* uvicorn must run a **single worker**. Multiple workers would each start a JVM
  and race on the port; this is a hard constraint, not a tuning preference.

`_free_port` and `_ensure_java_home` are imported from `ampops.training.automl`
rather than copied, for the same reason `app.features` delegates to
`ampops.features.build`: two implementations of the same rule eventually stop
agreeing, and this one governs whether the JVM starts at all.

A failed load is survivable and deliberately so. `register_champion()` soft-
fails on workspaces without a usable Unity Catalog, so `@champion` legitimately
may not resolve. Crashing at startup would make the container unrecoverable in
precisely the configuration the training pipeline already tolerates — instead
the process starts, `/health` passes, and `/ready` returns 503 naming the
failure.
"""

from __future__ import annotations

from typing import Any

import h2o
import mlflow
import mlflow.h2o
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from ampops.training.automl import _ensure_java_home, _free_port
from ampops.utils.io import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before a champion is available."""


def _resolve_version(model_uri: str) -> str:
    """The registry version string when it exists, else the run id.

    Never invented: this value is echoed in every prediction response and
    stored beside every cached forecast, so a made-up version would make the
    monitoring layer attribute errors to a model that never served them.

    The client is constructed with no explicit registry URI so it resolves the
    alias through exactly the same chain `mlflow.h2o.load_model` just used —
    anything else could report the version of a *different* registry's model.
    """
    if model_uri.startswith("runs:/"):
        return model_uri.split("/")[1]

    if model_uri.startswith("models:/"):
        spec = model_uri[len("models:/") :]
        if "@" in spec:
            name, alias = spec.split("@", 1)
            return str(MlflowClient().get_model_version_by_alias(name, alias).version)
        if "/" in spec:
            return spec.rsplit("/", 1)[1]

    return model_uri


class ChampionModel:
    """Holds the loaded H2O model plus the JVM it needs, for the app's lifetime."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.version: str | None = None
        self.model_uri: str | None = None
        self.load_error: str | None = None
        self._server: Any | None = None
        self._connection: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def start_cluster(self) -> None:
        """Start this process's one H2O cluster, on a fresh port."""
        _ensure_java_home()
        port = _free_port()
        h2o.init(port=port, start_h2o=True)
        # Keep a handle on *our* JVM. `h2o.cluster().shutdown()` acts on the
        # module-global connection, which a second app in the same process
        # (the test suite builds two) may have replaced by teardown time —
        # shutting down through this handle can only ever stop our own server.
        self._connection = h2o.connection()
        self._server = getattr(self._connection, "local_server", None)
        logger.info("H2O cluster started on port %s", port)

    def load(self, model_uri: str, tracking_uri: str) -> None:
        """Load the champion. Records the failure instead of raising.

        Note what is *not* here: `mlflow.set_registry_uri()`. That setter is
        process-global **and** writes `MLFLOW_REGISTRY_URI` into `os.environ`,
        so an app pinning its own registry silently reconfigures every other
        MLflow user in the process — including a training run started later in
        the same interpreter. MLflow already resolves the registry from
        `MLFLOW_REGISTRY_URI` (which compose sets from `.env`) and otherwise
        falls back to the tracking URI, which is exactly right for a local file
        store. Leaving that chain alone is both correct and side-effect free.
        """
        self.model_uri = model_uri
        mlflow.set_tracking_uri(tracking_uri)

        try:
            self.model = mlflow.h2o.load_model(model_uri)
            self.version = _resolve_version(model_uri)
            self.load_error = None
            logger.info("Loaded champion %s (version %s)", model_uri, self.version)
        except Exception as exc:  # noqa: BLE001 — an unresolved @champion is survivable
            self.model = None
            self.version = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Champion %s did not load (%s). /ready will report 503 until a model "
                "is available.",
                model_uri,
                self.load_error,
            )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """One `H2OFrame` round trip for the whole batch.

        Training always passed the target column into `predict` and serving
        never does. H2O accepts both; `tests/test_serving_features.py` is what
        proves the columns it *does* get are byte-identical to training's.
        """
        if self.model is None:
            raise ModelNotLoadedError(self.load_error or "no champion loaded")

        frame = h2o.H2OFrame(features)
        return self.model.predict(frame).as_data_frame()["predict"].to_numpy()

    def cluster_is_running(self) -> bool:
        try:
            cluster = h2o.cluster()
            return bool(cluster is not None and cluster.is_running())
        except Exception as exc:  # noqa: BLE001 — any probe failure means "down"
            logger.warning("H2O cluster probe failed: %s", exc)
            return False

    def shutdown_cluster(self) -> None:
        """Close our session and stop the JVM this app started, and only that one."""
        # Closing the session first is what keeps h2o's atexit hook quiet
        # ("session was not closed properly") — and, like the server handle,
        # it acts on our own connection object rather than the module global.
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort
                logger.debug("H2O session close failed: %s", exc)
            finally:
                self._connection = None

        if self._server is None:
            return
        try:
            self._server.shutdown()
            logger.info("H2O cluster stopped")
        except Exception as exc:  # noqa: BLE001 — shutdown is best-effort
            logger.warning("H2O cluster shutdown failed: %s", exc)
        finally:
            self._server = None
