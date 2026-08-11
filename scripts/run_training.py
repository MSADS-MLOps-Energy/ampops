#!/usr/bin/env python
"""Run the AmpOps bake-off + holdout eval against the configured MLflow backend.

With `.env` pointing at Databricks (from the repo root):

    conda activate ampops
    set -a && source .env && set +a
    make pipeline-local
    make train

Or:

    python scripts/run_training.py
    python scripts/run_training.py --skip-register
    python scripts/run_training.py --models xgboost

Re-run exact hyperparameters from a prior MLflow / Databricks run:

    python scripts/run_training.py --from-run <run_id> --skip-register

Override individual knobs on top of config (or --from-run):

    python scripts/run_training.py --models xgboost --param n_estimators=300 --param max_depth=6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from ampops import config  # noqa: E402
from ampops.training.bakeoff import (  # noqa: E402
    load_config_from_run,
    parse_param_overrides,
)
from ampops.training.pipeline import run_experiment_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default=str(config.TRAIN_PARQUET),
        help="Path to train.parquet",
    )
    parser.add_argument(
        "--test-path",
        default=str(config.TEST_PARQUET),
        help="Path to sealed test.parquet",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[c["model_name"] for c in config.MODEL_CONFIGS],
        help="Subset of MODEL_CONFIGS to train (default: all)",
    )
    parser.add_argument(
        "--from-run",
        metavar="RUN_ID",
        help="Re-train using hyperparameters logged on an existing MLflow run",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a hyperparameter (repeatable), e.g. --param n_estimators=300",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip MLflow Model Registry promotion",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip sealed test evaluation",
    )
    args = parser.parse_args(argv)

    overrides = parse_param_overrides(args.param)
    parent_run_id = None

    if args.from_run:
        loaded = load_config_from_run(args.from_run)
        parent_run_id = loaded["source_run_id"]
        params = {**loaded["params"], **overrides}
        configs = [{"model_name": loaded["model_name"], "params": params}]
    else:
        configs = list(config.MODEL_CONFIGS)
        if args.models:
            wanted = set(args.models)
            configs = [c for c in config.MODEL_CONFIGS if c["model_name"] in wanted]
        if overrides:
            configs = [
                {
                    "model_name": c["model_name"],
                    "params": {**(c.get("params") or {}), **overrides},
                }
                for c in configs
            ]

    if parent_run_id:
        for cfg in configs:
            cfg["_parent_run_id"] = parent_run_id

    summary = run_experiment_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        model_configs=configs,
        register=not args.skip_register,
        run_holdout=not args.skip_holdout,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
