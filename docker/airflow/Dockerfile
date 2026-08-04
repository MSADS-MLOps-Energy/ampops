# Airflow image for the AmpOps training pipeline.
#
# Pinning Python here is the reason this project runs Airflow in Docker rather
# than a local venv: Airflow 2.9.3 does not support Python 3.13, and macOS
# system Python is 3.9. The image settles that argument.
FROM apache/airflow:2.9.3-python3.11

USER root
# libgomp is XGBoost's OpenMP runtime — the slim base image omits it and
# `import xgboost` fails without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt

# --constraint keeps pip from upgrading Airflow's own pinned dependencies out
# from under it, which is the usual cause of a broken Airflow image.
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt" \
    -r /tmp/requirements-airflow.txt
