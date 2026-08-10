# Airflow image for the AmpOps training pipeline.
#
# Pinning Python here is the reason this project runs Airflow in Docker rather
# than a local venv: Airflow 2.9.3 does not support Python 3.13, and macOS
# system Python is 3.9. The image settles that argument.
FROM apache/airflow:2.9.3-python3.11

USER root
# libgomp is XGBoost's OpenMP runtime — the slim base image omits it and
# `import xgboost` fails without it.
# openjdk-17-jre-headless is for H2O (src/ampops/training/automl.py):
# h2o.init() launches a local Java process, and the base image ships no JRE.
# H2O 3.46.x supports Java 8-17 (21 is not yet supported); 17 is the newest
# supported LTS and matches the h2o==<pin> version below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# data/interim and data/processed are named volumes (docker-compose.yml), and
# Docker initializes an empty named volume from whatever exists at its mount
# path in the image — including ownership. The base image has no
# /opt/airflow/data at all, so without this the volumes would come up root-owned
# and every write_parquet() would fail with EACCES for the airflow user.
# (/opt/airflow/logs already ships as 50000:0 775, so it needs nothing here.)
RUN mkdir -p /opt/airflow/data/raw /opt/airflow/data/interim /opt/airflow/data/processed \
    && chown -R 50000:0 /opt/airflow/data \
    && chmod -R 775 /opt/airflow/data

USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt

# --constraint keeps pip from upgrading Airflow's own pinned dependencies out
# from under it, which is the usual cause of a broken Airflow image.
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt" \
    -r /tmp/requirements-airflow.txt
