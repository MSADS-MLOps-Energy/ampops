# Pinned to bookworm (Debian 12) deliberately, NOT plain python:3.11-slim.
# That tag now resolves to Debian 13 (trixie), which dropped OpenJDK 17 and
# ships only openjdk-21/25 — and H2O 3.46.x does not support Java 21, so the
# build fails with apt exit 100 and the "fix" of taking 21 instead would break
# model loading at runtime. Bookworm still carries openjdk-17-jre-headless.
# apache/airflow:2.9.3-python3.11 is bookworm-based, which is why the training
# image was unaffected; keep the two in step.
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps (build tools needed for some ML libs).
#
# libgomp is XGBoost's OpenMP runtime — the slim base image omits it and
# `import xgboost` fails without it.
# openjdk-17-jre-headless is for H2O: the champion is logged with the native
# `mlflow.h2o` flavour, so loading it starts a local Java process and the base
# image ships no JRE. H2O 3.46.x supports Java 8-17 (21 is not yet supported);
# 17 is the newest supported LTS and matches the h2o pin in requirements.txt.
# This mirrors docker/airflow/Dockerfile — the training and serving images must
# agree about Java or a model that trains fine will not load.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# src layout: this is what makes `import ampops` resolve, and app/features.py
# delegating to ampops.features.build is the whole anti-skew design.
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Single worker, deliberately. Each worker would start its own H2O JVM and race
# on the port — see app/model.py. Scale with replicas, never with --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
