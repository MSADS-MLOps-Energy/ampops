#!/bin/bash
# Create the `ampops` database alongside Airflow's own.
#
# The forecast DAG records its output in a separate database on the same
# instance, so nothing the serving layer writes can collide with Airflow's
# schema and the two lifecycles stay distinguishable.
#
# IMPORTANT: postgres:15 runs everything in /docker-entrypoint-initdb.d/ only on
# the FIRST initialization of its data volume. An existing `postgres-data`
# volume will never see this script — use `make ampops-db-init` for that case.
# The \gexec form below keeps it idempotent either way.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'EOSQL'
SELECT 'CREATE DATABASE ampops'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ampops')\gexec
EOSQL

echo "init-ampops-db: ampops database ready"
