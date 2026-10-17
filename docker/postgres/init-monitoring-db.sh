#!/bin/sh
# Runs once, on first start of an empty Postgres volume.
# The default database ($POSTGRES_DB) is MLflow's backend store; this creates a
# second database for the API's prediction log so the two never mix.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
CREATE DATABASE "${MONITORING_DB:-monitoring}";
EOSQL
echo "created database ${MONITORING_DB:-monitoring}"
