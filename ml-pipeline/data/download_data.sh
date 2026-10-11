#!/usr/bin/env sh
# Download + validate + split the UCI Heart Disease dataset into data/.
# Equivalent to `make ingest`. Pass --force to re-download.
set -eu
cd "$(dirname "$0")/.."
exec python -m src.ingest "$@"
