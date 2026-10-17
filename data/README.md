# data/

Everything in this folder except this README and `download_data.sh` is
git-ignored and re-created by `make ingest` (or `sh data/download_data.sh`).

| Path | Produced by | Contents |
|------|-------------|----------|
| `raw/heart_disease_raw.csv` | `src/ingest.py` | Raw UCI rows from 4 hospitals + `source` column |
| `raw/manifest.json` | `src/ingest.py` | Source, SHA-256 of the raw file, row counts, timestamp |
| `processed/heart_disease_clean.parquet` | `src/ingest.py` | Cleaned dataset (validated by `CLEAN_SCHEMA`) |
| `processed/reference.parquet` | `src/ingest.py` | Reference / training set (60 %, stratified) |
| `production/batch_XX.parquet` | `src/ingest.py` | Simulated production batches (remaining 40 %) |
| `production/drifted_batch.parquet` | `scripts/simulate_drift.py` | Drifted batch for the retraining demo |
| `monitoring.db` | API (local runs only) | SQLite prediction log; docker-compose uses Postgres |

Source: UCI Machine Learning Repository, *Heart Disease* (Janosi, Steinbrunn,
Pfisterer, Detrano, 1989), CC BY 4.0. https://archive.ics.uci.edu/dataset/45/heart+disease
