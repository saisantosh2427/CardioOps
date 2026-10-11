# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

A production-style ML system built **entirely from free, open-source tools** that
runs on a laptop (CPU only) with `docker compose up`. It predicts the presence of
heart disease from 13 clinical measurements (UCI Heart Disease dataset) and covers
the full lifecycle: ingestion → validation → feature engineering → training with
hyper-parameter search → experiment tracking → model registry → serving →
monitoring → drift-triggered retraining with a champion/challenger promotion gate.

![Monitor dashboard detecting drift](docs/images/monitor-dashboard.png)

> **Not a medical device.** This is an MLOps reference project trained on a small,
> public 1988 dataset. Do not use it for clinical decisions.

**Stack:** Python 3.11 · scikit-learn · pandera · MLflow 3 (Postgres + MinIO) ·
FastAPI · Evidently · Prometheus client · SQLAlchemy · structlog · Docker Compose ·
GitHub Actions · (optional) Hugging Face Spaces.

---

## Contents

- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Demo: drift → retrain → promote](#demo-drift--retrain--promote)
- [Pipeline stages](#pipeline-stages)
- [How this maps to AWS SageMaker](#how-this-maps-to-aws-sagemaker)
- [Configuration](#configuration)
- [Development, tests and CI/CD](#development-tests-and-cicd)
- [Reproducibility and data versioning](#reproducibility-and-data-versioning)
- [Deploying the API to Hugging Face Spaces (free)](#deploying-the-api-to-hugging-face-spaces-free)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Repository layout](#repository-layout)

---

## Architecture

```mermaid
flowchart LR
    subgraph jobs["Pipeline jobs (make · trainer container · GitHub Actions)"]
        S1["1 · Ingest<br/>src/ingest.py<br/>UCI download, pandera, split"]
        S2["2 · Preprocess<br/>src/preprocess.py<br/>types, sentinels, target"]
        S3["3 · Features<br/>src/features.py<br/>engineered + ColumnTransformer"]
        S4["4 · Train<br/>src/train.py<br/>LR / RF / GB · GridSearchCV"]
        S9["9 · Retrain<br/>src/retrain.py<br/>drift + perf → challenger"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph compose["docker compose"]
        MLF["MLflow server<br/>5 · tracking · 6 · registry<br/>@champion / @challenger"]
        PG[("Postgres<br/>mlflow DB")]
        MINIO[("MinIO<br/>s3://mlflow")]
        INIT["minio-init<br/>creates bucket"]
        API["7 · FastAPI api :8000<br/>/predict /predict/batch<br/>/feedback /metrics"]
        LOG[("Postgres<br/>monitoring DB<br/>prediction log")]
        MON["8 · Monitor :8050<br/>Evidently drift +<br/>performance reports"]
        MLF --> PG
        MLF -->|proxied artifacts| MINIO
        INIT --> MINIO
    end

    S4 -->|params, metrics, plots,<br/>model, data snapshots| MLF
    Client(("clients")) -->|requests + ground truth| API
    API -->|"loads models:/…@champion<br/>(hot-swap on alias move)"| MLF
    API -->|every prediction + label| LOG
    MON -->|served window| LOG
    MON -->|champion reference data| MLF
    LOG -->|served window| S9
    S9 -->|"register challenger;<br/>promote only if better"| MLF
    PROM(("Prometheus<br/>(optional)")) -. scrape /metrics .-> API
    PROM -. scrape /metrics .-> MON
```

| Service | URL | What it is |
|---|---|---|
| API | http://localhost:8000/docs | FastAPI + Swagger UI |
| Monitor | http://localhost:8050 | Drift / performance dashboard (Evidently) |
| MLflow | http://localhost:5000 | Experiments, runs, Model Registry |
| MinIO console | http://localhost:9001 | Artifact bucket (`minioadmin` / `minioadmin`) |
| Postgres | localhost:5432 | `mlflow` DB (MLflow) + `monitoring` DB (prediction log) |

---

## Quickstart

Requirements: Docker (with Compose v2) and ~4 GB free RAM. Nothing else — no
cloud account, no API keys. Optional: Python 3.11 + `make` for running jobs on
the host (on Windows use Git Bash or WSL for `make`).

```bash
git clone https://github.com/<your-user>/ml-pipeline.git
cd ml-pipeline

# 1. Start the stack: postgres, minio (+ bucket init), mlflow, api, monitor
docker compose up -d --build

# 2. Ingest the data and train (runs in the `trainer` job container)
docker compose run --rm trainer python -m src.ingest
docker compose run --rm trainer python -m src.train
```

The API polls the registry every 15 s and loads the new `champion` automatically
(`GET /health` switches from `degraded` to `ok`). Then:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
       "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
       "ca": 0, "thal": 6}'
# {"request_id":"…","prediction":1,"probability":0.558026,"risk_label":"high",
#  "model_name":"heart-disease-classifier","model_version":"1"}

curl http://localhost:8000/model-info     # which registry version is served, how it scored
curl http://localhost:8000/metrics        # Prometheus metrics
```

Only `age`, `sex` and `cp` are required; any other field may be `null` and is
imputed exactly as in training. Invalid values (e.g. `"age": 7`, `"cp": 9`) get
a `422` with a precise message.

**Using local Python instead of the trainer container** (same commands, faster
iteration): `make setup` (creates `.venv` and `.env`), then `make train`,
`make drift-demo`, `make retrain`. Every job target also runs in the container
with `DOCKER=1`, e.g. `make train DOCKER=1`. `make help` lists all targets.

---

## Demo: drift → retrain → promote

The scenario: after deployment **a new hospital site** starts sending patients.
They are older (+7 y), with higher blood pressure (+12 mm Hg) and cholesterol
(+35 mg/dl), and the site's stress-test equipment reports max heart rate ~25 bpm
lower and ST depression on a different scale. The patients and their diagnoses
are real (held-back UCI rows) — only the recorded measurements shift, so the
relationship the champion learned no longer holds.

```bash
# 1. Normal traffic: replay a held-back batch through the API, then send the
#    ground-truth diagnoses to /feedback (delayed labels)
docker compose run --rm trainer python -m scripts.send_sample_requests --batch data/production/batch_01.parquet
docker compose run --rm trainer python -m src.drift
#  -> "retrain_recommended": false   (no drift)

# 2. The new site goes live
docker compose run --rm trainer python -m scripts.simulate_drift
docker compose run --rm trainer python -m scripts.send_sample_requests --batch data/production/drifted_batch.parquet
docker compose run --rm trainer python -m src.drift
#  -> "data drift: 5/13 features drifted (share 0.38 >= 0.3): age, chol, oldpeak, thalach, trestbps"

# 3. Look at it: http://localhost:8050 (dashboard + Evidently reports)

# 4. Retrain with the promotion gate
docker compose run --rm trainer python -m src.retrain
```

`make drift-demo DOCKER=1` runs steps 1–2 and `make retrain DOCKER=1` runs step 4.
What `src.retrain` does and what we measured on this data (seed 42):

| Step | Result |
|---|---|
| Drift (Evidently, KS / chi-square, p < 0.05) | 5 / 13 features drifted → share 0.38 ≥ 0.30 |
| Champion performance on labeled drifted traffic | ROC-AUC **0.952 → 0.874** (drop 0.078 > 0.05) |
| Challenger training data | champion's training history (550 rows) + 120 labeled production rows, production rows up-weighted (`balanced`) |
| Fair comparison on the *same* 80-row production hold-out (never trained on) | challenger ROC-AUC **0.898** vs champion **0.887** |
| Decision | **promoted**: `champion` → v2, v1 kept as `previous-champion` |
| API | picks up v2 within 15 s, no restart (`GET /model-info`) |
| Second cycle on the same traffic | drift share 0.0 → no action (the new champion's reference is the window it adapted to) |

Every cycle is logged to the `heart-disease-risk-monitoring` MLflow experiment
(drift share, decision, live metrics, the HTML report) and summarised in
`reports/retrain_summary.json`. Rollback is one command:
`MlflowClient().set_registered_model_alias("heart-disease-classifier", "champion", "<previous version>")`.

### Screenshots

All captured from the running stack during the demo above (regenerate with
`python -m scripts.capture_screenshots`, see [docs/images](docs/images/README.md)).

**Monitor dashboard** after the drifted batch: 5/13 features drifted, the
champion's live ROC-AUC fell from 0.952 to 0.874, so retraining is recommended
(http://localhost:8050).

![Monitor dashboard](docs/images/monitor-dashboard.png)

**Evidently data-drift report**: reference (champion training data) vs. the last
200 served predictions (http://localhost:8050/reports/drift).

![Evidently drift report](docs/images/drift-report.png)

**Evidently classification report** on the labeled production traffic
(http://localhost:8050/reports/performance).

![Evidently performance report](docs/images/performance-report.png)

**MLflow experiment**: the initial training run and the drift-triggered retrain,
each with nested Logistic Regression / Random Forest / Gradient Boosting runs and
their train/hold-out datasets; the winners are registry versions v1 and v2
(http://localhost:5000).

![MLflow training runs](docs/images/mlflow-runs.png)

**MLflow Model Registry** after promotion: v2 holds `@champion`, v1 is kept as
`@previous-champion` for rollback; tags record the trigger, hold-out metrics and
promotion reason.

![MLflow model registry](docs/images/mlflow-registry.png)

**API docs**: Swagger UI generated from the pydantic schemas
(http://localhost:8000/docs).

![API docs](docs/images/api-docs.png)

---

## Pipeline stages

| # | Stage | Where | Highlights |
|---|---|---|---|
| 1 | Data ingestion | `src/ingest.py`, `src/schemas.py` | Downloads the 4 UCI hospital files (920 rows), falls back to OpenML `heart-statlog`; pandera contracts for raw and clean data; SHA-256 manifest; stratified 60 % reference set + 4 simulated production batches |
| 2 | Preprocessing | `src/preprocess.py` | `?` → NaN, type coercion, physiologically impossible zeros (`chol`, `trestbps`) → NaN, invalid category codes → NaN, binary target (`num > 0`), dedup |
| 3 | Feature engineering | `src/features.py` | `HeartFeatureEngineer` (cleaning + `max_hr_pct`, hypertension, high cholesterol, severe ST depression) → `ColumnTransformer` (median impute + missing indicators + scaling; mode impute + one-hot). **Serialized inside the model**, so serving runs identical code |
| 4 | Training | `src/train.py` | Logistic Regression, Random Forest, Gradient Boosting; `GridSearchCV` with stratified 5-fold CV on ROC-AUC; best by CV score |
| 5 | Experiment tracking | MLflow | Nested run per candidate: params, CV + hold-out accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC curve, CV table, dataset lineage (`mlflow.log_input`), model with signature + input example |
| 6 | Model registry | `src/registry.py` | Best candidate registered; aliases `champion`, `challenger`, `previous-champion`; tags carry hold-out metrics and promotion/rejection reasons |
| 7 | Deployment | `api/` | FastAPI: `/health`, `/predict`, `/predict/batch`, `/model-info`, `/feedback`, `/model/reload`, `/metrics`; pydantic validation; multi-stage non-root image; hot-swaps on alias change |
| 8 | Monitoring | `src/drift.py`, `monitor/`, `src/store.py` | Every prediction logged to Postgres; Evidently data-drift + classification reports on served traffic; dashboard + Prometheus gauges |
| 9 | Automated retraining | `src/retrain.py`, `.github/workflows/retrain.yml` | Retrain on drift or performance drop (or `--force`); challenger promoted only if it beats the champion on the same production hold-out |

---

## How this maps to AWS SageMaker

| SageMaker capability | Free equivalent used here |
|---|---|
| Processing jobs | `src/ingest.py` + `src/preprocess.py` (pandera-validated), run via `make` / the `trainer` container |
| Training jobs | `src/train.py` on the laptop, in the `trainer` container, or on a GitHub Actions runner |
| Automatic Model Tuning (HPO) | `GridSearchCV` + `StratifiedKFold` (grid in `config.yaml`) |
| SageMaker Experiments | MLflow Tracking (Postgres backend, MinIO artifacts) |
| Model Registry (package groups, approval status) | MLflow Model Registry with aliases `champion` / `challenger` / `previous-champion` + reason tags |
| Real-time endpoints | FastAPI + uvicorn container (`api`), or a free Hugging Face Docker Space |
| Batch transform | `POST /predict/batch`, `scripts/send_sample_requests.py` |
| Model Monitor: data capture | Prediction log table in Postgres (`src/store.py`) |
| Model Monitor: baseline | Champion's reference snapshot stored as an MLflow run artifact |
| Model Monitor: data quality / drift | Evidently `DataDriftPreset` (KS / chi-square) — `src/drift.py`, monitor dashboard |
| Model Monitor: model quality (ground truth) | `POST /feedback` + Evidently `ClassificationPreset` + baseline comparison |
| CloudWatch metrics / alarms | Prometheus `/metrics` on API and monitor (`monitor_retrain_recommended`, latency, prediction counts …) |
| SageMaker Pipelines (+ condition step, RegisterModel) | `make` targets + `src/retrain.py` decision/promotion gate + scheduled `retrain.yml` workflow |
| Endpoint update / rollback | Alias move → API hot-swap; `previous-champion` alias for rollback |
| S3 | MinIO |
| ECR | GitHub Container Registry (pushed by `ci.yml`) |

---

## Configuration

Everything lives in **`config.yaml`** (paths, split sizes, model grids, MLflow
names, drift thresholds, promotion policy, API settings), loaded into typed
pydantic models by `src/config.py`. Any key can be overridden with an
environment variable `PIPELINE_<SECTION>__<KEY>`:

```bash
PIPELINE_MLFLOW__TRACKING_URI=http://mlflow:5000
PIPELINE_MONITORING__DATABASE_URL=postgresql+psycopg://mlflow:mlflow@postgres:5432/monitoring
PIPELINE_MONITORING__DRIFT_SHARE_THRESHOLD=0.2
PIPELINE_RETRAIN__MIN_IMPROVEMENT=0.01
```

Precedence: init kwargs > environment > `.env` > `config.yaml`. docker-compose
injects the in-network URLs; `.env.example` documents host-side settings and the
(local-only) infrastructure credentials. No secrets are committed — copy
`.env.example` to `.env` (`make setup` does it) and change the passwords before
exposing anything beyond localhost.

Key knobs:

| Key | Default | Meaning |
|---|---|---|
| `monitoring.window_size` | 200 | Most recent served predictions compared with the reference |
| `monitoring.drift_share_threshold` | 0.3 | Retrain when ≥ 30 % of features drift |
| `monitoring.max_metric_drop` | 0.05 | Retrain when live ROC-AUC falls this much below the champion's hold-out baseline |
| `retrain.promotion_metric` | `roc_auc` | Metric the challenger must beat (F1 breaks exact ties) |
| `retrain.min_improvement` | 0.0 | Required margin for promotion |
| `retrain.production_weight` | `balanced` | Up-weight new production rows to the same total weight as history |
| `api.refresh_interval_seconds` | 30 (15 in compose) | How often the API checks the `champion` alias |

---

## Development, tests and CI/CD

```bash
make setup        # .venv with pinned deps, .env, pre-commit hooks
make lint         # ruff check + ruff format --check
make test         # pytest with coverage (~2.5 min; `make test-fast` skips model training)
make serve        # API with auto-reload on :8001 (talks to the compose MLflow)
```

The test suite (88 tests, 86 % line coverage) needs no network or running
services — every test gets its own SQLite MLflow store and prediction log:

| File | Covers |
|---|---|
| `test_preprocess.py` | cleaning rules, target encoding, pandera contracts (valid + invalid data) |
| `test_features.py` | engineered features, pipeline fit/predict, **train/serve parity** (ints, nulls, missing keys), skops round-trip with the type allow-list |
| `test_ingest.py` | download → validate → split → manifest with mocked HTTP, caching, OpenML fallback |
| `test_train.py` | training smoke test (nested runs, artifacts, registry, aliases), "identical model is not promoted", comparison rules |
| `test_api.py` | every endpoint via `TestClient`, validation errors, batch limits, feedback, metrics, no-champion degradation, serving an exported model |
| `test_drift_retrain.py` | `decide_retrain` truth table, performance check, drift detection, promotion, rejection, insufficient labels, dry run, no endless retraining |
| `test_monitor.py`, `test_scripts.py`, `test_store.py`, `test_config.py` | dashboard over logged traffic, demo scripts, prediction log, config overrides, image requirement pins match the root pins |

**`ci.yml`** (push / PR): ruff → pytest + coverage → build the `api`, `monitor`
and `pipeline` images with a container smoke test each → on `main`, push to
`ghcr.io/<owner>/<repo>/<image>`.

**`retrain.yml`** (weekly cron + manual with `force` / `drift_intensity`
inputs): ingest → bootstrap a champion if none → simulate drifted traffic →
`src.retrain` → job summary + reports uploaded as artifacts. GitHub-hosted
runners cannot reach the MLflow server in your local compose stack, so the
workflow uses a **file-based MLflow store** (SQLite + local artifacts) inside the
runner, persisted between runs with `actions/cache`. Run it on a self-hosted
runner (and set `PIPELINE_MLFLOW__TRACKING_URI`) to retrain against a real server.

Structured logging (structlog) everywhere: human-readable locally, one JSON
object per line in containers (`PIPELINE_PROJECT__LOG_FORMAT=json`), including
uvicorn's own logs.

---

## Reproducibility and data versioning

- **Seeds:** `project.random_seed` (42) drives the reference/production split,
  train/test split, CV folds, every estimator and the drift simulation. The
  demo numbers above were identical on Windows and in Linux containers.
- **Pinned dependencies:** every package in `requirements.txt` is pinned; the
  slim image requirement files must match (enforced by a test).
- **Data lineage without extra tools:** `data/raw/manifest.json` records the
  source and SHA-256 of the raw download (OS-independent line endings). Every
  training run logs the hash as tags, the train/hold-out sets via
  `mlflow.log_input` (with content digests), and parquet snapshots of the
  training history and drift reference as run artifacts. Any registered model
  version can therefore be traced to — and retrained from — the exact data it saw.
- **DVC (optional):** to version `data/` outside MLflow, reuse MinIO as the remote:

  ```bash
  pip install "dvc[s3]"
  dvc init && dvc add data/raw/heart_disease_raw.csv
  dvc remote add -d minio s3://dvc && dvc remote modify minio endpointurl http://localhost:9000
  # create the "dvc" bucket in the MinIO console, export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  dvc push
  ```

  It is not wired in by default because the dataset is small, public and
  re-downloadable, and the manifest + MLflow snapshots already pin each model to its data.

---

## Deploying the API to Hugging Face Spaces (free)

A Space cannot reach your local MLflow server, so the champion is baked into the image:

```bash
make hf-space                       # or: python -m scripts.export_model --output hf_space/build
docker build -t heart-api-hf hf_space/build && docker run -p 7860:7860 heart-api-hf   # optional local check
```

Then create a Space (SDK: **Docker**, free CPU) and push the folder:

```bash
cd hf_space/build
git init && git remote add space https://huggingface.co/spaces/<your-user>/<space-name>
git add . && git commit -m "Deploy heart disease API" && git push space main
```

`hf_space/README.md` carries the Space metadata (`sdk: docker`, `app_port: 7860`).
The container runs as UID 1000, serves the model from `/app/model`
(`PIPELINE_API__MODEL_URI`) and logs predictions to an ephemeral SQLite file.
Re-run `make hf-space` and push again after each promotion.

---

## Design decisions

- **Aliases, not stages.** MLflow stages are deprecated; `champion` / `challenger`
  / `previous-champion` aliases make promotion and rollback a single atomic pointer move.
- **One pipeline object for train and serve.** Cleaning, feature engineering,
  imputation and encoding are steps of the serialized scikit-learn `Pipeline`;
  the API only builds a DataFrame from the validated request. A test asserts that
  JSON-style payloads (ints, nulls, missing keys) score identically to training data.
- **skops instead of pickle.** MLflow 3.17's default; models load only with an
  explicit allow-list of 3 types, and training fails if a model needs anything else.
- **Proxied artifacts.** The MLflow server is the only component with MinIO
  credentials (`--serve-artifacts`). Clients disable multipart transfer because
  presigned URLs would point at the internal `minio:9000` host.
- **Fair, leak-free promotion.** Champion and challenger are scored on the same
  slice of labeled *production* data, which the challenger never trained on.
  ROC-AUC is the gate (threshold-independent, steadier on small hold-outs); F1 breaks ties.
- **Reference follows the champion.** Drift is measured against the data the
  current champion was validated for. A retrained model's reference is the
  production window it adapted to, while its *training history* (stored
  separately) keeps all labeled data — this avoids endless "still drifted" retraining.
- **Recency weighting.** New production rows get the same total weight as the
  history, so the challenger adapts to concept drift without discarding history.
  Without it, a strong concept change was out-voted by historical rows in testing.
- **Fixed statistical tests.** KS (numeric) and chi-square (categorical) at
  p < 0.05 for every window size, so the threshold always means the same thing.
- **Real served data for monitoring.** The API logs synchronously (a few ms)
  so `/feedback` can always find the request id; logging failures never fail a prediction.
- **Hot reload by polling.** The API re-resolves the alias every 15–30 s and
  swaps models under a lock; `POST /model/reload` forces it. No restarts, no webhooks.
- **MinIO image.** Official MinIO Docker images were discontinued in late 2025; the
  stack uses the maintained community build `pgsty/minio` (pinned, overridable via `MINIO_IMAGE`).
- **Small things that bit:** SQLAlchemy 2.1 defaults to the psycopg 3 driver;
  MLflow ≥ 3.5 rejects unknown `Host` headers (`--allowed-hosts`); registry
  versions are ints (normalized to strings — otherwise the API reloaded the model on every poll).

## Limitations

- **Tiny dataset** (918 patients) → noisy metrics. The demo's promotion margin
  (+0.010 ROC-AUC on 80 hold-out rows) is within sampling noise; a real system
  should require a bootstrap confidence interval or a DeLong test before promoting.
- **Simulated production.** Traffic, drift and delayed labels are simulated from
  held-back UCI rows; real label delay and partial feedback are only approximated
  (`--label-fraction`).
- **Security is local-dev grade:** no API authentication (including
  `/model/reload`), no TLS, MLflow without auth, default credentials in `.env.example`.
- **Single node, no orchestration.** Retraining runs on demand or in CI with a
  file-based store; there is no scheduler talking to the live compose stack, and
  no Prometheus/Grafana server is bundled (the `/metrics` endpoints are ready to scrape).
- **Unbounded training history** grows with every retrain (fine at this scale).
- Images are ~950 MB, dominated by SciPy / scikit-learn / pandas.
- The dataset is old, small and ~79 % male; no fairness analysis is included.

## Future improvements

- Prometheus + Grafana + Alertmanager in compose, alerting on `monitor_retrain_recommended`.
- An orchestrator (Prefect / Dagster / Airflow) to run `src.retrain` against the live stack.
- Statistical promotion gate (bootstrap CI / DeLong), shadow or canary rollout of the challenger.
- Subgroup / fairness metrics (sex, age bands) and probability calibration + threshold tuning.
- DVC with a MinIO remote, or lakeFS, for full data versioning; a feature store (Feast).
- Auth (API keys / OAuth2), TLS, secrets management; Kubernetes (kind + KServe) deployment.

---

## Repository layout

```
ml-pipeline/
├── config.yaml                 # single source of configuration
├── data/                       # git-ignored; README + download_data.sh committed
├── src/
│   ├── config.py               # pydantic-settings: YAML + env overrides
│   ├── logging_utils.py        # structlog (console / JSON)
│   ├── columns.py, schemas.py  # column definitions, pandera contracts
│   ├── ingest.py               # 1 · download, validate, split, manifest
│   ├── preprocess.py           # 2 · cleaning shared by train and serve
│   ├── features.py             # 3 · feature engineering + model pipeline
│   ├── train.py, evaluate.py   # 4/5 · tuning, metrics, MLflow logging
│   ├── registry.py             # 6 · registration, aliases, data snapshots
│   ├── store.py                # prediction log (SQLite / Postgres)
│   ├── drift.py                # 8 · Evidently drift/performance + decision logic
│   └── retrain.py              # 9 · retraining + promotion gate
├── api/                        # 7 · FastAPI app, schemas, model service, Dockerfile
├── monitor/                    # 8 · dashboard app + Dockerfile
├── scripts/                    # simulate_drift, send_sample_requests, export_model,
│                               # capture_screenshots (README images)
├── docker/                     # mlflow + pipeline images, postgres init script
├── docs/images/                # README screenshots
├── hf_space/                   # Hugging Face Space Dockerfile + metadata
├── tests/                      # pytest suite (no network / services needed)
├── .github/workflows/          # ci.yml, retrain.yml
├── docker-compose.yml, Makefile, pyproject.toml
├── requirements.txt, requirements-dev.txt, requirements-docs.txt (screenshots only)
└── .env.example, .pre-commit-config.yaml, .gitignore, .dockerignore
```

Dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
*Heart Disease* [Dataset]. UCI Machine Learning Repository. CC BY 4.0.
