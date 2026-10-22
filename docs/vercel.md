# Vercel services deployment

This configuration deploys the two existing web applications. There is no third
root application in this repository. Service names and exposure are pending owner
confirmation before deployment.

| Service | Entrypoint | Public paths |
| --- | --- | --- |
| api | api.vercel:app | /api and /api/* |
| monitor | monitor.vercel:app | / and all remaining paths |

Both services use repository root `.` so that `src/`, `config.yaml`, and pinned
root requirements are available to each independent build. The full runtime
requirements are installed for both services. `.vercelignore` excludes local
environments, recordings, generated artifacts, data and secrets from upload.

Vercel preserves the original URL path. The API wrapper mounts the existing app
at `/api` and forwards lifespan startup/shutdown. Swagger, OpenAPI, redirects and
prediction URLs therefore use the same prefix. Monitor JSON is `/summary` to
avoid `/api/summary` colliding with the API service. The old JSON route remains
available in standalone local/Docker monitoring deployments.

## Service connections and bindings

There are **no service-to-service HTTP calls between api and monitor**. The
dashboard's API docs hyperlink is a browser link, not an internal call. It uses
the public same-origin `/api/docs`. Neither service needs a binding to the other.

Both services call the **external MLflow tracking/registry server** and connect
directly to the **same external Postgres prediction database**. External resources
are not entries in `services`, so Vercel service bindings do not apply to them.
Docker Compose remains the local infrastructure setup. It is not launched by Vercel.

If MLflow is later added as a Vercel service, both callers will need a binding to
that actual service, and tracking configuration must read its injected URL at
runtime. Do not create or manually set an invented binding URL now.

## Required cloud configuration

Set these in Vercel project environment settings for each intended environment:

- `PIPELINE_MLFLOW__TRACKING_URI`: reachable hosted MLflow HTTPS endpoint.
- `PIPELINE_MONITORING__DATABASE_URL`: shared hosted Postgres URL, preferably
  `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`.
- `PIPELINE_MONITORING__MLFLOW_UI_URL`: optional browser-accessible MLflow UI URL.
  If omitted, the local-default link is hidden.
- MLflow authentication environment variables required by your host, such as
  `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, stored as secrets.

The MLflow registry must already have the trained `heart-disease-classifier`
champion and its reference-data artifacts. Configure the MLflow server to proxy
artifacts from persistent object storage, rather than returning Docker-only
MinIO hostnames. Training and retraining remain separate jobs outside these web
services. API alias refresh is request-driven on Vercel; it does not depend on a
continuously running polling task. Reports use temporary scratch storage and are
regenerated per instance; model/traffic state stays in external services.

Cloud entrypoints reject the local MLflow/SQLite defaults. Do not upload the
local `.env`, databases, or model recordings. Use separate external databases
and tracking environments for previews when isolation is required.

## Local verification and deployment

1. Start your local MLflow service and ensure a champion exists. Set
   `PIPELINE_MLFLOW__TRACKING_URI` and `PIPELINE_MONITORING__DATABASE_URL` for
   local development (local SQLite is allowed for `vercel dev`).
2. Use Python 3.12 (selected by `.python-version` for the Vercel runtime), install
   `requirements.txt`, and run `npx vercel dev -L` from
   the repository root. Vercel starts both services together. No binding variables
   need to be supplied for this topology.
3. Check `/`, `/summary`, `/api/docs`, `/api/openapi.json`, `/api/health`, and
   submit a sample request to `/api/predict`.
4. Confirm service names, public exposure and external infrastructure with the
   project owner. Import the repository in Vercel using this directory as the
   project root, add the cloud environment variables, then deploy.

The API includes feedback and model-reload endpoints; the selected `/api/*`
rewrite exposes them as well. The monitor publishes reports and metrics.
Confirm this intended demo exposure before publishing. This configuration does
not provision infrastructure or introduce application authentication.

Reference: https://vercel.com/docs/services

## Verification performed

Validated using Vercel CLI 65.0.0 with `vercel dev -L --listen 3100` and Python
3.12.13. Both services started. Dashboard, summary, API health, Swagger, OpenAPI,
and prefix redirects returned the expected responses. A real `/api/predict`
request returned HTTP 200 and the monitor read its logged prediction from the
shared local store. API, monitor, routing and configuration tests passed in the
existing Python 3.11 test environment; changed Python files passed Ruff checks.
Cloud provisioning, production builds and public deployment have not been performed.
