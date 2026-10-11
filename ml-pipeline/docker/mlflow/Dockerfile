# syntax=docker/dockerfile:1
# MLflow tracking server + model registry.
# Backend store: Postgres. Artifacts: MinIO (S3 API), proxied by the server
# (--serve-artifacts) so clients only ever need the tracking URL.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

# SQLAlchemy 2.1 defaults to the psycopg (v3) driver for PostgreSQL.
RUN pip install mlflow==3.17.0 "psycopg[binary]==3.3.6" boto3==1.43.111

RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --create-home --home-dir /mlflow mlflow
USER mlflow
WORKDIR /mlflow

EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)" || exit 1

# The full `mlflow server ...` command is set in docker-compose.yml.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
