# syntax=docker/dockerfile:1
# Heart Disease Risk API - slim multi-stage image, runs as a non-root user.
# Build from the repository root:  docker build -f api/Dockerfile -t heart-api .

ARG PYTHON_VERSION=3.11

# ---- builder: install dependencies into an isolated virtualenv ----
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---- runtime: only the venv + application code ----
FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service serving the MLflow champion model" \
      org.opencontainers.image.licenses="MIT"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PIPELINE_PROJECT__LOG_FORMAT=json \
    MLFLOW_HTTP_REQUEST_MAX_RETRIES=2 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=15 \
    PORT=8000

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Code stays root-owned (read-only for the app user); only data/ is writable
# (used for the SQLite prediction log when no Postgres URL is configured).
COPY config.yaml ./
COPY src/ ./src/
COPY api/ ./api/
RUN mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
