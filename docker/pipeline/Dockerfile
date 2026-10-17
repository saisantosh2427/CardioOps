# syntax=docker/dockerfile:1
# Batch-job image (ingest / train / simulate drift / retrain) used by the
# `trainer` service in docker-compose (profile "jobs"), e.g.
#   docker compose run --rm trainer python -m src.train

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
# data/ and reports/ are named volumes in docker-compose; create them owned by
# the app user so the volumes inherit writable ownership.
RUN mkdir -p /app/data /app/reports && chown app:app /app/data /app/reports

USER app
CMD ["python", "-m", "src.train"]
