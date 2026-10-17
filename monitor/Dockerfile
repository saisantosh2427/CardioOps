# syntax=docker/dockerfile:1
# Monitoring dashboard (Evidently reports over served traffic), non-root.
# Build from the repository root:  docker build -f monitor/Dockerfile -t heart-monitor .

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY monitor/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="heart-disease-monitor" \
      org.opencontainers.image.description="Evidently drift/performance dashboard" \
      org.opencontainers.image.licenses="MIT"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_HTTP_REQUEST_MAX_RETRIES=2 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=15 \
    PIPELINE_PROJECT__LOG_FORMAT=json

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml ./
COPY src/ ./src/
COPY monitor/ ./monitor/
# Generated HTML reports are written to reports/ (the only writable path).
RUN mkdir -p /app/reports /app/data && chown app:app /app/reports /app/data

USER app
EXPOSE 8050
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=4)" || exit 1

CMD ["uvicorn", "monitor.app:app", "--host", "0.0.0.0", "--port", "8050", "--no-access-log"]
