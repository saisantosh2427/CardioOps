# syntax=docker/dockerfile:1
# Hugging Face Docker Space: the API serving a champion baked into the image.
# Build context is the folder produced by `python -m scripts.export_model`.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PIPELINE_PROJECT__LOG_FORMAT=json \
    PIPELINE_API__MODEL_URI=/app/model \
    PIPELINE_API__REFRESH_INTERVAL_SECONDS=0 \
    PIPELINE_MONITORING__DATABASE_URL=sqlite:////app/data/predictions.db \
    PORT=7860

# Spaces run containers as UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml ./
COPY src/ ./src/
COPY api/ ./api/
COPY model/ ./model/
# Prediction log (SQLite) is ephemeral on a free Space.
RUN mkdir -p /app/data && chown user /app/data

USER user
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
