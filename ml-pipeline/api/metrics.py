"""Prometheus metrics exposed on ``GET /metrics``."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests by route and status", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency by route",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "model_predictions_total",
    "Predictions served by model version and predicted class",
    ["model_version", "predicted_class"],
)
PREDICTED_PROBABILITY = Histogram(
    "model_predicted_probability",
    "Distribution of predicted P(heart disease) - shifts here hint at drift",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded and serving")
MODEL_INFO = Gauge(
    "model_info", "Currently served model (value is always 1)", ["name", "version", "model_type"]
)
MODEL_RELOADS = Counter("model_reloads_total", "Times a new model version was swapped in")
PREDICTION_LOG_ERRORS = Counter(
    "prediction_log_errors_total", "Failed writes of predictions to the prediction store"
)
FEEDBACK_RECEIVED = Counter("feedback_labels_total", "Ground-truth labels received", ["actual"])


def set_model_info(name: str, version: str, model_type: str | None) -> None:
    """Expose exactly one ``model_info`` series for the currently served model."""
    MODEL_INFO.clear()
    MODEL_INFO.labels(name=name, version=version, model_type=model_type or "unknown").set(1)
    MODEL_LOADED.set(1)
