"""Deployment-specific settings; Docker and local CLI settings stay unchanged."""

import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from src.config import Settings


def deployment_settings() -> Settings:
    settings = Settings()
    # Vercel's filesystem is read-only except for temporary scratch space.
    settings.paths.reports_dir = Path(tempfile.gettempdir()) / "heart-mlops-reports"
    settings.monitoring.api_url = "/api"
    settings.api.refresh_on_request = True
    if os.environ.get("VERCEL") == "1" and os.environ.get("VERCEL_ENV") != "development":
        tracking = urlsplit(settings.mlflow.tracking_uri)
        if tracking.scheme not in {"http", "https"} or tracking.hostname in {
            None,
            "localhost",
            "127.0.0.1",
            "mlflow",
        }:
            raise ValueError("Set PIPELINE_MLFLOW__TRACKING_URI to your hosted MLflow URL")
        if not settings.monitoring.database_url.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("Set PIPELINE_MONITORING__DATABASE_URL to shared hosted Postgres")
    # A browser link is not an internal service call and must use a public URL.
    if settings.monitoring.mlflow_ui_url == "http://localhost:5000":
        settings.monitoring.mlflow_ui_url = ""
    return settings
