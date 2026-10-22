"""Public routing and deployment configuration regression tests."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.vercel import create_vercel_app
from monitor.app import create_app as create_monitor
from src.config import Settings
from src.vercel import deployment_settings


def test_mounted_api_docs_and_redirects(monkeypatch):
    settings = Settings()
    settings.api.refresh_on_request = False
    app = create_vercel_app(settings)
    # No lifespan: these routing checks deliberately do not contact MLflow or a DB.
    client = TestClient(app)
    for path in ("/api", "/api/"):
        assert client.get(path, follow_redirects=False).headers["location"] == "/api/docs"
    assert "/api/openapi.json" in client.get("/api/docs").text
    spec = client.get("/api/openapi.json").json()
    assert {"url": "/api"} in spec["servers"]
    assert "/predict" in spec["paths"]
    assert client.get("/api/health").status_code == 200
    assert client.get("/predict").status_code == 404


def test_monitor_links_do_not_collide_with_api(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    settings = deployment_settings()
    app = create_monitor(settings)
    monkeypatch.setattr(app.state.service, "summary", lambda refresh=False: {"status": "no_data"})
    client = TestClient(app)
    page = client.get("/").text
    assert 'href="/api/docs"' in page
    assert 'href="/summary"' in page
    assert 'href="/api/summary"' not in page
    assert "localhost" not in page
    assert client.get("/summary").json()["status"] == "no_data"
    assert client.get("/api/summary").json()["status"] == "no_data"  # local compatibility


def test_deployed_services_require_remote_shared_state(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.setenv("PIPELINE_MLFLOW__TRACKING_URI", "http://localhost:5000")
    with pytest.raises(ValueError, match="hosted MLflow"):
        deployment_settings()
    monkeypatch.setenv("PIPELINE_MLFLOW__TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("PIPELINE_MONITORING__DATABASE_URL", "sqlite:///data/monitoring.db")
    with pytest.raises(ValueError, match="shared hosted Postgres"):
        deployment_settings()
    monkeypatch.setenv("PIPELINE_MONITORING__DATABASE_URL", "postgresql+psycopg://host/db")
    settings = deployment_settings()
    assert settings.api.refresh_on_request
    assert settings.paths.reports_dir.is_absolute()


def test_service_rewrites_are_specific_before_catch_all():
    config = json.loads(Path("vercel.json").read_text())
    assert set(config["services"]) == {"api", "monitor"}
    assert [r["source"] for r in config["rewrites"]] == ["/api", "/api/(.*)", "/(.*)"]
    assert config["rewrites"][-1]["destination"] == {"service": "monitor"}


def test_request_refresh_runs_without_background_worker(monkeypatch):
    from api.main import create_app

    settings = Settings()
    settings.api.refresh_on_request = True
    settings.api.refresh_interval_seconds = 30
    app = create_app(settings)
    calls = []
    monkeypatch.setattr(app.state.model_service, "refresh", lambda: calls.append(True))
    client = TestClient(app)
    client.get("/health")
    client.get("/health")
    assert len(calls) == 1
