"""Monitoring dashboard over a populated prediction log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from monitor.app import create_app
from scripts.send_sample_requests import row_to_payload
from scripts.simulate_drift import apply_drift
from src.columns import TARGET
from src.config import Settings
from src.registry import load_model
from src.store import PredictionRecord, PredictionStore
from tests.conftest import make_synthetic_heart

pytestmark = pytest.mark.slow


def _monitor_settings(env: dict[str, Any], tmp_path: Path) -> Settings:
    base: Settings = env["settings"]
    return base.model_copy(
        update={
            "monitoring": base.monitoring.model_copy(
                update={
                    "database_url": f"sqlite:///{(tmp_path / 'preds.db').as_posix()}",
                    "report_cache_seconds": 0,
                }
            ),
            "paths": base.paths.model_copy(update={"reports_dir": tmp_path / "reports"}),
        }
    )


def _log_traffic(settings: Settings, env: dict[str, Any], frame: pd.DataFrame) -> None:
    """Log predictions the way the API does, then attach ground truth."""
    model = load_model(settings.champion_model_uri)
    proba = model.predict_proba(frame.drop(columns=[TARGET, "source"]))[:, 1]
    store = PredictionStore(settings.database_url)
    records = [
        PredictionRecord(
            request_id=f"r{i}",
            model_name=settings.mlflow.registered_model_name,
            model_version=env["version"],
            features=row_to_payload(row),
            prediction=int(p >= 0.5),
            probability=float(p),
        )
        for i, ((_, row), p) in enumerate(zip(frame.iterrows(), proba, strict=True))
    ]
    store.log_predictions(records)
    store.add_feedback([(f"r{i}", int(t)) for i, t in enumerate(frame[TARGET])])


def test_dashboard_detects_drift_on_logged_traffic(
    trained_env: dict[str, Any], tmp_path: Path
) -> None:
    settings = _monitor_settings(trained_env, tmp_path)
    _log_traffic(settings, trained_env, apply_drift(make_synthetic_heart(150, seed=31)))

    with TestClient(create_app(settings)) as client:
        summary = client.get("/api/summary").json()
        assert summary["status"] == "ok"
        assert summary["window_size"] == 150
        assert summary["drift"]["dataset_drift"] is True
        assert summary["performance"]["n_labeled"] == 150
        assert summary["decision"]["retrain"] is True

        page = client.get("/")
        assert page.status_code == 200
        assert "Drifted features" in page.text
        for report in ("/reports/drift", "/reports/performance"):
            response = client.get(report)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")

        metrics = client.get("/metrics").text
        assert "monitor_drift_share" in metrics
        assert "monitor_retrain_recommended 1.0" in metrics


def test_dashboard_without_traffic(trained_env: dict[str, Any], tmp_path: Path) -> None:
    settings = _monitor_settings(trained_env, tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/summary").json()["status"] == "no_data"
        assert "No predictions logged yet" in client.get("/").text
        assert client.get("/reports/drift").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}
