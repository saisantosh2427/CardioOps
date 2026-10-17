"""API endpoint tests with FastAPI's TestClient against a real (SQLite) registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas import EXAMPLE_PATIENT
from src.config import Settings
from src.store import PredictionStore
from tests.conftest import make_settings

pytestmark = pytest.mark.slow


@pytest.fixture
def api_settings(trained_env: dict[str, Any], tmp_path: Path) -> Settings:
    """Registry from the trained env, fresh prediction log per test."""
    base: Settings = trained_env["settings"]
    return base.model_copy(
        update={
            "monitoring": base.monitoring.model_copy(
                update={"database_url": f"sqlite:///{(tmp_path / 'preds.db').as_posix()}"}
            ),
            "api": base.api.model_copy(update={"max_batch_size": 5}),
        }
    )


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient, trained_env: dict[str, Any]) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == trained_env["version"]


def test_predict_returns_probability_and_logs_request(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post("/predict", json=EXAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == int(body["probability"] >= 0.5)
    assert body["risk_label"] == ("high" if body["prediction"] else "low")

    logged = PredictionStore(api_settings.database_url).load_recent(10)
    assert logged["request_id"].tolist() == [body["request_id"]]
    assert logged.loc[0, "age"] == EXAMPLE_PATIENT["age"]


def test_predict_accepts_missing_optional_fields(client: TestClient) -> None:
    response = client.post("/predict", json={"age": 50, "sex": 0, "cp": 3, "chol": None})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {**EXAMPLE_PATIENT, "age": 7},
        {**EXAMPLE_PATIENT, "cp": 5},
        {**EXAMPLE_PATIENT, "chol": 10_000},
        {**EXAMPLE_PATIENT, "unknown_field": 1},
        {"sex": 1, "cp": 4},
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_predict(client: TestClient) -> None:
    instances = [EXAMPLE_PATIENT, {**EXAMPLE_PATIENT, "age": 40, "cp": 1, "oldpeak": 0.0}]
    response = client.post("/predict/batch", json={"instances": instances})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert len({p["request_id"] for p in predictions}) == 2


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    too_many = {"instances": [EXAMPLE_PATIENT] * 6}
    assert client.post("/predict/batch", json=too_many).status_code == 413


def test_feedback_attaches_ground_truth(client: TestClient, api_settings: Settings) -> None:
    request_id = client.post("/predict", json=EXAMPLE_PATIENT).json()["request_id"]
    response = client.post(
        "/feedback",
        json={"items": [{"request_id": request_id, "actual": 1}, {"request_id": "x", "actual": 0}]},
    )
    assert response.json() == {"updated": 1, "unknown_request_ids": ["x"]}
    labeled = PredictionStore(api_settings.database_url).load_recent(10, labeled_only=True)
    assert labeled["target"].tolist() == [1]


def test_model_info(client: TestClient, trained_env: dict[str, Any]) -> None:
    body = client.get("/model-info").json()
    assert body["model_version"] == trained_env["version"]
    assert body["source"] == "registry"
    assert body["alias"] == "champion"
    assert body["model_type"] == trained_env["outcome"].best.name
    assert "age" in body["features"]


def test_metrics_endpoint_exposes_prometheus_series(client: TestClient) -> None:
    client.post("/predict", json=EXAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "model_predictions_total" in text
    assert 'http_requests_total{method="POST",route="/predict",status="200"}' in text
    assert "model_info{" in text


def test_reload_endpoint(client: TestClient, trained_env: dict[str, Any]) -> None:
    body = client.post("/model/reload").json()
    assert body["model_version"] == trained_env["version"]
    assert body["reloaded"] is False  # alias did not move


def test_serves_exported_model_without_registry(
    trained_env: dict[str, Any], api_settings: Settings, tmp_path: Path
) -> None:
    """Hugging Face Spaces path: model baked into the image, no MLflow server."""
    from scripts.export_model import export_champion

    model_dir = tmp_path / "model"
    export_champion(trained_env["settings"], model_dir)
    offline = api_settings.model_copy(
        update={
            "api": api_settings.api.model_copy(update={"model_uri": str(model_dir)}),
            "mlflow": api_settings.mlflow.model_copy(
                update={"tracking_uri": "http://127.0.0.1:9/unreachable"}
            ),
        }
    )
    with TestClient(create_app(offline)) as client:
        info = client.get("/model-info").json()
        assert info["source"] == "local"
        assert info["model_version"] == trained_env["version"]
        assert client.post("/predict", json=EXAMPLE_PATIENT).status_code == 200


def test_service_degrades_gracefully_without_champion(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)  # empty registry
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health == {**health, "status": "degraded", "model_loaded": False}
        assert client.post("/predict", json=EXAMPLE_PATIENT).status_code == 503
        assert client.get("/model-info").status_code == 503
