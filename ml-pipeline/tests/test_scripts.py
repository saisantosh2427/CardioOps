"""Demo scripts: drift simulation and replaying traffic through the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas import PatientFeatures
from scripts.send_sample_requests import row_to_payload, send_batch
from scripts.simulate_drift import apply_drift
from src.columns import FEATURE_COLUMNS, TARGET
from src.config import Settings
from src.store import PredictionStore
from tests.conftest import make_synthetic_heart


def test_zero_intensity_is_identity() -> None:
    df = make_synthetic_heart(50, seed=1)
    pd.testing.assert_frame_equal(apply_drift(df, intensity=0.0), df, check_dtype=False)


def test_drift_shifts_in_documented_directions() -> None:
    df = make_synthetic_heart(300, seed=1)
    out = apply_drift(df)
    assert out["age"].mean() - df["age"].mean() == pytest.approx(7, abs=1)
    assert out["thalach"].mean() < df["thalach"].mean() - 20
    assert out["oldpeak"].mean() > df["oldpeak"].mean() * 1.5
    pd.testing.assert_series_equal(out[TARGET], df[TARGET])  # labels are real, never altered


def test_drifted_rows_remain_valid_api_payloads() -> None:
    out = apply_drift(make_synthetic_heart(200, seed=2), intensity=3.0)
    for _, row in out.iterrows():
        PatientFeatures(**row_to_payload(row))  # raises if out of the API contract


def test_row_to_payload_converts_missing_and_codes() -> None:
    row = pd.Series({**{c: 1.0 for c in FEATURE_COLUMNS}, "age": 54.0, "chol": np.nan})
    payload = row_to_payload(row)
    assert payload["chol"] is None
    assert payload["age"] == 54 and isinstance(payload["age"], int)
    assert isinstance(payload["cp"], int)
    assert isinstance(payload["oldpeak"], float)


@pytest.mark.slow
def test_send_batch_logs_predictions_and_feedback(
    trained_env: dict[str, Any], tmp_path: Path
) -> None:
    base: Settings = trained_env["settings"]
    settings = base.model_copy(
        update={
            "monitoring": base.monitoring.model_copy(
                update={"database_url": f"sqlite:///{(tmp_path / 'p.db').as_posix()}"}
            )
        }
    )
    frame = make_synthetic_heart(60, seed=5)
    with TestClient(create_app(settings)) as client:
        summary = send_batch(
            frame, "http://testserver", chunk_size=25, label_fraction=0.5, session=client
        )
    assert summary["sent"] == 60
    assert 0 < summary["labels_sent"] < 60
    assert 0 <= summary["served_accuracy"] <= 1
    counts = PredictionStore(settings.database_url).counts()
    assert counts == {"total": 60, "labeled": summary["labels_sent"]}
