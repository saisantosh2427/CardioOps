"""Prediction log store."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.schemas import EXAMPLE_PATIENT
from src.columns import FEATURE_COLUMNS, TARGET
from src.store import PredictionRecord, PredictionStore


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(f"sqlite:///{(tmp_path / 'nested' / 'preds.db').as_posix()}")


def _records(n: int, start: int = 0) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            request_id=f"req-{i}",
            model_name="m",
            model_version="1",
            features={**EXAMPLE_PATIENT, "age": 30 + i, "chol": None},
            prediction=i % 2,
            probability=0.1 * (i % 10),
        )
        for i in range(start, start + n)
    ]


def test_log_and_load_recent_window(store: PredictionStore) -> None:
    store.log_predictions(_records(5))
    store.log_predictions(_records(5, start=5))
    recent = store.load_recent(limit=4)
    assert recent["request_id"].tolist() == ["req-6", "req-7", "req-8", "req-9"]  # oldest first
    assert set(FEATURE_COLUMNS) <= set(recent.columns)
    assert recent["age"].tolist() == [36, 37, 38, 39]
    assert recent["chol"].isna().all()
    assert recent[TARGET].isna().all()


def test_feedback_and_labeled_only(store: PredictionStore) -> None:
    store.log_predictions(_records(4))
    updated, unknown = store.add_feedback([("req-1", 1), ("req-3", 0), ("missing", 1)])
    assert updated == 2
    assert unknown == ["missing"]
    labeled = store.load_recent(limit=10, labeled_only=True)
    assert labeled["request_id"].tolist() == ["req-1", "req-3"]
    assert labeled[TARGET].tolist() == [1, 0]
    assert store.counts() == {"total": 4, "labeled": 2}


def test_empty_store_returns_typed_empty_frame(store: PredictionStore) -> None:
    empty = store.load_recent(limit=10)
    assert empty.empty
    assert TARGET in empty.columns
