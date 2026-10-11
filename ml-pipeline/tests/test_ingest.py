"""Ingestion with the network mocked out: download -> validate -> split -> manifest."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
import requests

from src import ingest
from src.columns import FEATURE_COLUMNS, TARGET
from src.config import Settings
from tests.conftest import make_synthetic_heart


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def _uci_text(seed: int) -> str:
    """Render synthetic rows in the raw UCI file format ('?' for missing)."""
    df = make_synthetic_heart(60, seed=seed)
    df["num"] = df[TARGET] * 2
    return df[[*FEATURE_COLUMNS, "num"]].to_csv(header=False, index=False, na_rep="?")


@pytest.fixture
def mocked_uci(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_get(url: str, timeout: int) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(_uci_text(len(calls)))

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    return calls


def test_run_ingest_end_to_end(settings: Settings, mocked_uci: list[str]) -> None:
    result = ingest.run_ingest(settings)

    assert len(mocked_uci) == 4  # one request per hospital file
    assert result.source == "uci"
    assert result.n_raw_rows == 240
    assert len(result.production_batches) == settings.data.n_production_batches
    reference = pd.read_parquet(settings.reference_path)
    assert len(reference) == result.n_reference_rows
    assert set(reference["source"]) == set(settings.data.uci_files)

    batches = [pd.read_parquet(settings.resolve(p)) for p in result.production_batches]
    assert sum(len(b) for b in batches) + len(reference) == result.n_clean_rows
    manifest = json.loads((settings.raw_dir / ingest.MANIFEST_FILENAME).read_text())
    assert manifest["raw_sha256"] == ingest.sha256_file(settings.raw_dir / ingest.RAW_FILENAME)


def test_run_ingest_is_cached_unless_forced(settings: Settings, mocked_uci: list[str]) -> None:
    ingest.run_ingest(settings)
    ingest.run_ingest(settings)
    assert len(mocked_uci) == 4
    ingest.run_ingest(settings, force=True)
    assert len(mocked_uci) == 8


def test_split_is_stratified_and_reproducible(settings: Settings) -> None:
    clean = make_synthetic_heart(500, seed=3)
    ref_a, batches_a = ingest.split_reference_production(clean, settings)
    ref_b, _ = ingest.split_reference_production(clean, settings)
    pd.testing.assert_frame_equal(ref_a, ref_b)
    assert len(ref_a) == pytest.approx(500 * settings.data.reference_fraction, abs=1)
    assert ref_a[TARGET].mean() == pytest.approx(clean[TARGET].mean(), abs=0.02)
    assert len(batches_a) == settings.data.n_production_batches


def test_download_falls_back_to_openml(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_get(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("UCI unreachable")

    fallback = make_synthetic_heart(30, seed=4).assign(num=0)
    monkeypatch.setattr(ingest.requests, "get", failing_get)
    monkeypatch.setattr(ingest, "download_openml", lambda s: fallback)
    frame, source = ingest.download_raw(settings)
    assert source == "openml"
    assert frame is fallback
