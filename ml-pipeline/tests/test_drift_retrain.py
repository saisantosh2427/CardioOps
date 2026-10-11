"""Drift detection, the retrain decision logic, and the full retrain/promotion loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_drift import apply_drift
from src.columns import TARGET
from src.config import Settings
from src.drift import (
    DriftResult,
    PerformanceResult,
    baseline_metrics_from_tags,
    check_performance,
    decide_retrain,
    run_drift_report,
)
from src.registry import configure_mlflow, get_champion, get_version_by_alias
from src.retrain import run_retrain, split_labeled_current
from tests.conftest import make_settings, make_synthetic_heart, train_champion


def _drift(dataset_drift: bool, n_current: int = 200, share: float = 0.5) -> DriftResult:
    drifted = ["age", "chol"] if dataset_drift else []
    return DriftResult(dataset_drift, share, len(drifted), 13, drifted, {}, 400, n_current)


def _perf(degraded: bool) -> PerformanceResult:
    return PerformanceResult("roc_auc", 100, {"roc_auc": 0.7}, 0.9, degraded, 0.2)


# ----------------------------------------------------------------- decision logic


@pytest.mark.parametrize(
    ("drift", "perf", "force", "expected", "reason_fragment"),
    [
        (_drift(False), _perf(False), False, False, "no drift"),
        (_drift(True), _perf(False), False, True, "data drift"),
        (_drift(False), _perf(True), False, True, "performance drop"),
        (_drift(True), _perf(True), False, True, "performance drop"),
        (_drift(False), None, False, False, "no drift"),
        (_drift(True, n_current=5), _perf(True), False, False, "not enough production data"),
        (None, None, False, False, "not enough production data"),
        (_drift(False), _perf(False), True, True, "forced"),
    ],
)
def test_decide_retrain(
    settings: Settings,
    drift: DriftResult | None,
    perf: PerformanceResult | None,
    force: bool,
    expected: bool,
    reason_fragment: str,
) -> None:
    decision = decide_retrain(drift, perf, settings, force=force)
    assert decision.retrain is expected
    assert any(reason_fragment in r for r in decision.reasons)


def test_check_performance_requires_enough_labels(settings: Settings) -> None:
    result = check_performance([1, 0, 1], [0.9, 0.1, 0.8], {"roc_auc": 0.9}, settings)
    assert result.current is None
    assert result.degraded is False


def test_check_performance_flags_drop_beyond_tolerance(settings: Settings) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 200), 0, 1)
    noisy = rng.random(200)
    baseline = {"roc_auc": 0.95}
    assert not check_performance(y, good, baseline, settings).degraded
    degraded = check_performance(y, noisy, baseline, settings)
    assert degraded.degraded
    assert degraded.drop == pytest.approx(0.95 - degraded.current["roc_auc"], abs=1e-3)


def test_baseline_metrics_from_tags() -> None:
    tags = {"holdout_roc_auc": "0.9100", "holdout_f1": "0.8", "model_type": "rf"}
    assert baseline_metrics_from_tags(tags) == {"roc_auc": 0.91, "f1": 0.8}


# ----------------------------------------------------------------- drift detection


def test_no_drift_on_same_distribution(settings: Settings) -> None:
    reference = make_synthetic_heart(400, seed=10)
    current = make_synthetic_heart(200, seed=11)
    result = run_drift_report(reference, current, settings)
    assert result.dataset_drift is False
    assert result.n_columns == 13
    assert result.drift_share < settings.monitoring.drift_share_threshold


def test_drift_detected_on_simulated_new_site(settings: Settings, tmp_path: Path) -> None:
    reference = make_synthetic_heart(400, seed=10)
    current = apply_drift(make_synthetic_heart(200, seed=11))
    html = tmp_path / "drift.html"
    result = run_drift_report(reference, current, settings, html_path=html)
    assert result.dataset_drift is True
    assert {"age", "thalach", "oldpeak"} <= set(result.drifted_columns)
    assert result.n_drifted == len(result.drifted_columns)
    assert html.stat().st_size > 10_000


def test_holdout_split_is_stratified(settings: Settings) -> None:
    labeled = make_synthetic_heart(200, seed=3)
    train, holdout = split_labeled_current(labeled, settings)
    assert len(holdout) == pytest.approx(200 * settings.retrain.holdout_fraction, abs=1)
    assert holdout[TARGET].mean() == pytest.approx(labeled[TARGET].mean(), abs=0.05)
    assert len(train) + len(holdout) == 200


# ----------------------------------------------------------------- retrain loop (slow)


def _write_current(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "current.parquet"
    frame.to_parquet(path, index=False)
    return path


def _fresh(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return train_champion(tmp_path, make_settings(tmp_path, **overrides))


def _retrain(env: dict[str, Any], path: Path, **kwargs: Any) -> dict[str, Any]:
    return run_retrain(env["settings"], source="file", current_path=path, **kwargs)


@pytest.mark.slow
def test_retrain_skipped_when_no_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = _fresh(tmp_path, monitoring={"max_metric_drop": 0.3})
    path = _write_current(tmp_path, make_synthetic_heart(200, seed=21))
    summary = _retrain(env, path)
    assert summary["action"] == "none"
    assert summary["decision"]["retrain"] is False
    assert summary["champion_version_after"] == "1"
    saved = json.loads((env["settings"].reports_dir / "retrain_summary.json").read_text())
    assert saved["action"] == "none"


@pytest.mark.slow
def test_retrain_promotes_better_challenger_and_reference_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New site + changed disease relationship: the challenger must win and be promoted."""
    monkeypatch.chdir(tmp_path)
    env = _fresh(tmp_path)
    drifted = apply_drift(make_synthetic_heart(300, seed=22))
    drifted[TARGET] = 1 - drifted[TARGET]  # strong concept drift the champion cannot handle
    path = _write_current(tmp_path, drifted)

    summary = _retrain(env, path)
    settings: Settings = env["settings"]
    client = configure_mlflow(settings)
    assert summary["decision"]["retrain"] is True
    assert summary["action"] == "retrained_promoted"
    assert summary["comparison"]["challenger_value"] > summary["comparison"]["champion_value"]
    assert str(get_champion(client, settings).version) == "2"
    name = settings.mlflow.registered_model_name
    previous = get_version_by_alias(client, name, settings.mlflow.previous_champion_alias)
    assert str(previous.version) == "1"

    # The promoted model's drift reference is the window it adapted to, so the
    # same traffic no longer looks drifted (no endless retraining).
    again = _retrain(env, path)
    assert again["drift"]["dataset_drift"] is False


@pytest.mark.slow
def test_challenger_rejected_when_not_better_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = _fresh(tmp_path, retrain={"min_improvement": 1.0, "min_labeled_samples": 30})
    path = _write_current(tmp_path, make_synthetic_heart(200, seed=23))
    summary = _retrain(env, path, force=True)
    settings: Settings = env["settings"]
    client = configure_mlflow(settings)
    assert summary["action"] == "retrained_rejected"
    assert str(get_champion(client, settings).version) == "1"
    name = settings.mlflow.registered_model_name
    challenger = get_version_by_alias(client, name, settings.mlflow.challenger_alias)
    assert str(challenger.version) == "2"


@pytest.mark.slow
def test_retrain_needs_labels_and_supports_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = _fresh(tmp_path)
    drifted = apply_drift(make_synthetic_heart(200, seed=24))

    unlabeled = _write_current(tmp_path, drifted.drop(columns=[TARGET]))
    summary = _retrain(env, unlabeled)
    assert summary["decision"]["retrain"] is True
    assert summary["action"] == "skipped_insufficient_labels"

    labeled = _write_current(tmp_path, drifted)
    assert _retrain(env, labeled, dry_run=True)["action"] == "dry_run"
    client = configure_mlflow(env["settings"])
    assert str(get_champion(client, env["settings"]).version) == "1"
