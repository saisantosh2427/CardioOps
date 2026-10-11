"""Training smoke tests against an isolated SQLite MLflow store."""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
from mlflow import MlflowClient

from src.evaluate import compare_models, compute_metrics
from src.registry import (
    configure_mlflow,
    get_champion,
    get_version_by_alias,
    load_model,
    load_reference_data,
)
from src.train import register_and_maybe_promote, split_xy

pytestmark = pytest.mark.slow


def test_training_logs_candidates_and_registers_champion(trained_env: dict[str, Any]) -> None:
    settings = trained_env["settings"]
    outcome = trained_env["outcome"]
    client = configure_mlflow(settings)

    enabled = [n for n, c in settings.training.candidates.items() if c.enabled]
    assert sorted(c.name for c in outcome.candidates) == sorted(enabled)
    assert outcome.best.cv_mean == max(c.cv_mean for c in outcome.candidates)

    champion = get_champion(client, settings)
    assert champion is not None
    assert str(champion.version) == trained_env["version"] == "1"
    assert champion.run_id == outcome.best.run_id
    assert trained_env["comparison"].reason == "no champion"

    run = client.get_run(outcome.best.run_id)
    for metric in ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"):
        assert 0.0 <= run.data.metrics[metric] <= 1.0
    assert run.data.metrics["test_roc_auc"] > 0.7  # synthetic data has real signal
    artifacts = {a.path for a in client.list_artifacts(outcome.best.run_id, "evaluation")}
    assert {"evaluation/confusion_matrix.png", "evaluation/roc_curve.png"} <= artifacts


def test_champion_loads_and_reference_snapshot_is_logged(trained_env: dict[str, Any]) -> None:
    settings = trained_env["settings"]
    client = configure_mlflow(settings)
    model = load_model(settings.champion_model_uri)
    X_test, y_test = split_xy(trained_env["test_df"])
    metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
    assert math.isclose(
        metrics["roc_auc"], trained_env["outcome"].best.test_metrics["roc_auc"], rel_tol=1e-9
    )
    reference = load_reference_data(get_champion(client, settings).run_id)
    assert reference is not None
    assert len(reference) == len(trained_env["reference"])


def test_identical_model_is_not_promoted(trained_env: dict[str, Any]) -> None:
    """Re-registering an equivalent model ties the champion and must not replace it."""
    settings = trained_env["settings"]
    cwd = os.getcwd()
    os.chdir(trained_env["tmp_path"])
    try:
        X_test, y_test = split_xy(trained_env["test_df"])
        version, comparison = register_and_maybe_promote(
            trained_env["outcome"].best, X_test, y_test, settings, trigger="test-rerun"
        )
    finally:
        os.chdir(cwd)
    client: MlflowClient = configure_mlflow(settings)
    assert version == "2"
    assert not comparison.promote
    assert str(get_champion(client, settings).version) == "1"
    name = settings.mlflow.registered_model_name
    assert str(get_version_by_alias(client, name, settings.mlflow.challenger_alias).version) == "2"


def test_compute_metrics_single_class_auc_is_nan() -> None:
    metrics = compute_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(metrics["roc_auc"])
    assert metrics["recall"] == 1.0


@pytest.mark.parametrize(
    ("challenger", "champion", "min_improvement", "expected"),
    [
        ({"f1": 0.80, "roc_auc": 0.9}, None, 0.0, True),
        ({"f1": 0.81, "roc_auc": 0.9}, {"f1": 0.80, "roc_auc": 0.9}, 0.0, True),
        ({"f1": 0.81, "roc_auc": 0.9}, {"f1": 0.80, "roc_auc": 0.9}, 0.02, False),
        ({"f1": 0.80, "roc_auc": 0.91}, {"f1": 0.80, "roc_auc": 0.90}, 0.0, True),  # tiebreak
        ({"f1": 0.80, "roc_auc": 0.90}, {"f1": 0.80, "roc_auc": 0.90}, 0.0, False),
        ({"f1": 0.70, "roc_auc": 0.99}, {"f1": 0.80, "roc_auc": 0.90}, 0.0, False),
        ({"f1": float("nan"), "roc_auc": 0.9}, {"f1": 0.5, "roc_auc": 0.9}, 0.0, False),
    ],
)
def test_compare_models(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    min_improvement: float,
    expected: bool,
) -> None:
    result = compare_models(challenger, champion, "f1", "roc_auc", min_improvement)
    assert result.promote is expected
