"""Shared fixtures: synthetic data and fully isolated settings (no network, no servers).

Every test gets its own SQLite MLflow store and SQLite prediction log under
``tmp_path``; the tests never touch ``data/`` or a running MLflow server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from src.columns import FEATURE_COLUMNS, SOURCE, TARGET
from src.config import Settings

FAST_CANDIDATES: dict[str, Any] = {
    "logistic_regression": {
        "enabled": True,
        "param_grid": {"model__C": [0.1, 1.0], "model__class_weight": [None]},
    },
    "random_forest": {
        "enabled": True,
        "param_grid": {
            "model__n_estimators": [30],
            "model__max_depth": [4],
            "model__min_samples_leaf": [3],
        },
    },
    "gradient_boosting": {
        "enabled": False,
        "param_grid": {
            "model__n_estimators": [20],
            "model__learning_rate": [0.1],
            "model__max_depth": [2],
        },
    },
}


def make_synthetic_heart(n: int = 400, seed: int = 0, missing_rate: float = 0.05) -> pd.DataFrame:
    """Clean-schema frame with realistic marginals and a learnable signal."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n).astype(float)
    sex = rng.binomial(1, 0.7, n).astype(float)
    cp = rng.choice([1, 2, 3, 4], n, p=[0.08, 0.17, 0.25, 0.5]).astype(float)
    trestbps = rng.normal(132, 18, n).clip(90, 200).round()
    chol = rng.normal(240, 50, n).clip(120, 560).round()
    fbs = rng.binomial(1, 0.15, n).astype(float)
    restecg = rng.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2]).astype(float)
    thalach = (210 - 0.8 * age + rng.normal(0, 18, n)).clip(70, 202).round()
    exang = rng.binomial(1, 0.35, n).astype(float)
    oldpeak = rng.gamma(1.2, 0.8, n).round(1).clip(0, 6.2)
    slope = rng.choice([1, 2, 3], n, p=[0.45, 0.45, 0.1]).astype(float)
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]).astype(float)
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.07, 0.38]).astype(float)
    logit = (
        -6.0
        + 0.04 * age
        + 0.9 * sex
        + 1.4 * (cp == 4)
        - 0.03 * (thalach - 150)
        + 0.8 * oldpeak
        + 1.1 * exang
        + 0.8 * ca
        + 1.2 * (thal == 7)
    )
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    )
    for col in ("chol", "ca", "thal", "slope"):
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    df[TARGET] = target.astype("int64")
    df[SOURCE] = "synthetic"
    return df[[*FEATURE_COLUMNS, TARGET, SOURCE]]


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings fully isolated in ``tmp_path`` (ignores any developer .env file)."""
    base: dict[str, Any] = {
        "paths": {
            "raw_dir": tmp_path / "data" / "raw",
            "processed_dir": tmp_path / "data" / "processed",
            "production_dir": tmp_path / "data" / "production",
            "reports_dir": tmp_path / "reports",
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "experiment_name": "test-experiment",
        },
        "monitoring": {
            "database_url": f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}",
            "window_size": 200,
            "min_current_samples": 20,
            "min_labeled_samples": 20,
        },
        "retrain": {"min_labeled_samples": 30},
        "training": {"cv_folds": 3, "n_jobs": 1, "candidates": FAST_CANDIDATES},
        "api": {"refresh_interval_seconds": 0},
    }
    for key, value in overrides.items():
        base[key] = {**base.get(key, {}), **value}
    return Settings(_env_file=None, **base)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove PIPELINE_* variables from the developer's shell."""
    for key in list(os.environ):
        if key.startswith("PIPELINE_") or key == "MLFLOW_TRACKING_URI":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_heart(400, seed=0)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    # MLflow's SQLite store writes artifacts to ./mlruns: keep them in tmp_path.
    monkeypatch.chdir(tmp_path)
    return make_settings(tmp_path)


def train_champion(tmp_path: Path, settings: Settings | None = None) -> dict[str, Any]:
    """Train, register and promote a champion into the isolated store under ``tmp_path``."""
    from src.registry import configure_mlflow
    from src.train import register_and_maybe_promote, split_xy, train_candidates

    cwd = os.getcwd()
    os.chdir(tmp_path)  # MLflow's SQLite store writes artifacts to ./mlruns
    try:
        settings = settings or make_settings(tmp_path)
        reference = make_synthetic_heart(400, seed=1)
        train_df, test_df = train_test_split(
            reference, test_size=0.25, stratify=reference[TARGET], random_state=0
        )
        configure_mlflow(settings)
        outcome = train_candidates(
            train_df, test_df, settings, run_name="test-training", reference_df=reference
        )
        X_test, y_test = split_xy(test_df)
        version, comparison = register_and_maybe_promote(
            outcome.best, X_test, y_test, settings, trigger="test"
        )
    finally:
        os.chdir(cwd)
    return {
        "settings": settings,
        "tmp_path": tmp_path,
        "reference": reference,
        "test_df": test_df,
        "outcome": outcome,
        "version": version,
        "comparison": comparison,
    }


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A trained champion shared by the tests of one module (training is the slow part)."""
    return train_champion(tmp_path_factory.mktemp("trained"))
