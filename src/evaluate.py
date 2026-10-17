"""Model evaluation: metrics, diagnostic plots and champion/challenger comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Binary classification metrics from positive-class probabilities.

    ROC-AUC is NaN when ``y_true`` contains a single class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_proba) if np.unique(y_true).size == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
    }


def predict_positive_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """Probability of heart disease for each row."""
    return model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]


def evaluate_model(
    model: object, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all metrics for a fitted model on ``(X, y)``."""
    return compute_metrics(y, predict_positive_proba(model, X), threshold)


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Confusion matrix as a flat, JSON-friendly dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _pyplot() -> ModuleType:
    """Import pyplot lazily with a headless backend (containers/CI have no display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> Path:
    """Render a confusion matrix PNG."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["no disease", "disease"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, path: Path, title: str) -> Path:
    """Render a ROC curve PNG."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a champion vs challenger comparison on the same held-out data."""

    promote: bool
    metric: str
    challenger_value: float
    champion_value: float
    delta: float
    reason: str


def compare_models(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    metric: str,
    tiebreak_metric: str | None = None,
    min_improvement: float = 0.0,
) -> ComparisonResult:
    """Decide whether the challenger should replace the champion.

    The challenger wins if it beats the champion on ``metric`` by more than
    ``min_improvement``. On an exact tie (only possible with
    ``min_improvement == 0``) the ``tiebreak_metric`` decides. A NaN challenger
    score never wins; with no champion the challenger always wins.
    """
    chal_value = challenger.get(metric, float("nan"))
    if champion is None:
        return ComparisonResult(True, metric, chal_value, float("nan"), float("nan"), "no champion")
    champ_value = champion.get(metric, float("nan"))
    if math.isnan(chal_value):
        return ComparisonResult(
            False, metric, chal_value, champ_value, float("nan"), "challenger metric is NaN"
        )
    if math.isnan(champ_value):
        return ComparisonResult(
            True, metric, chal_value, champ_value, float("nan"), "champion metric is NaN"
        )

    delta = chal_value - champ_value
    if delta > min_improvement + 1e-12:
        reason = f"{metric} improved by {delta:+.4f} (> {min_improvement})"
        return ComparisonResult(True, metric, chal_value, champ_value, delta, reason)
    if abs(delta) <= 1e-12 and min_improvement <= 0 and tiebreak_metric:
        tb_delta = challenger.get(tiebreak_metric, float("nan")) - champion.get(
            tiebreak_metric, float("nan")
        )
        if tb_delta > 1e-12:
            reason = f"{metric} tied; {tiebreak_metric} improved by {tb_delta:+.4f}"
            return ComparisonResult(True, metric, chal_value, champ_value, delta, reason)
    reason = f"{metric} delta {delta:+.4f} does not exceed min_improvement {min_improvement}"
    return ComparisonResult(False, metric, chal_value, champ_value, delta, reason)
