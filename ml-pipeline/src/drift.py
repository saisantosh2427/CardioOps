"""Stage 8 - monitoring: data drift and model performance with Evidently.

*Reference* = the data the current champion was trained on (stored with its
MLflow run). *Current* = the most recent predictions actually served by the
API (prediction log), or a file for offline/CI runs.

``decide_retrain`` is the single, pure decision function used by both the
monitor dashboard and ``src.retrain``.

Usage: ``python -m src.drift [--source db|file] [--current-path PATH]``
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from src.columns import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.config import Settings, get_settings
from src.evaluate import compute_metrics
from src.logging_utils import configure_logging, get_logger
from src.preprocess import clean_features

logger = get_logger(__name__)

Source = Literal["db", "file"]

# Evidently tests whose score is a distance (drift if score >= threshold);
# every other test reports a p-value (drift if p < threshold).
_DISTANCE_METHODS = ("wasserstein", "jensenshannon", "psi", "kl_div", "hellinger", "energy")


@dataclass
class DriftResult:
    """Per-feature statistical tests aggregated into a dataset-level verdict."""

    dataset_drift: bool
    drift_share: float
    n_drifted: int
    n_columns: int
    drifted_columns: list[str]
    column_tests: dict[str, dict[str, Any]]
    n_reference: int
    n_current: int


@dataclass
class PerformanceResult:
    """Champion performance on labeled production traffic vs. its training baseline."""

    metric: str
    n_labeled: int
    current: dict[str, float] | None
    baseline_value: float | None
    degraded: bool
    drop: float | None


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as the model pipeline, so drift is measured on what the model sees."""
    return clean_features(df)


def _drift_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=list(NUMERIC_FEATURES), categorical_columns=list(CATEGORICAL_FEATURES)
    )


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    settings: Settings,
    html_path: Path | None = None,
) -> DriftResult:
    """Run Evidently's data-drift preset; optionally save the interactive HTML report.

    Tests are fixed to Kolmogorov-Smirnov (numeric) and chi-square
    (categorical) so ``stattest_threshold`` always means a p-value cut-off,
    regardless of sample size.
    """
    cfg = settings.monitoring
    ref, cur = _feature_frame(reference), _feature_frame(current)
    report = Report(
        [
            DataDriftPreset(
                columns=FEATURE_COLUMNS,
                drift_share=cfg.drift_share_threshold,
                num_method="ks",
                cat_method="chisquare",
                threshold=cfg.stattest_threshold,
            )
        ]
    )
    definition = _drift_definition()
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur, data_definition=definition),
        reference_data=Dataset.from_pandas(ref, data_definition=definition),
    )
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    share, count, tests = 0.0, 0, {}
    for metric in snapshot.dict()["metrics"]:
        config, value = metric["config"], metric["value"]
        if config["type"].endswith("DriftedColumnsCount"):
            share, count = float(value["share"]), int(value["count"])
        elif config["type"].endswith("ValueDrift"):
            method, threshold = config["method"], float(config["threshold"])
            score = float(value)
            is_distance = any(name in method.lower() for name in _DISTANCE_METHODS)
            drifted = score >= threshold if is_distance else score < threshold
            tests[config["column"]] = {
                "method": method,
                "score": round(score, 6),
                "threshold": threshold,
                "drifted": drifted,
            }
    drifted_columns = sorted(c for c, t in tests.items() if t["drifted"])
    return DriftResult(
        dataset_drift=share >= cfg.drift_share_threshold,
        drift_share=round(share, 4),
        n_drifted=count,
        n_columns=len(tests),
        drifted_columns=drifted_columns,
        column_tests=tests,
        n_reference=len(ref),
        n_current=len(cur),
    )


def run_performance_report(labeled: pd.DataFrame, html_path: Path) -> None:
    """Evidently classification report for labeled production traffic.

    ``labeled`` needs ``target``, ``prediction`` and ``probability`` columns.
    """
    frame = labeled[[TARGET, "prediction", "probability"]].astype(
        {TARGET: "int64", "prediction": "int64", "probability": "float64"}
    )
    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET, prediction_labels="prediction", prediction_probas="probability"
            )
        ]
    )
    snapshot = Report([ClassificationPreset()]).run(
        current_data=Dataset.from_pandas(frame, data_definition=definition), reference_data=None
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))


def check_performance(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    baseline: dict[str, float] | None,
    settings: Settings,
) -> PerformanceResult:
    """Compare current performance on labeled traffic with the champion's hold-out baseline."""
    cfg = settings.monitoring
    metric = cfg.performance_metric
    n = len(y_true)
    baseline_value = (baseline or {}).get(metric)
    if n < cfg.min_labeled_samples:
        return PerformanceResult(metric, n, None, baseline_value, False, None)
    current = compute_metrics(y_true, y_proba, settings.training.decision_threshold)
    drop = None
    degraded = False
    if baseline_value is not None and not math.isnan(current[metric]):
        drop = round(baseline_value - current[metric], 4)
        degraded = drop > cfg.max_metric_drop
    return PerformanceResult(
        metric, n, {k: round(v, 4) for k, v in current.items()}, baseline_value, degraded, drop
    )


def decide_retrain(
    drift: DriftResult | None,
    performance: PerformanceResult | None,
    settings: Settings,
    force: bool = False,
) -> RetrainDecision:
    """Retrain if forced, if the dataset drifted, or if performance degraded."""
    if force:
        return RetrainDecision(True, ["forced by operator"])
    min_current = settings.monitoring.min_current_samples
    if drift is None or drift.n_current < min_current:
        n = 0 if drift is None else drift.n_current
        return RetrainDecision(False, [f"not enough production data ({n} < {min_current})"])
    reasons = []
    if drift.dataset_drift:
        reasons.append(
            f"data drift: {drift.n_drifted}/{drift.n_columns} features drifted "
            f"(share {drift.drift_share:.2f} >= {settings.monitoring.drift_share_threshold}): "
            f"{', '.join(drift.drifted_columns)}"
        )
    if performance is not None and performance.degraded:
        reasons.append(
            f"performance drop: {performance.metric} {performance.baseline_value:.3f} -> "
            f"{(performance.current or {}).get(performance.metric, float('nan')):.3f} "
            f"(drop {performance.drop:.3f} > {settings.monitoring.max_metric_drop})"
        )
    return RetrainDecision(bool(reasons), reasons or ["no drift and no performance drop"])


# --------------------------------------------------------------------------- data access


def baseline_metrics_from_tags(tags: dict[str, str]) -> dict[str, float]:
    """Hold-out metrics recorded as ``holdout_<metric>`` tags at registration time."""
    return {
        k.removeprefix("holdout_"): float(v) for k, v in tags.items() if k.startswith("holdout_")
    }


def load_reference(settings: Settings) -> tuple[pd.DataFrame, Any]:
    """Champion training snapshot (preferred) or the local reference file.

    Returns ``(reference, champion_model_version_or_None)``.
    """
    from src.registry import configure_mlflow, get_champion, load_reference_data

    champion = None
    try:
        client = configure_mlflow(settings)
        champion = get_champion(client, settings)
    except Exception as exc:
        logger.warning("registry_unavailable", error=str(exc))
    if champion is not None:
        reference = load_reference_data(champion.run_id)
        if reference is not None:
            return reference, champion
    if settings.reference_path.exists():
        logger.info("using_local_reference", path=str(settings.reference_path))
        return pd.read_parquet(settings.reference_path), champion
    raise FileNotFoundError("No reference data: train a model or run `python -m src.ingest`.")


def load_current(
    settings: Settings, source: Source = "db", current_path: Path | None = None
) -> pd.DataFrame:
    """Most recent ``window_size`` rows of production data (served or from a file)."""
    window = settings.monitoring.window_size
    if source == "file":
        if current_path is None:
            raise ValueError("--current-path is required with --source file")
        frame = (
            pd.read_parquet(current_path)
            if current_path.suffix == ".parquet"
            else pd.read_csv(current_path)
        )
        return frame.tail(window).reset_index(drop=True)
    from src.store import PredictionStore

    return PredictionStore(settings.database_url).load_recent(window)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a drift check and write an HTML report.")
    parser.add_argument("--source", choices=["db", "file"], default="db")
    parser.add_argument("--current-path", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    reference, champion = load_reference(settings)
    current = load_current(settings, args.source, args.current_path)
    if len(current) == 0:
        raise SystemExit("No production data yet - send requests to the API first.")

    html = settings.reports_dir / "drift_report.html"
    drift = run_drift_report(reference, current, settings, html_path=html)
    decision = decide_retrain(drift, None, settings)
    summary = {
        "champion_version": str(champion.version) if champion else None,
        "drift": asdict(drift),
        "retrain_recommended": decision.retrain,
        "reasons": decision.reasons,
        "report": str(html),
    }
    (settings.reports_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "drift_check",
        dataset_drift=drift.dataset_drift,
        drift_share=drift.drift_share,
        drifted=drift.drifted_columns,
        n_current=drift.n_current,
        report=str(html),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "drift"}, indent=2))


if __name__ == "__main__":
    main()
