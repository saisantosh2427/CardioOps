"""Stage 9 - automated retraining with a champion/challenger promotion gate.

1. Load the champion and the data it was trained on (its *reference*).
2. Load the current production window (prediction log, or a file in CI).
3. Data drift (Evidently) + champion performance on labeled traffic.
4. ``decide_retrain``: retrain on drift, on a performance drop, or ``--force``.
5. Training set = the champion's training history + labeled production rows
   (production rows are up-weighted, ``retrain.production_weight``); a
   stratified slice of the labeled production rows is held out and never
   trained on. The new model's drift reference is that production window.
6. Re-run the same candidate search as initial training.
7. Champion and challenger are scored on that *same* hold-out; the challenger
   is promoted to ``champion`` only if it is better (``retrain.promotion_metric``).

Every run (retrain or not) is logged to the ``<experiment>-monitoring`` MLflow
experiment and summarised in ``reports/retrain_summary.json``.

Usage: ``python -m src.retrain [--source db|file] [--current-path PATH] [--force] [--dry-run]``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.columns import FEATURE_COLUMNS, SOURCE, TARGET
from src.config import Settings, get_settings
from src.drift import (
    PerformanceResult,
    RetrainDecision,
    Source,
    baseline_metrics_from_tags,
    check_performance,
    decide_retrain,
    load_current,
    load_reference,
    run_drift_report,
)
from src.evaluate import predict_positive_proba
from src.logging_utils import configure_logging, get_logger
from src.registry import (
    configure_mlflow,
    get_champion,
    load_model,
    load_training_data,
    version_uri,
)
from src.train import register_and_maybe_promote, split_xy, train_candidates

logger = get_logger(__name__)


def split_labeled_current(
    labeled: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (when possible) train / hold-out split of labeled production rows."""
    stratify = labeled[TARGET] if labeled[TARGET].value_counts().min() >= 2 else None
    train, holdout = train_test_split(
        labeled,
        test_size=settings.retrain.holdout_fraction,
        stratify=stratify,
        random_state=settings.project.random_seed,
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def _as_training_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df[[*FEATURE_COLUMNS, TARGET]].copy()
    out[TARGET] = out[TARGET].astype("int64")
    out[SOURCE] = df[SOURCE].astype(str).to_numpy() if SOURCE in df.columns else source
    return out


def production_sample_weights(n_history: int, n_production: int, settings: Settings) -> np.ndarray:
    """Per-row weights: 1 for historical rows, ``retrain.production_weight`` for new ones."""
    policy = settings.retrain.production_weight
    if policy == "none" or n_production == 0:
        weight = 1.0
    elif policy == "balanced":
        weight = max(1.0, n_history / n_production)
    else:
        weight = float(policy)
    return np.concatenate([np.ones(n_history), np.full(n_production, weight)])


def _trigger(decision: RetrainDecision) -> str:
    text = " ".join(decision.reasons)
    if "forced" in text:
        return "forced"
    if "drift" in text and "performance" in text:
        return "drift+performance"
    return "performance" if "performance" in text else "drift"


def run_retrain(
    settings: Settings | None = None,
    source: Source = "db",
    current_path: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one monitoring + (maybe) retraining cycle and return a summary dict."""
    settings = settings or get_settings()
    configure_mlflow(settings)
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    reference, champion = load_reference(settings)
    if champion is None:
        raise SystemExit("No champion model registered yet - run `python -m src.train` first.")
    current = load_current(settings, source, current_path)
    labeled = current[current[TARGET].notna()] if TARGET in current.columns else current.iloc[0:0]

    # --- monitoring: drift + performance of the current champion -------------------
    drift_html = reports / "retrain_drift_report.html"
    drift = run_drift_report(reference, current, settings, html_path=drift_html)
    performance: PerformanceResult | None = None
    if len(labeled):
        champion_model = load_model(version_uri(settings, str(champion.version)))
        performance = check_performance(
            labeled[TARGET].astype(int),
            predict_positive_proba(champion_model, labeled[FEATURE_COLUMNS]),
            baseline_metrics_from_tags(dict(champion.tags or {})),
            settings,
        )
    decision = decide_retrain(drift, performance, settings, force=force)

    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source if source == "db" else str(current_path),
        "champion_version_before": str(champion.version),
        "champion_version_after": str(champion.version),
        "n_current": len(current),
        "n_labeled": len(labeled),
        "decision": asdict(decision),
        "drift": {
            "dataset_drift": drift.dataset_drift,
            "drift_share": drift.drift_share,
            "drifted_columns": drift.drifted_columns,
        },
        "performance": asdict(performance) if performance else None,
        "action": "none",
        "reports": {"drift": str(drift_html)},
    }
    logger.info("retrain_decision", retrain=decision.retrain, reasons=decision.reasons)

    if decision.retrain and len(labeled) < settings.retrain.min_labeled_samples:
        summary["action"] = "skipped_insufficient_labels"
        summary["decision"]["reasons"].append(
            f"only {len(labeled)} labeled rows (< {settings.retrain.min_labeled_samples}); "
            "collect more ground truth via /feedback before retraining"
        )
    elif decision.retrain and dry_run:
        summary["action"] = "dry_run"
    elif decision.retrain:
        history = load_training_data(champion.run_id)
        if history is None:  # models trained before training_data/ existed
            history = reference
        summary.update(_retrain_and_compare(history, labeled, decision, settings))

    _log_monitoring_run(settings, summary, drift_html)
    (reports / "retrain_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _retrain_and_compare(
    history: pd.DataFrame,
    labeled: pd.DataFrame,
    decision: RetrainDecision,
    settings: Settings,
) -> dict[str, Any]:
    """Train a challenger on the champion's history + new labeled data; gate its promotion."""
    current_train, holdout = split_labeled_current(labeled, settings)
    train_df = pd.concat(
        [
            _as_training_frame(history, "history"),
            _as_training_frame(current_train, "production"),
        ],
        ignore_index=True,
    )
    holdout_df = _as_training_frame(holdout, "production")
    weights = production_sample_weights(len(history), len(current_train), settings)
    trigger = _trigger(decision)
    # Training uses history + new data (volume), but the drift *reference* of the
    # new model is the production window it was adapted to. Otherwise the same
    # traffic would keep looking "drifted" against a mostly-historical reference
    # and retraining would be recommended forever.
    adapted_window = _as_training_frame(labeled, "production")
    outcome = train_candidates(
        train_df,
        holdout_df,
        settings,
        run_name=f"retrain-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        reference_df=adapted_window,
        tags={"trigger": trigger, "retrain_reasons": " | ".join(decision.reasons)[:5000]},
        sample_weight=weights,
    )
    X_hold, y_hold = split_xy(holdout_df)
    version, comparison = register_and_maybe_promote(
        outcome.best, X_hold, y_hold, settings, trigger=trigger
    )
    champion = get_champion(configure_mlflow(settings), settings)
    return {
        "action": "retrained_promoted" if comparison.promote else "retrained_rejected",
        "challenger_version": version,
        "champion_version_after": str(champion.version) if champion else None,
        "training_rows": {"history": len(history), "production": len(current_train)},
        "production_row_weight": float(weights[-1]) if len(current_train) else None,
        "holdout_rows": len(holdout_df),
        "comparison": asdict(comparison),
        "best_candidate": outcome.best.name,
    }


def _log_monitoring_run(settings: Settings, summary: dict[str, Any], drift_html: Path) -> None:
    """Record this cycle (metrics, decision, report) in the monitoring experiment."""
    mlflow.set_experiment(f"{settings.mlflow.experiment_name}-monitoring")
    with mlflow.start_run(run_name=f"monitor-{datetime.now(UTC):%Y%m%d-%H%M%S}"):
        mlflow.set_tags(
            {
                "action": summary["action"],
                "champion_version_before": summary["champion_version_before"],
                "champion_version_after": summary["champion_version_after"] or "",
                "reasons": " | ".join(summary["decision"]["reasons"])[:5000],
            }
        )
        metrics = {
            "drift_share": summary["drift"]["drift_share"],
            "dataset_drift": float(summary["drift"]["dataset_drift"]),
            "n_current": summary["n_current"],
            "n_labeled": summary["n_labeled"],
            "retrain": float(summary["decision"]["retrain"]),
        }
        perf = summary.get("performance") or {}
        for name, value in (perf.get("current") or {}).items():
            metrics[f"current_{name}"] = value
        comparison = summary.get("comparison") or {}
        if comparison:
            metrics["challenger_metric"] = comparison["challenger_value"]
            metrics["champion_metric"] = comparison["champion_value"]
            metrics["promoted"] = float(comparison["promote"])
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.log_artifact(str(drift_html), artifact_path="reports")
        mlflow.log_dict(summary, "retrain_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift check + retraining with promotion gate.")
    parser.add_argument("--source", choices=["db", "file"], default="db")
    parser.add_argument("--current-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="retrain even without drift")
    parser.add_argument("--dry-run", action="store_true", help="decide, but do not retrain")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    current_path = settings.resolve(args.current_path) if args.current_path else None
    summary = run_retrain(settings, args.source, current_path, args.force, args.dry_run)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
