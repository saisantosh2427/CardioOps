"""Stages 4-6 - training, experiment tracking and registration.

Each enabled candidate is tuned with a stratified K-fold grid search inside its
own nested MLflow run (params, CV + hold-out metrics, confusion matrix, ROC
curve, CV table, model). The candidate with the best cross-validated score is
registered; it becomes ``champion`` if there is none yet or if it beats the
current champion on the same hold-out set.

Usage: ``python -m src.train``
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy
import sklearn
import skops
import skops.io
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.columns import FEATURE_COLUMNS, TARGET
from src.config import Settings, get_settings
from src.evaluate import (
    ComparisonResult,
    compare_models,
    confusion_matrix_dict,
    evaluate_model,
    predict_positive_proba,
    save_confusion_matrix,
    save_roc_curve,
)
from src.features import SKOPS_TRUSTED_TYPES, build_model_pipeline
from src.logging_utils import configure_logging, get_logger
from src.registry import (
    configure_mlflow,
    get_champion,
    load_model,
    log_model_data,
    mark_challenger,
    promote_to_champion,
    register_model,
    version_uri,
)

logger = get_logger(__name__)

ESTIMATOR_FACTORIES: dict[str, Callable[[int], ClassifierMixin]] = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=5000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
}


@dataclass
class CandidateResult:
    """One tuned candidate and where it was logged."""

    name: str
    model: Pipeline
    best_params: dict[str, Any]
    cv_mean: float
    cv_std: float
    test_metrics: dict[str, float]
    run_id: str
    model_uri: str


@dataclass
class TrainingOutcome:
    """All candidates of a training run plus the selected best one."""

    parent_run_id: str
    candidates: list[CandidateResult] = field(default_factory=list)

    @property
    def best(self) -> CandidateResult:
        return max(self.candidates, key=lambda c: c.cv_mean)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a clean frame into model inputs and the binary target."""
    return df[FEATURE_COLUMNS], df[TARGET].astype(int)


def model_pip_requirements() -> list[str]:
    """Explicit runtime requirements recorded with the model (no slow inference)."""
    return [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"scipy=={scipy.__version__}",
        f"skops=={skops.__version__}",
    ]


def fit_candidate(
    name: str,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
    sample_weight: np.ndarray | None = None,
) -> GridSearchCV:
    """Grid-search one candidate with stratified K-fold cross-validation.

    ``sample_weight`` (optional) is routed to the estimator and split per fold.
    """
    if name not in ESTIMATOR_FACTORIES:
        raise ValueError(f"Unknown candidate '{name}'. Known: {sorted(ESTIMATOR_FACTORIES)}")
    seed = settings.project.random_seed
    search = GridSearchCV(
        estimator=build_model_pipeline(ESTIMATOR_FACTORIES[name](seed)),
        param_grid=param_grid or {},
        scoring=settings.training.search_scoring,
        cv=StratifiedKFold(n_splits=settings.training.cv_folds, shuffle=True, random_state=seed),
        n_jobs=settings.training.n_jobs,
        refit=True,
    )
    fit_params = {} if sample_weight is None else {"model__sample_weight": sample_weight}
    search.fit(X, y, **fit_params)
    return search


def log_model_artifact(model: Pipeline, X_sample: pd.DataFrame) -> ModelInfo:
    """Log the fitted pipeline in the skops format with an explicit type allow-list."""
    untrusted = set(skops.io.get_untrusted_types(data=skops.io.dumps(model)))
    unexpected = untrusted - set(SKOPS_TRUSTED_TYPES)
    if unexpected:
        raise RuntimeError(f"Model contains types outside the skops allow-list: {unexpected}")
    return mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="skops",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
        signature=infer_signature(X_sample, model.predict(X_sample)),
        input_example=X_sample.head(3),
        pip_requirements=model_pip_requirements(),
    )


def _log_diagnostics(
    name: str, search: GridSearchCV, y_eval: pd.Series, proba: np.ndarray, threshold: float
) -> None:
    """Log confusion matrix, ROC curve and CV results as run artifacts."""
    y_pred = (proba >= threshold).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        save_confusion_matrix(y_eval, y_pred, tmp_dir / "confusion_matrix.png", name)
        save_roc_curve(y_eval, proba, tmp_dir / "roc_curve.png", name)
        cv_cols = [
            c for c in search.cv_results_ if c.startswith(("param_", "mean_", "std_", "rank"))
        ]
        pd.DataFrame(search.cv_results_)[cv_cols].to_csv(tmp_dir / "cv_results.csv", index=False)
        (tmp_dir / "confusion_matrix.json").write_text(
            json.dumps(confusion_matrix_dict(np.asarray(y_eval), y_pred), indent=2)
        )
        mlflow.log_artifacts(str(tmp_dir), artifact_path="evaluation")


def train_candidates(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    settings: Settings,
    run_name: str,
    reference_df: pd.DataFrame,
    tags: dict[str, str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> TrainingOutcome:
    """Tune every enabled candidate in nested MLflow runs and select the best by CV score.

    The best run also stores ``reference_df`` (drift reference) and
    ``train_df + eval_df`` (training history used by future retraining).
    """
    X_train, y_train = split_xy(train_df)
    X_eval, y_eval = split_xy(eval_df)
    threshold = settings.training.decision_threshold
    scoring = settings.training.search_scoring
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run(run_name=run_name) as parent:
        outcome = TrainingOutcome(parent_run_id=parent.info.run_id)
        mlflow.set_tags({"pipeline": settings.project.name, **(tags or {})})
        mlflow.log_params(
            {
                "random_seed": settings.project.random_seed,
                "cv_folds": settings.training.cv_folds,
                "search_scoring": scoring,
                "decision_threshold": threshold,
                "n_train": len(train_df),
                "n_eval": len(eval_df),
                "train_positive_rate": round(float(y_train.mean()), 4),
                "sample_weighting": sample_weight is not None,
            }
        )
        train_ds = mlflow.data.from_pandas(train_df, name="train", targets=TARGET)
        eval_ds = mlflow.data.from_pandas(eval_df, name="holdout", targets=TARGET)

        for name, candidate in settings.training.candidates.items():
            if not candidate.enabled:
                continue
            with mlflow.start_run(run_name=name, nested=True) as child:
                log = logger.bind(candidate=name)
                log.info("tuning_candidate", grid=candidate.param_grid)
                search = fit_candidate(
                    name, candidate.param_grid, X_train, y_train, settings, sample_weight
                )
                best_idx = search.best_index_
                cv_mean = float(search.cv_results_["mean_test_score"][best_idx])
                cv_std = float(search.cv_results_["std_test_score"][best_idx])
                proba = predict_positive_proba(search.best_estimator_, X_eval)
                metrics = evaluate_model(search.best_estimator_, X_eval, y_eval, threshold)

                mlflow.set_tag("model_type", name)
                mlflow.log_params({"model_type": name, **search.best_params_})
                mlflow.log_metrics(
                    {
                        f"cv_{scoring}_mean": cv_mean,
                        f"cv_{scoring}_std": cv_std,
                        **{f"test_{k}": v for k, v in metrics.items()},
                    }
                )
                mlflow.log_input(train_ds, context="training")
                mlflow.log_input(eval_ds, context="evaluation")
                _log_diagnostics(name, search, y_eval, proba, threshold)
                model_info = log_model_artifact(search.best_estimator_, X_train)
                outcome.candidates.append(
                    CandidateResult(
                        name=name,
                        model=search.best_estimator_,
                        best_params=search.best_params_,
                        cv_mean=cv_mean,
                        cv_std=cv_std,
                        test_metrics=metrics,
                        run_id=child.info.run_id,
                        model_uri=model_info.model_uri,
                    )
                )
                log.info("candidate_done", cv_mean=round(cv_mean, 4), **_rounded(metrics))

        if not outcome.candidates:
            raise ValueError("No training candidates are enabled in the configuration.")
        best = outcome.best
        mlflow.set_tag("best_candidate", best.name)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in best.test_metrics.items()})

    with mlflow.start_run(run_id=best.run_id):
        log_model_data(reference_df, pd.concat([train_df, eval_df], ignore_index=True))
    logger.info("selected_best_candidate", candidate=best.name, cv_mean=round(best.cv_mean, 4))
    return outcome


def register_and_maybe_promote(
    best: CandidateResult,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    settings: Settings,
    trigger: str,
) -> tuple[str, ComparisonResult]:
    """Register ``best``; promote it if it beats the champion on ``(X_eval, y_eval)``.

    Both models are scored on the exact same rows, so the comparison is fair.
    Returns the new version number and the comparison result.
    """
    client = configure_mlflow(settings)
    champion = get_champion(client, settings)
    champion_metrics = None
    if champion is not None:
        champion_model = load_model(version_uri(settings, champion.version))
        champion_metrics = evaluate_model(
            champion_model, X_eval, y_eval, settings.training.decision_threshold
        )

    challenger_metrics = evaluate_model(
        best.model, X_eval, y_eval, settings.training.decision_threshold
    )
    comparison = compare_models(
        challenger_metrics,
        champion_metrics,
        metric=settings.retrain.promotion_metric,
        tiebreak_metric=settings.retrain.tiebreak_metric,
        min_improvement=settings.retrain.min_improvement,
    )
    model_version = register_model(
        best.model_uri,
        settings,
        tags={
            "model_type": best.name,
            "trigger": trigger,
            "trained_at": datetime.now(UTC).isoformat(),
            **{f"holdout_{k}": f"{v:.4f}" for k, v in challenger_metrics.items()},
        },
    )
    new_version = str(model_version.version)
    if comparison.promote:
        promote_to_champion(client, settings, new_version, comparison.reason)
    else:
        mark_challenger(client, settings, new_version, comparison.reason)

    logger.info(
        "champion_challenger_comparison",
        challenger_version=new_version,
        champion_version=str(champion.version) if champion else None,
        promoted=comparison.promote,
        challenger=_rounded(challenger_metrics),
        champion=_rounded(champion_metrics) if champion_metrics else None,
        reason=comparison.reason,
    )
    return new_version, comparison


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in metrics.items()}


def _data_tags(settings: Settings) -> dict[str, str]:
    """Dataset version tags (content hashes) for lineage."""
    manifest = settings.raw_dir / "manifest.json"
    tags = {}
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        tags["data_source"] = meta.get("source", "unknown")
        tags["raw_data_sha256"] = meta.get("raw_sha256", "unknown")
    return tags


def run_training(settings: Settings | None = None) -> dict[str, Any]:
    """Train on the reference set, register the best model and update aliases."""
    settings = settings or get_settings()
    configure_mlflow(settings)
    if not settings.reference_path.exists():
        raise FileNotFoundError(
            f"{settings.reference_path} not found - run `python -m src.ingest` first."
        )
    reference = pd.read_parquet(settings.reference_path)
    train_df, test_df = train_test_split(
        reference,
        test_size=settings.training.test_size,
        stratify=reference[TARGET],
        random_state=settings.project.random_seed,
    )
    outcome = train_candidates(
        train_df,
        test_df,
        settings,
        run_name=f"training-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        reference_df=reference,
        tags={"trigger": "initial_training", **_data_tags(settings)},
    )
    X_test, y_test = split_xy(test_df)
    version, comparison = register_and_maybe_promote(
        outcome.best, X_test, y_test, settings, trigger="initial_training"
    )
    summary = {
        "parent_run_id": outcome.parent_run_id,
        "best_candidate": outcome.best.name,
        "registered_version": version,
        "promoted": comparison.promote,
        "promotion_reason": comparison.reason,
        "candidates": {
            c.name: {"cv_mean": round(c.cv_mean, 4), **_rounded(c.test_metrics)}
            for c in outcome.candidates
        },
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidates and update the registry.")
    parser.add_argument(
        "--skip-if-champion",
        action="store_true",
        help="do nothing if a champion already exists (bootstrap mode, used by CI)",
    )
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    if args.skip_if_champion:
        champion = get_champion(configure_mlflow(settings), settings)
        if champion is not None:
            logger.info("champion_exists_skipping_training", version=str(champion.version))
            return
    summary = run_training(settings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
