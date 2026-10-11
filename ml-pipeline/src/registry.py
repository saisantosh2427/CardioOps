"""Stage 6 - MLflow Model Registry helpers (aliases, promotion, loading)."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from src.config import Settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Two data snapshots are stored with every candidate's run:
# - reference_data/: the distribution the model is validated for. Drift is
#   always measured against the *current champion's* reference.
# - training_data/: every labeled row the model was developed on; retraining
#   starts from the champion's training history and adds new production rows.
REFERENCE_ARTIFACT_DIR = "reference_data"
REFERENCE_ARTIFACT_FILE = "reference.parquet"
TRAINING_ARTIFACT_DIR = "training_data"
TRAINING_ARTIFACT_FILE = "training.parquet"


def configure_mlflow(settings: Settings) -> MlflowClient:
    """Point the MLflow fluent API at the configured server and return a client.

    Multipart artifact transfer is disabled: with an S3 backend the server hands
    out presigned URLs for the *internal* MinIO host (``minio:9000``), which
    clients outside the compose network cannot resolve. Our artifacts are small,
    so everything goes through the server's artifact proxy instead.
    """
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow.tracking_uri)


def get_version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    """Return the model version holding ``alias`` or ``None`` if unset/unregistered."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"):
            return None
        raise


def get_champion(client: MlflowClient, settings: Settings) -> ModelVersion | None:
    """Current production model version, if any."""
    return get_version_by_alias(
        client, settings.mlflow.registered_model_name, settings.mlflow.champion_alias
    )


def register_model(
    model_uri: str, settings: Settings, tags: dict[str, str] | None = None
) -> ModelVersion:
    """Register ``model_uri`` as a new version of the configured registered model."""
    version = mlflow.register_model(
        model_uri, settings.mlflow.registered_model_name, tags=tags or {}
    )
    logger.info(
        "registered_model_version",
        name=settings.mlflow.registered_model_name,
        version=version.version,
        run_id=version.run_id,
    )
    return version


def promote_to_champion(
    client: MlflowClient, settings: Settings, version: str, reason: str
) -> str | None:
    """Move the champion alias to ``version``; keep the old one as previous-champion.

    Returns the previous champion version (if there was one) to allow rollback.
    """
    name = settings.mlflow.registered_model_name
    version = str(version)
    previous = get_champion(client, settings)
    previous_version = str(previous.version) if previous is not None else None
    if previous_version is not None and previous_version != version:
        client.set_registered_model_alias(
            name, settings.mlflow.previous_champion_alias, previous_version
        )
    client.set_registered_model_alias(name, settings.mlflow.champion_alias, version)
    client.set_model_version_tag(name, version, "promotion_reason", reason)
    client.set_model_version_tag(name, version, "promoted_at", datetime.now(UTC).isoformat())
    # The challenger alias, if it pointed at this version, is now redundant.
    challenger = get_version_by_alias(client, name, settings.mlflow.challenger_alias)
    if challenger is not None and str(challenger.version) == version:
        client.delete_registered_model_alias(name, settings.mlflow.challenger_alias)
    logger.info(
        "promoted_champion",
        name=name,
        version=version,
        previous_version=previous_version,
        reason=reason,
    )
    return previous_version


def mark_challenger(client: MlflowClient, settings: Settings, version: str, reason: str) -> None:
    """Tag a version that was evaluated but not promoted."""
    name = settings.mlflow.registered_model_name
    client.set_registered_model_alias(name, settings.mlflow.challenger_alias, str(version))
    client.set_model_version_tag(name, str(version), "rejection_reason", reason)
    logger.info("kept_as_challenger", name=name, version=str(version), reason=reason)


def load_model(model_uri: str) -> object:
    """Load a scikit-learn pipeline (``predict_proba`` available) from MLflow."""
    return mlflow.sklearn.load_model(model_uri)


def version_uri(settings: Settings, version: str) -> str:
    return f"models:/{settings.mlflow.registered_model_name}/{version}"


def _log_frame(frame: pd.DataFrame, artifact_dir: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        frame.to_parquet(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=artifact_dir)


def _load_frame(run_id: str, artifact_dir: str, filename: str) -> pd.DataFrame | None:
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_dir}/{filename}"
        )
    except (MlflowException, OSError) as exc:
        logger.warning(
            "data_artifact_missing", run_id=run_id, artifact=artifact_dir, error=str(exc)
        )
        return None
    return pd.read_parquet(local)


def log_model_data(reference: pd.DataFrame, training: pd.DataFrame) -> None:
    """Attach the drift reference and the training history to the active run."""
    _log_frame(reference, REFERENCE_ARTIFACT_DIR, REFERENCE_ARTIFACT_FILE)
    _log_frame(training, TRAINING_ARTIFACT_DIR, TRAINING_ARTIFACT_FILE)


def load_reference_data(run_id: str) -> pd.DataFrame | None:
    """Drift reference logged with ``run_id`` (None if missing)."""
    return _load_frame(run_id, REFERENCE_ARTIFACT_DIR, REFERENCE_ARTIFACT_FILE)


def load_training_data(run_id: str) -> pd.DataFrame | None:
    """Training history logged with ``run_id`` (None if missing)."""
    return _load_frame(run_id, TRAINING_ARTIFACT_DIR, TRAINING_ARTIFACT_FILE)
