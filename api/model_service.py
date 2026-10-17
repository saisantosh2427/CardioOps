"""Loads the champion model from the MLflow registry and hot-swaps it on promotion."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.config import Settings
from src.logging_utils import get_logger
from src.registry import configure_mlflow, get_version_by_alias

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """A model plus the registry metadata needed to explain what is being served."""

    model: object
    name: str
    version: str
    model_uri: str
    source: str  # "registry" | "local"
    alias: str | None
    run_id: str | None
    tags: dict[str, str] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def model_type(self) -> str | None:
        return self.tags.get("model_type")


class ModelUnavailableError(RuntimeError):
    """Raised when a prediction is requested before any model is loaded."""


class ModelService:
    """Thread-safe holder of the currently served model.

    With the default ``models:/<name>@champion`` URI the alias is resolved to
    a concrete version first, so the service always knows exactly which version
    it serves and can detect when the alias moves.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._current: LoadedModel | None = None
        self.last_error: str | None = None

    @property
    def current(self) -> LoadedModel | None:
        with self._lock:
            return self._current

    @property
    def is_registry_alias(self) -> bool:
        return not self.settings.api.model_uri

    def _client(self) -> MlflowClient:
        return configure_mlflow(self.settings)

    def _load_from_registry(self) -> LoadedModel | None:
        cfg = self.settings.mlflow
        client = self._client()
        mv = get_version_by_alias(client, cfg.registered_model_name, cfg.champion_alias)
        if mv is None:
            self.last_error = f"No model version has alias '{cfg.champion_alias}' yet"
            return None
        current = self.current
        if current is not None and current.version == str(mv.version):
            return current
        uri = f"models:/{cfg.registered_model_name}/{mv.version}"
        model = mlflow.sklearn.load_model(uri)
        return LoadedModel(
            model=model,
            name=cfg.registered_model_name,
            version=str(mv.version),
            model_uri=uri,
            source="registry",
            alias=cfg.champion_alias,
            run_id=mv.run_id,
            tags=dict(mv.tags or {}),
        )

    def _load_from_path(self) -> LoadedModel:
        uri = self.settings.api.model_uri
        current = self.current
        if current is not None and current.model_uri == uri:
            return current
        model = mlflow.sklearn.load_model(uri)
        info = mlflow.models.get_model_info(uri)
        return LoadedModel(
            model=model,
            name=self.settings.mlflow.registered_model_name,
            version=str((info.metadata or {}).get("version", "local")),
            model_uri=uri,
            source="local",
            alias=None,
            run_id=info.run_id,
            tags={k: str(v) for k, v in (info.metadata or {}).items()},
        )

    def refresh(self) -> bool:
        """Load the champion if it changed. Returns True if a new model was swapped in.

        Failures (e.g. MLflow temporarily unreachable) are logged and the
        currently served model is kept.
        """
        try:
            loaded = (
                self._load_from_registry() if self.is_registry_alias else self._load_from_path()
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("model_refresh_failed", error=self.last_error)
            return False
        if loaded is None:
            logger.warning("no_champion_available", detail=self.last_error)
            return False
        with self._lock:
            changed = self._current is None or self._current.version != loaded.version
            previous = self._current.version if self._current else None
            self._current = loaded
        if changed:
            self.last_error = None
            logger.info(
                "model_loaded",
                name=loaded.name,
                version=loaded.version,
                previous_version=previous,
                source=loaded.source,
                model_type=loaded.model_type,
            )
        return changed

    def predict_proba(self, features: pd.DataFrame) -> tuple[np.ndarray, LoadedModel]:
        """Positive-class probabilities and the model that produced them."""
        loaded = self.current
        if loaded is None:
            raise ModelUnavailableError(self.last_error or "Model not loaded")
        proba = loaded.model.predict_proba(features)[:, 1]  # type: ignore[attr-defined]
        return proba, loaded
