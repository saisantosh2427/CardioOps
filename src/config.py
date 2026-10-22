"""Typed configuration loaded from ``config.yaml`` with environment overrides.

Precedence (highest first): explicit init kwargs, ``PIPELINE_*`` environment
variables, a ``.env`` file in the repository root, then ``config.yaml``.
Nested keys use a double underscore, e.g. ``PIPELINE_MLFLOW__TRACKING_URI``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_ENV = "PIPELINE_CONFIG_FILE"


def config_file_path() -> Path:
    """Return the YAML config path (``$PIPELINE_CONFIG_FILE`` or ``<root>/config.yaml``)."""
    return Path(os.environ.get(CONFIG_FILE_ENV, PROJECT_ROOT / "config.yaml"))


class ProjectConfig(BaseModel):
    name: str = "heart-disease-risk"
    random_seed: int = 42
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    production_dir: Path = Path("data/production")
    reports_dir: Path = Path("reports")


class DataConfig(BaseModel):
    uci_base_url: str
    uci_files: dict[str, str]
    openml_fallback_name: str = "heart-statlog"
    openml_fallback_version: int = 1
    download_timeout_seconds: int = 30
    reference_fraction: float = Field(0.6, gt=0, lt=1)
    n_production_batches: int = Field(4, ge=1)


class CandidateConfig(BaseModel):
    enabled: bool = True
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    test_size: float = Field(0.2, gt=0, lt=1)
    cv_folds: int = Field(5, ge=2)
    search_scoring: str = "roc_auc"
    decision_threshold: float = Field(0.5, gt=0, lt=1)
    n_jobs: int = -1
    candidates: dict[str, CandidateConfig]


class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "heart-disease-risk"
    registered_model_name: str = "heart-disease-classifier"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    previous_champion_alias: str = "previous-champion"


class MonitoringConfig(BaseModel):
    database_url: str = "sqlite:///data/monitoring.db"
    window_size: int = Field(200, ge=1)
    min_current_samples: int = Field(30, ge=1)
    drift_share_threshold: float = Field(0.3, gt=0, le=1)
    stattest_threshold: float = Field(0.05, gt=0, lt=1)
    performance_metric: str = "roc_auc"
    min_labeled_samples: int = Field(30, ge=1)
    max_metric_drop: float = Field(0.05, ge=0)
    report_cache_seconds: int = Field(60, ge=0)
    mlflow_ui_url: str = "http://localhost:5000"
    api_url: str = "http://localhost:8000"


class RetrainConfig(BaseModel):
    holdout_fraction: float = Field(0.4, gt=0, lt=1)
    promotion_metric: str = "roc_auc"
    tiebreak_metric: str = "f1"
    min_improvement: float = 0.0
    min_labeled_samples: int = Field(50, ge=1)
    production_weight: Literal["balanced", "none"] | float = "balanced"


class ApiConfig(BaseModel):
    model_uri: str = ""
    refresh_interval_seconds: int = Field(30, ge=0)
    refresh_on_request: bool = False
    log_predictions: bool = True
    max_batch_size: int = Field(1000, ge=1)


class Settings(BaseSettings):
    """Root settings object; see ``config.yaml`` for documentation of each field."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    data: DataConfig
    training: TrainingConfig
    mlflow: MlflowConfig = MlflowConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    retrain: RetrainConfig = RetrainConfig()
    api: ApiConfig = ApiConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_file_path())
        return init_settings, env_settings, dotenv_settings, yaml_source

    def resolve(self, path: Path | str) -> Path:
        """Resolve a (possibly relative) path against the repository root."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.paths.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.paths.processed_dir)

    @property
    def production_dir(self) -> Path:
        return self.resolve(self.paths.production_dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports_dir)

    @property
    def reference_path(self) -> Path:
        return self.processed_dir / "reference.parquet"

    @property
    def champion_model_uri(self) -> str:
        """URI the API serves: an explicit override or the registry champion alias."""
        if self.api.model_uri:
            return self.api.model_uri
        return f"models:/{self.mlflow.registered_model_name}@{self.mlflow.champion_alias}"

    @property
    def database_url(self) -> str:
        """Database URL with relative SQLite paths anchored at the repository root."""
        url = self.monitoring.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith(prefix + "/"):
            db_path = Path(url[len(prefix) :])
            if not db_path.is_absolute():
                return f"{prefix}{(PROJECT_ROOT / db_path).as_posix()}"
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
