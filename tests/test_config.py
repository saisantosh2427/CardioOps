"""Configuration loading, environment overrides and dependency pins."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, Settings


def test_yaml_defaults_load() -> None:
    settings = Settings(_env_file=None)
    assert settings.mlflow.registered_model_name == "heart-disease-classifier"
    assert set(settings.training.candidates) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert settings.champion_model_uri == "models:/heart-disease-classifier@champion"


def test_env_overrides_nested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_MLFLOW__TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("PIPELINE_MONITORING__WINDOW_SIZE", "42")
    settings = Settings(_env_file=None)
    assert settings.mlflow.tracking_uri == "http://mlflow:5000"
    assert settings.monitoring.window_size == 42
    # untouched siblings still come from config.yaml
    assert settings.mlflow.champion_alias == "champion"


def test_local_model_uri_override() -> None:
    settings = Settings(_env_file=None, api={"model_uri": "/models/exported"})
    assert settings.champion_model_uri == "/models/exported"


def test_relative_paths_and_sqlite_urls_are_anchored_at_repo_root() -> None:
    settings = Settings(_env_file=None, monitoring={"database_url": "sqlite:///data/x.db"})
    assert settings.reference_path == PROJECT_ROOT / "data" / "processed" / "reference.parquet"
    assert settings.database_url == f"sqlite:///{(PROJECT_ROOT / 'data' / 'x.db').as_posix()}"
    pg = "postgresql+psycopg://u:p@db:5432/monitoring"
    assert Settings(_env_file=None, monitoring={"database_url": pg}).database_url == pg


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s#]+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(3)
    return pins


@pytest.mark.parametrize("subset", ["api/requirements.txt", "monitor/requirements.txt"])
def test_image_requirements_match_root_pins(subset: str) -> None:
    """Slim images must use exactly the versions the pipeline was tested with."""
    path = PROJECT_ROOT / subset
    if not path.exists():
        pytest.skip(f"{subset} not present")
    root = _pins(PROJECT_ROOT / "requirements.txt")
    root["mlflow-skinny"] = root["mlflow"]  # client-only build of the same release
    for package, version in _pins(path).items():
        assert package in root, f"{package} is pinned in {subset} but not in requirements.txt"
        assert root[package] == version, f"{package}: {version} != {root[package]}"


def test_every_root_requirement_is_pinned() -> None:
    lines = [
        ln.strip()
        for ln in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert all("==" in ln for ln in lines)


def test_vercel_project_dependencies_match_requirements() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    requirements = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(project["project"]["dependencies"]) == requirements
