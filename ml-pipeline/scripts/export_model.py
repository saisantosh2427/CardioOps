"""Export the registry champion for registry-less serving (Hugging Face Spaces).

A Space cannot reach your local MLflow server, so this assembles a
self-contained Docker build context that bakes the champion into the image::

    <output>/
      Dockerfile  README.md   # from hf_space/ (README carries the Space metadata)
      config.yaml src/ api/   # application code
      model/                  # champion as a local MLflow model (skops format)

Usage::

    python -m scripts.export_model --output hf_space/build
    docker build -t heart-api-hf hf_space/build      # optional local check
    # then push the folder to a Hugging Face Docker Space (see README.md)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow.sklearn

from src.config import PROJECT_ROOT, Settings, get_settings
from src.features import SKOPS_TRUSTED_TYPES
from src.logging_utils import configure_logging, get_logger
from src.registry import configure_mlflow, get_champion, load_model, version_uri
from src.train import model_pip_requirements

logger = get_logger(__name__)

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "Dockerfile")


def export_champion(settings: Settings, model_dir: Path) -> dict[str, Any]:
    """Save the current champion as a standalone MLflow model directory."""
    client = configure_mlflow(settings)
    champion = get_champion(client, settings)
    if champion is None:
        raise SystemExit("No champion registered - run `make train` first.")
    model = load_model(version_uri(settings, str(champion.version)))
    metadata = {
        "version": str(champion.version),
        "registered_model_name": settings.mlflow.registered_model_name,
        "source_run_id": champion.run_id,
        "exported_at": datetime.now(UTC).isoformat(),
        **{k: str(v) for k, v in (champion.tags or {}).items()},
    }
    if model_dir.exists():
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(
        model,
        str(model_dir),
        serialization_format="skops",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
        pip_requirements=model_pip_requirements(),
        metadata=metadata,
    )
    logger.info("exported_champion", version=metadata["version"], path=str(model_dir))
    return metadata


def build_space(settings: Settings, output: Path) -> dict[str, Any]:
    """Assemble the full Space build context in ``output``."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "hf_space" / "Dockerfile", output / "Dockerfile")
    shutil.copy(PROJECT_ROOT / "hf_space" / "README.md", output / "README.md")
    shutil.copy(PROJECT_ROOT / "config.yaml", output / "config.yaml")
    shutil.copytree(PROJECT_ROOT / "src", output / "src", ignore=_IGNORE)
    shutil.copytree(PROJECT_ROOT / "api", output / "api", ignore=_IGNORE)
    return export_champion(settings, output / "model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Hugging Face Space from the champion.")
    parser.add_argument("--output", type=Path, default=Path("hf_space/build"))
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    metadata = build_space(settings, settings.resolve(args.output))
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
