"""Stage 1 - data ingestion.

Downloads the UCI Heart Disease data (falling back to OpenML), validates it
against a pandera contract, cleans it, and splits it into a *reference* set
(used for training) and several simulated *production* batches that "arrive"
after deployment.

Usage: ``python -m src.ingest [--force]``
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.columns import SOURCE, TARGET, UCI_COLUMNS, UCI_TARGET_COLUMN
from src.config import PROJECT_ROOT, Settings, get_settings
from src.logging_utils import configure_logging, get_logger
from src.preprocess import preprocess_dataset
from src.schemas import CLEAN_SCHEMA, RAW_SCHEMA

logger = get_logger(__name__)

RAW_FILENAME = "heart_disease_raw.csv"
MANIFEST_FILENAME = "manifest.json"

# OpenML "heart-statlog" uses the same codes as UCI with descriptive names.
OPENML_COLUMN_MAP = {
    "age": "age",
    "sex": "sex",
    "chest": "cp",
    "resting_blood_pressure": "trestbps",
    "serum_cholestoral": "chol",
    "fasting_blood_sugar": "fbs",
    "resting_electrocardiographic_results": "restecg",
    "maximum_heart_rate_achieved": "thalach",
    "exercise_induced_angina": "exang",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "number_of_major_vessels": "ca",
    "thal": "thal",
}


@dataclass
class IngestResult:
    """Summary of an ingestion run (also written to the data manifest)."""

    source: str
    raw_path: str
    raw_sha256: str
    n_raw_rows: int
    n_clean_rows: int
    n_reference_rows: int
    production_batches: list[str]
    created_at: str


def _display_path(path: Path) -> str:
    """Repo-relative POSIX path when possible (portable manifests), else absolute."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    """Content hash used to version datasets in MLflow."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uci(settings: Settings) -> pd.DataFrame:
    """Download and concatenate the four processed UCI hospital files."""
    frames = []
    for site, filename in settings.data.uci_files.items():
        url = f"{settings.data.uci_base_url.rstrip('/')}/{filename}"
        response = requests.get(url, timeout=settings.data.download_timeout_seconds)
        response.raise_for_status()
        frame = pd.read_csv(
            io.StringIO(response.text), header=None, names=UCI_COLUMNS, na_values="?"
        )
        frame[SOURCE] = site
        frames.append(frame)
        logger.info("downloaded_uci_file", site=site, rows=len(frame), url=url)
    return pd.concat(frames, ignore_index=True)


def download_openml(settings: Settings) -> pd.DataFrame:
    """Fallback source: OpenML ``heart-statlog`` mapped onto the UCI schema."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        name=settings.data.openml_fallback_name,
        version=settings.data.openml_fallback_version,
        as_frame=True,
    )
    frame = bunch.data.rename(columns=OPENML_COLUMN_MAP)[list(OPENML_COLUMN_MAP.values())]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame[UCI_TARGET_COLUMN] = (bunch.target.astype(str) == "present").astype(int)
    frame[SOURCE] = "statlog"
    logger.info("downloaded_openml", name=settings.data.openml_fallback_name, rows=len(frame))
    return frame


def download_raw(settings: Settings) -> tuple[pd.DataFrame, str]:
    """Download raw data from UCI, falling back to OpenML. Returns (frame, source)."""
    try:
        return download_uci(settings), "uci"
    except (requests.RequestException, ValueError) as exc:
        logger.warning("uci_download_failed_using_openml", error=str(exc))
        return download_openml(settings), "openml"


def split_reference_production(
    clean: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Stratified split into a reference set and ``n_production_batches`` batches."""
    seed = settings.project.random_seed
    reference, production = train_test_split(
        clean,
        train_size=settings.data.reference_fraction,
        stratify=clean[TARGET],
        random_state=seed,
    )
    production = production.sample(frac=1.0, random_state=seed)
    batch_indices = np.array_split(np.arange(len(production)), settings.data.n_production_batches)
    batches = [production.iloc[idx].reset_index(drop=True) for idx in batch_indices]
    return reference.reset_index(drop=True), batches


def run_ingest(settings: Settings | None = None, force: bool = False) -> IngestResult:
    """Run the full ingestion stage and write raw, reference and production files."""
    settings = settings or get_settings()
    raw_dir, processed_dir, production_dir = (
        settings.raw_dir,
        settings.processed_dir,
        settings.production_dir,
    )
    for directory in (raw_dir, processed_dir, production_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / RAW_FILENAME
    manifest_path = raw_dir / MANIFEST_FILENAME
    if raw_path.exists() and manifest_path.exists() and not force:
        source = json.loads(manifest_path.read_text())["source"]
        raw = pd.read_csv(raw_path)
        logger.info("using_cached_raw_data", path=str(raw_path), rows=len(raw))
    else:
        raw, source = download_raw(settings)
        # Fixed line endings keep the SHA-256 identical on every OS (data versioning).
        raw.to_csv(raw_path, index=False, lineterminator="\n")

    raw = RAW_SCHEMA.validate(raw, lazy=True)
    clean = CLEAN_SCHEMA.validate(preprocess_dataset(raw), lazy=True)
    clean.to_parquet(processed_dir / "heart_disease_clean.parquet", index=False)

    reference, batches = split_reference_production(clean, settings)
    reference.to_parquet(settings.reference_path, index=False)
    for stale in production_dir.glob("batch_*.parquet"):
        stale.unlink()
    batch_paths = []
    for i, batch in enumerate(batches, start=1):
        path = production_dir / f"batch_{i:02d}.parquet"
        batch.to_parquet(path, index=False)
        batch_paths.append(_display_path(path))

    result = IngestResult(
        source=source,
        raw_path=_display_path(raw_path),
        raw_sha256=sha256_file(raw_path),
        n_raw_rows=len(raw),
        n_clean_rows=len(clean),
        n_reference_rows=len(reference),
        production_batches=batch_paths,
        created_at=datetime.now(UTC).isoformat(),
    )
    manifest_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info(
        "ingest_complete",
        **{k: v for k, v in asdict(result).items() if k != "production_batches"},
        n_batches=len(batch_paths),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    run_ingest(settings, force=args.force)


if __name__ == "__main__":
    main()
