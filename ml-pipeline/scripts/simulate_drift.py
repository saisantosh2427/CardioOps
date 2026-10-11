"""Simulate drifted production data: a new hospital site starts sending patients.

The rows are real held-back patients (with their real diagnoses); only the
*recorded measurements* change, which is what happens when a new site joins:

- population shift: patients are ~7 years older, with higher resting blood
  pressure (+12 mm Hg) and cholesterol (+35 mg/dl);
- measurement shift: the site's stress-test equipment reports max heart rate
  ~25 bpm lower and ST depression on a different scale (x2 + 0.5 mm).

Because labels are unchanged, the feature -> label relationship the champion
learned no longer holds (it over-predicts disease), so drift detection,
a performance drop and a genuinely better retrained challenger all follow.

Usage: ``python -m scripts.simulate_drift [--batches 2 3 4] [--intensity 1.0]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

DRIFTED_FILENAME = "drifted_batch.parquet"


def apply_drift(df: pd.DataFrame, intensity: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Return a drifted copy of ``df``; ``intensity`` scales every shift (0 = no drift).

    Values are clipped to the API's accepted ranges so the batch can be replayed
    through ``/predict``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    def noise(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, n) * intensity

    out["age"] = (out["age"] + 7 * intensity + noise(2)).round().clip(18, 100)
    out["trestbps"] = (out["trestbps"] + 12 * intensity + noise(4)).round().clip(60, 250)
    out["chol"] = (out["chol"] + 35 * intensity + noise(10)).round().clip(80, 700)
    out["thalach"] = (out["thalach"] - 25 * intensity + noise(4)).round().clip(50, 250)
    out["oldpeak"] = (out["oldpeak"] * (1 + intensity) + 0.5 * intensity).round(1).clip(-5, 10)
    if "source" in out.columns and intensity > 0:
        out["source"] = "new_site"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="production batch numbers to draw patients from (batch 1 stays 'normal')",
    )
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    paths = [settings.production_dir / f"batch_{i:02d}.parquet" for i in args.batches]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing} - run `python -m src.ingest` first.")

    pool = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    drifted = apply_drift(pool, args.intensity, seed=settings.project.random_seed)
    output = args.output or settings.production_dir / DRIFTED_FILENAME
    drifted.to_parquet(output, index=False)

    deltas = {
        col: round(float(drifted[col].mean() - pool[col].mean()), 2)
        for col in ("age", "trestbps", "chol", "thalach", "oldpeak")
    }
    logger.info(
        "drifted_batch_written",
        path=str(output),
        rows=len(drifted),
        intensity=args.intensity,
        mean_shift=deltas,
    )


if __name__ == "__main__":
    main()
