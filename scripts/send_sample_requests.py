"""Replay a production batch through the API, then send the ground truth.

1. ``POST /predict/batch`` in chunks (each prediction is logged by the API),
2. ``POST /feedback`` with the real diagnoses for a fraction of the requests
   (simulating labels that arrive after follow-up), which enables performance
   monitoring and retraining on served traffic.

Usage::

    python -m scripts.send_sample_requests --batch data/production/batch_01.parquet
    python -m scripts.send_sample_requests --batch ... --api-url http://localhost:8000 --no-feedback
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.columns import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET
from src.config import get_settings
from src.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a dataset row to an API payload (NaN -> null, codes -> int)."""
    payload: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        value = row.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
            payload[col] = None
        elif col in CATEGORICAL_FEATURES or col in ("age", "ca"):
            payload[col] = int(value)
        else:
            payload[col] = float(value)
    return payload


def wait_for_model(session: requests.Session, api_url: str, timeout: float = 90.0) -> str:
    """Block until the API reports a loaded model; return its version."""
    deadline = time.monotonic() + timeout
    last = "unreachable"
    while time.monotonic() < deadline:
        try:
            health = session.get(f"{api_url}/health", timeout=5).json()
            if health.get("model_loaded"):
                return str(health["model_version"])
            last = "no model loaded (train one with `make train`)"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2)
    raise SystemExit(f"API at {api_url} not ready: {last}")


def send_batch(
    frame: pd.DataFrame,
    api_url: str,
    chunk_size: int = 50,
    feedback: bool = True,
    label_fraction: float = 1.0,
    seed: int = 42,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Send every row for prediction and (optionally) its label as feedback."""
    session = session or requests.Session()
    model_version = wait_for_model(session, api_url)
    request_ids: list[str] = []
    predictions: list[int] = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        body = {"instances": [row_to_payload(row) for _, row in chunk.iterrows()]}
        response = session.post(f"{api_url}/predict/batch", json=body, timeout=60)
        response.raise_for_status()
        for item in response.json()["predictions"]:
            request_ids.append(item["request_id"])
            predictions.append(item["prediction"])

    summary: dict[str, Any] = {
        "model_version": model_version,
        "sent": len(request_ids),
        "predicted_positive_rate": round(float(np.mean(predictions)), 3) if predictions else None,
        "labels_sent": 0,
    }
    if feedback and TARGET in frame.columns:
        rng = np.random.default_rng(seed)
        labels = frame[TARGET].to_numpy()
        chosen = [
            i
            for i in range(len(request_ids))
            if not pd.isna(labels[i]) and rng.random() < label_fraction
        ]
        items = [{"request_id": request_ids[i], "actual": int(labels[i])} for i in chosen]
        for start in range(0, len(items), 500):
            response = session.post(
                f"{api_url}/feedback", json={"items": items[start : start + 500]}, timeout=60
            )
            response.raise_for_status()
        summary["labels_sent"] = len(items)
        if chosen:
            correct = sum(predictions[i] == int(labels[i]) for i in chosen)
            summary["served_accuracy"] = round(correct / len(chosen), 3)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True, help="parquet or csv file")
    parser.add_argument(
        "--api-url", default=os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    path = settings.resolve(args.batch)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    summary = send_batch(
        frame,
        args.api_url.rstrip("/"),
        chunk_size=args.chunk_size,
        feedback=not args.no_feedback,
        label_fraction=args.label_fraction,
        seed=settings.project.random_seed,
    )
    logger.info("replayed_batch", batch=str(args.batch), **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
