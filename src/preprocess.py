"""Data cleaning: type fixes, missing-value sentinels and target encoding.

``clean_features`` is the single cleaning function used both offline (training)
and online (it is the first step of the serialized model pipeline), so serving
can never drift from training. Imputation and categorical encoding are learned
inside the model pipeline (see ``src.features``) so they are fit on training
data only and travel with the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.columns import (
    CATEGORICAL_DOMAINS,
    FEATURE_COLUMNS,
    SOURCE,
    TARGET,
    UCI_TARGET_COLUMN,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

# The UCI files encode some missing measurements as 0, which is physiologically
# impossible for these columns.
ZERO_AS_MISSING: tuple[str, ...] = ("chol", "trestbps")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature columns as float64 with invalid values set to NaN.

    - missing columns are added as NaN (robust to partial payloads),
    - non-numeric values (e.g. '?') are coerced to NaN,
    - zero sentinels for cholesterol / blood pressure become NaN,
    - categorical codes outside the documented domain become NaN.
    """
    out = df.reindex(columns=FEATURE_COLUMNS).copy()
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ZERO_AS_MISSING:
        out.loc[out[col] <= 0, col] = np.nan
    for col, allowed in CATEGORICAL_DOMAINS.items():
        invalid = out[col].notna() & ~out[col].isin(allowed)
        out.loc[invalid, col] = np.nan
    return out


def binarize_target(num: pd.Series) -> pd.Series:
    """Map UCI severity (0-4) to presence of heart disease (0/1)."""
    return (pd.to_numeric(num, errors="coerce") > 0).astype("int64")


def preprocess_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw UCI frame into features + binary ``target`` + ``source``.

    Rows without a label are dropped, as are exact duplicate records.
    """
    labeled = raw.loc[raw[UCI_TARGET_COLUMN].notna()]
    features = clean_features(labeled)
    clean = features.assign(
        **{
            TARGET: binarize_target(labeled[UCI_TARGET_COLUMN]).to_numpy(),
            SOURCE: labeled[SOURCE].astype(str).to_numpy(),
        }
    )
    n_before = len(clean)
    clean = clean.drop_duplicates(subset=[*FEATURE_COLUMNS, TARGET]).reset_index(drop=True)
    missing = {c: int(n) for c, n in clean[FEATURE_COLUMNS].isna().sum().items() if n}
    logger.info(
        "preprocessed_dataset",
        rows_in=len(raw),
        rows_out=len(clean),
        duplicates_dropped=n_before - len(clean),
        unlabeled_dropped=len(raw) - len(labeled),
        missing_values=missing,
        positive_rate=round(float(clean[TARGET].mean()), 3),
    )
    return clean
