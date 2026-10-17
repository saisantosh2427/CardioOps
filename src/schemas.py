"""Pandera data contracts for the raw download and the cleaned dataset."""

from __future__ import annotations

import pandera.pandas as pa

from src.columns import CATEGORICAL_DOMAINS, SOURCE, TARGET, UCI_TARGET_COLUMN


def _categorical_columns() -> dict[str, pa.Column]:
    return {
        col: pa.Column(float, pa.Check.isin(values), nullable=col not in ("sex", "cp"))
        for col, values in CATEGORICAL_DOMAINS.items()
    }


# Raw UCI files: '?' already parsed to NaN; zeros are still present as
# "missing" sentinels for chol / trestbps, so the ranges include 0.
RAW_SCHEMA = pa.DataFrameSchema(
    {
        "age": pa.Column(float, pa.Check.in_range(18, 100)),
        "trestbps": pa.Column(float, pa.Check.in_range(0, 250), nullable=True),
        "chol": pa.Column(float, pa.Check.in_range(0, 700), nullable=True),
        "thalach": pa.Column(float, pa.Check.in_range(50, 250), nullable=True),
        "oldpeak": pa.Column(float, pa.Check.in_range(-5, 10), nullable=True),
        "ca": pa.Column(float, pa.Check.isin([0, 1, 2, 3]), nullable=True),
        **_categorical_columns(),
        UCI_TARGET_COLUMN: pa.Column(int, pa.Check.in_range(0, 4)),
        SOURCE: pa.Column(str),
    },
    strict=True,
    coerce=True,
    name="heart_disease_raw",
)

# After preprocessing: sentinels replaced by NaN, binary target added.
CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "age": pa.Column(float, pa.Check.in_range(18, 100)),
        "trestbps": pa.Column(float, pa.Check.in_range(60, 250), nullable=True),
        "chol": pa.Column(float, pa.Check.in_range(80, 700), nullable=True),
        "thalach": pa.Column(float, pa.Check.in_range(50, 250), nullable=True),
        "oldpeak": pa.Column(float, pa.Check.in_range(-5, 10), nullable=True),
        "ca": pa.Column(float, pa.Check.isin([0, 1, 2, 3]), nullable=True),
        **_categorical_columns(),
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
        SOURCE: pa.Column(str),
    },
    strict=True,
    coerce=True,
    name="heart_disease_clean",
)
