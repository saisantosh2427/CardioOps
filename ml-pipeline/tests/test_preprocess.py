"""Preprocessing: cleaning rules, target encoding and data contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from src.columns import FEATURE_COLUMNS, SOURCE, TARGET, UCI_TARGET_COLUMN
from src.preprocess import binarize_target, clean_features, preprocess_dataset
from src.schemas import CLEAN_SCHEMA, RAW_SCHEMA


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        UCI_TARGET_COLUMN: 2,
        SOURCE: "cleveland",
    }
    row.update(overrides)
    return row


def test_clean_features_coerces_types_and_question_marks() -> None:
    df = pd.DataFrame([_raw_row(ca="?", thal="7", oldpeak="1.5")])
    out = clean_features(df)
    assert list(out.columns) == FEATURE_COLUMNS
    assert (out.dtypes == "float64").all()
    assert np.isnan(out.loc[0, "ca"])
    assert out.loc[0, "thal"] == 7.0
    assert out.loc[0, "oldpeak"] == 1.5


def test_clean_features_zero_sentinels_become_missing() -> None:
    out = clean_features(pd.DataFrame([_raw_row(chol=0, trestbps=0)]))
    assert out[["chol", "trestbps"]].isna().all(axis=None)


def test_clean_features_invalid_category_codes_become_missing() -> None:
    out = clean_features(pd.DataFrame([_raw_row(cp=9, thal=5, sex=1)]))
    assert np.isnan(out.loc[0, "cp"])
    assert np.isnan(out.loc[0, "thal"])
    assert out.loc[0, "sex"] == 1.0


def test_clean_features_adds_missing_columns_and_drops_extras() -> None:
    out = clean_features(pd.DataFrame([{"age": 50, "sex": 0, "cp": 2, "unexpected": 1}]))
    assert list(out.columns) == FEATURE_COLUMNS
    assert out[["chol", "thal", "ca"]].isna().all(axis=None)


def test_clean_features_does_not_mutate_input() -> None:
    df = pd.DataFrame([_raw_row(chol=0)])
    clean_features(df)
    assert df.loc[0, "chol"] == 0


def test_binarize_target() -> None:
    assert binarize_target(pd.Series([0, 1, 2, 3, 4])).tolist() == [0, 1, 1, 1, 1]


def test_preprocess_dataset_drops_duplicates_and_unlabeled() -> None:
    raw = pd.DataFrame(
        [_raw_row(), _raw_row(), _raw_row(age=40, **{UCI_TARGET_COLUMN: 0}), _raw_row(age=55)]
    )
    raw.loc[3, UCI_TARGET_COLUMN] = np.nan
    clean = preprocess_dataset(raw)
    assert len(clean) == 2
    assert sorted(clean[TARGET].tolist()) == [0, 1]
    assert set(clean.columns) == {*FEATURE_COLUMNS, TARGET, SOURCE}


def test_schemas_accept_valid_data_and_preprocessing_output() -> None:
    raw = pd.DataFrame([_raw_row(), _raw_row(age=41, chol=0, **{UCI_TARGET_COLUMN: 0})])
    validated = RAW_SCHEMA.validate(raw)
    CLEAN_SCHEMA.validate(preprocess_dataset(validated))


@pytest.mark.parametrize(
    "bad",
    [{"age": 5}, {"cp": 7}, {UCI_TARGET_COLUMN: 9}, {"thalach": 400}],
)
def test_raw_schema_rejects_out_of_contract_values(bad: dict[str, object]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        RAW_SCHEMA.validate(pd.DataFrame([_raw_row(**bad)]), lazy=True)


def test_raw_schema_rejects_unexpected_columns() -> None:
    df = pd.DataFrame([_raw_row()]).assign(extra=1)
    with pytest.raises(pandera.errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)
