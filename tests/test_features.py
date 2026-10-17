"""Feature engineering and the serialized model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import skops.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.columns import ENGINEERED_FEATURES, FEATURE_COLUMNS, TARGET
from src.features import (
    SKOPS_TRUSTED_TYPES,
    HeartFeatureEngineer,
    add_engineered_features,
    build_model_pipeline,
)


def test_engineered_features_values() -> None:
    df = pd.DataFrame(
        [
            {"age": 60.0, "thalach": 160.0, "trestbps": 150.0, "chol": 250.0, "oldpeak": 2.5},
            {"age": 40.0, "thalach": 90.0, "trestbps": 120.0, "chol": 200.0, "oldpeak": 0.0},
        ]
    )
    out = add_engineered_features(df)
    assert out["max_hr_pct"].tolist() == pytest.approx([1.0, 0.5])
    assert out["hypertension"].tolist() == [1.0, 0.0]
    assert out["high_cholesterol"].tolist() == [1.0, 0.0]
    assert out["st_depression_severe"].tolist() == [1.0, 0.0]


def test_engineered_flags_propagate_missing_values() -> None:
    df = pd.DataFrame([{"age": 50.0, "thalach": np.nan, "trestbps": np.nan, "chol": np.nan,
                        "oldpeak": np.nan}])  # fmt: skip
    out = add_engineered_features(df)
    assert out[ENGINEERED_FEATURES].isna().all(axis=None)


def test_feature_engineer_output_columns(synthetic_df: pd.DataFrame) -> None:
    transformer = HeartFeatureEngineer().fit(synthetic_df[FEATURE_COLUMNS])
    out = transformer.transform(synthetic_df[FEATURE_COLUMNS])
    assert list(out.columns) == [*FEATURE_COLUMNS, *ENGINEERED_FEATURES]
    assert list(transformer.get_feature_names_out()) == list(out.columns)


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(max_iter=2000), RandomForestClassifier(n_estimators=20)]
)
def test_pipeline_fits_and_predicts_probabilities(
    synthetic_df: pd.DataFrame, estimator: object
) -> None:
    pipeline = build_model_pipeline(estimator).fit(
        synthetic_df[FEATURE_COLUMNS], synthetic_df[TARGET]
    )
    proba = pipeline.predict_proba(synthetic_df[FEATURE_COLUMNS])[:, 1]
    assert proba.shape == (len(synthetic_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_learns_signal(synthetic_df: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score

    train, test = synthetic_df.iloc[:300], synthetic_df.iloc[300:]
    pipeline = build_model_pipeline(LogisticRegression(max_iter=2000)).fit(
        train[FEATURE_COLUMNS], train[TARGET]
    )
    auc = roc_auc_score(test[TARGET], pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1])
    assert auc > 0.75


def test_serving_payloads_match_training_representation(synthetic_df: pd.DataFrame) -> None:
    """API payloads (ints, None, missing keys) must score exactly like training floats/NaN."""
    pipeline = build_model_pipeline(LogisticRegression(max_iter=2000)).fit(
        synthetic_df[FEATURE_COLUMNS], synthetic_df[TARGET]
    )
    row = synthetic_df[FEATURE_COLUMNS].iloc[[0]].copy()
    row.loc[row.index[0], ["chol", "ca"]] = np.nan
    as_floats = pipeline.predict_proba(row)[:, 1]

    payload = {k: (None if pd.isna(v) else int(v) if float(v).is_integer() else float(v))
               for k, v in row.iloc[0].items()}  # fmt: skip
    payload.pop("ca")  # a missing key is equivalent to null
    as_json_like = pipeline.predict_proba(pd.DataFrame([payload], columns=FEATURE_COLUMNS))[:, 1]
    np.testing.assert_allclose(as_floats, as_json_like)


def test_pipeline_handles_unseen_category_and_all_missing_optionals(
    synthetic_df: pd.DataFrame,
) -> None:
    pipeline = build_model_pipeline(LogisticRegression(max_iter=2000)).fit(
        synthetic_df[FEATURE_COLUMNS], synthetic_df[TARGET]
    )
    sparse = pd.DataFrame([{"age": 55, "sex": 1, "cp": 4}], columns=FEATURE_COLUMNS)
    assert 0 <= pipeline.predict_proba(sparse)[0, 1] <= 1


def test_skops_roundtrip_with_allow_list(synthetic_df: pd.DataFrame) -> None:
    pipeline = build_model_pipeline(RandomForestClassifier(n_estimators=10, random_state=0)).fit(
        synthetic_df[FEATURE_COLUMNS], synthetic_df[TARGET]
    )
    blob = skops.io.dumps(pipeline)
    assert set(skops.io.get_untrusted_types(data=blob)) <= set(SKOPS_TRUSTED_TYPES)
    restored = skops.io.loads(blob, trusted=SKOPS_TRUSTED_TYPES)
    X = synthetic_df[FEATURE_COLUMNS]
    np.testing.assert_allclose(restored.predict_proba(X), pipeline.predict_proba(X))
