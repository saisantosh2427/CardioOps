"""Feature engineering and the full scikit-learn model pipeline.

The serialized artifact is ``Pipeline([features, preprocess, model])`` so the
API applies exactly the same cleaning, feature engineering, imputation,
scaling and encoding that training used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.columns import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from src.preprocess import clean_features

# Models are stored in the skops format, which refuses to load any type that is
# not explicitly trusted. This is the complete allow-list for our candidates;
# training fails loudly if a model ever needs anything else.
SKOPS_TRUSTED_TYPES: list[str] = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "src.features.HeartFeatureEngineer",
]


def _flag(condition: pd.Series, source: pd.Series) -> pd.Series:
    """Boolean flag as float, NaN where the underlying measurement is missing."""
    return condition.astype("float64").where(source.notna())


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append clinically motivated features to a cleaned feature frame."""
    out = df.copy()
    out["max_hr_pct"] = out["thalach"] / (220.0 - out["age"])
    out["hypertension"] = _flag(out["trestbps"] >= 140, out["trestbps"])
    out["high_cholesterol"] = _flag(out["chol"] >= 240, out["chol"])
    out["st_depression_severe"] = _flag(out["oldpeak"] >= 2.0, out["oldpeak"])
    return out


class HeartFeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless transformer: ``clean_features`` followed by ``add_engineered_features``."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HeartFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURE_COLUMNS)
        return add_engineered_features(clean_features(frame))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray([*FEATURE_COLUMNS, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numeric features, imputation + one-hot for categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, [*NUMERIC_FEATURES, *ENGINEERED_FEATURES]),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Wrap an estimator with the shared feature pipeline."""
    return Pipeline(
        [
            ("features", HeartFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
