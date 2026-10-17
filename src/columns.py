"""Column definitions for the UCI Heart Disease dataset.

Kept dependency-free so the slim API image can import it.
See https://archive.ics.uci.edu/dataset/45/heart+disease for the codebook.
"""

from __future__ import annotations

# Model input features, in UCI order.
FEATURE_COLUMNS: list[str] = [
    "age",  # years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl
    "restecg",  # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise induced angina
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",  # number of major vessels colored by fluoroscopy (0-3)
    "thal",  # 3 normal, 6 fixed defect, 7 reversible defect
]

UCI_TARGET_COLUMN = "num"  # 0 = no disease, 1-4 = increasing severity
UCI_COLUMNS: list[str] = [*FEATURE_COLUMNS, UCI_TARGET_COLUMN]

TARGET = "target"  # binary: 1 if num > 0
SOURCE = "source"  # hospital the record came from (metadata, not a feature)

NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ENGINEERED_FEATURES: list[str] = [
    "max_hr_pct",  # thalach / age-predicted max heart rate (220 - age)
    "hypertension",  # trestbps >= 140
    "high_cholesterol",  # chol >= 240
    "st_depression_severe",  # oldpeak >= 2
]

# Valid codes for categorical features; anything else is treated as missing.
CATEGORICAL_DOMAINS: dict[str, list[float]] = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}
