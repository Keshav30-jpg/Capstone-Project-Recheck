"""Shared constants and the leakage-safe production pipeline.

Academic scripts in Parts 1-3 stay readable and self-contained. This module is
used by the *final* Part 3 pipeline and by Part 4 inference so encoding,
imputation, and validation cannot drift apart.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "customerID"
TARGET_COL = "Churn"
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
CATEGORICAL_FEATURES = [
    "gender",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INTERNET_MAPPING = {
    "fiberoptic": "fiber optic",
    "fibre optic": "fiber optic",
    "fiber optic": "fiber optic",
}

CONTRACT_MAPPING = {
    "1 year": "one year",
    "2 year": "two year",
    "one year": "one year",
    "two year": "two year",
    "month to month": "month-to-month",
    "month-to-m": "month-to-month",
    "monthly": "month-to-month",
    "month-to-month": "month-to-month",
}

ALLOWED_CATEGORIES = {
    "gender": ["female", "male"],
    "PhoneService": ["yes", "no"],
    "InternetService": ["dsl", "fiber optic", "no"],
    "Contract": ["month-to-month", "one year", "two year"],
    "PaperlessBilling": ["yes", "no"],
    "PaymentMethod": [
        "bank transfer",
        "credit card",
        "electronic check",
        "mailed check",
    ],
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gender": {"type": "string"},
        "SeniorCitizen": {"type": "integer", "enum": [0, 1]},
        "tenure": {"type": "number", "minimum": 0},
        "PhoneService": {"type": "string"},
        "InternetService": {"type": "string"},
        "Contract": {"type": "string"},
        "PaperlessBilling": {"type": "string"},
        "PaymentMethod": {"type": "string"},
        "MonthlyCharges": {"type": "number", "minimum": 0},
        "TotalCharges": {"type": "number", "minimum": 0},
    },
    "required": [
        "gender",
        "SeniorCitizen",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ],
    "additionalProperties": False,
}


def build_churn_pipeline(estimator):
    """ColumnTransformer + estimator. Fit this on *raw* train columns only."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
