"""
Preprocessing and feature engineering module for Loan Approval ML.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERICAL_COLS, CATEGORICAL_COLS


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that generates domain-specific loan features:
    - Total_Income: ApplicantIncome + CoapplicantIncome
    - Loan_Amount_to_Total_Income: LoanAmount / Total_Income
    - Monthly_EMI: (LoanAmount * 1000) / Loan_Amount_Term
    - EMI_to_Income_Ratio: Monthly_EMI / Total_Income
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Handle types
        for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
            if col in X_out.columns:
                X_out[col] = pd.to_numeric(X_out[col], errors="coerce")

        total_income = X_out["ApplicantIncome"].fillna(0) + X_out["CoapplicantIncome"].fillna(0)
        X_out["Total_Income"] = total_income

        # Avoid zero division
        safe_income = total_income.replace(0, np.nan)
        safe_term = X_out["Loan_Amount_Term"].replace(0, np.nan)

        # LoanAmount is in thousands
        X_out["Loan_Amount_to_Total_Income"] = (X_out["LoanAmount"] * 1000) / safe_income
        X_out["Monthly_EMI"] = (X_out["LoanAmount"] * 1000) / safe_term
        X_out["EMI_to_Income_Ratio"] = X_out["Monthly_EMI"] / safe_income

        # Replace any infs with nan for the imputer
        for col in ["Total_Income", "Loan_Amount_to_Total_Income", "Monthly_EMI", "EMI_to_Income_Ratio"]:
            X_out[col] = X_out[col].replace([np.inf, -np.inf], np.nan)

        return X_out


def create_preprocessor() -> Pipeline:
    """
    Construct the full scikit-learn preprocessing pipeline.

    Returns:
        Pipeline: Feature engineering followed by ColumnTransformer with imputation and encoding.
    """
    all_num_cols = NUMERICAL_COLS + [
        "Total_Income",
        "Loan_Amount_to_Total_Income",
        "Monthly_EMI",
        "EMI_to_Income_Ratio",
    ]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_num_cols),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineer", LoanFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    return preprocessor
