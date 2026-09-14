"""Project-derived helpers shared by the LOANWISE AI interface."""

from __future__ import annotations

import pandas as pd


def input_columns(df: pd.DataFrame, target: str = "Loan_Status", identifier: str = "Loan_ID") -> list[str]:
    return [column for column in df.columns if column not in {target, identifier}]


def feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = input_columns(df)
    numerical = df[features].select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features if column not in numerical]
    return numerical, categorical


def display_name(value: str) -> str:
    return value.replace("_", " ")


def best_row(comparison: pd.DataFrame, metric: str = "F1") -> pd.Series | None:
    if comparison is None or comparison.empty or metric not in comparison:
        return None
    return comparison.loc[comparison[metric].idxmax()]


def feature_label(feature: str) -> str:
    return feature.replace("num__", "").replace("cat__", "").replace("_", " ")
