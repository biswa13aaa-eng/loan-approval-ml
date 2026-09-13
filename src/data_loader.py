"""
Data loading and splitting module for Loan Approval ML.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_FILE,
    ID_COL,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_raw_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load raw loan data from CSV.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame containing the dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Clean string whitespace if any
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        # Restore 'nan' string to actual NaN
        df[col] = df[col].replace({"nan": None, "": None})

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified train and test sets.

    Args:
        df: Input DataFrame.
        test_size: Fraction of samples for the test set.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = df.copy()

    # Drop Loan_ID if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    # Convert target to binary: 'Y' -> 1, 'N' -> 0
    y = df[TARGET_COL].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
