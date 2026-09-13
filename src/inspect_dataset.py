import json
import pandas as pd
import numpy as np

file_path = r"data\raw\loan_data.csv"
df = pd.read_csv(file_path)

n_rows, n_cols = df.shape
dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

missing = df.isnull().sum()
missing_pct = (missing / n_rows * 100).round(2)
missing_dict = {
    col: {"count": int(missing[col]), "percent": float(missing_pct[col])}
    for col in df.columns if missing[col] > 0
}

duplicates = int(df.duplicated().sum())

target_counts = df["Loan_Status"].value_counts().to_dict()
target_dist = (df["Loan_Status"].value_counts(normalize=True) * 100).round(2).to_dict()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Loan_ID"]

cat_uniques = {col: df[col].dropna().unique().tolist() for col in cat_cols}

report = {
    "rows": n_rows,
    "columns": n_cols,
    "column_names": df.columns.tolist(),
    "dtypes": dtypes,
    "missing_values": missing_dict,
    "duplicate_rows": duplicates,
    "target_variable": "Loan_Status",
    "target_distribution": target_counts,
    "target_percentages": target_dist,
    "numerical_features": num_cols,
    "categorical_features": cat_cols,
    "categorical_unique_values": cat_uniques,
}

print(json.dumps(report, indent=2))
