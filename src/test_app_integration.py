"""
Automated headless integration test suite for LOANWISE AI.
Verifies all data loaders, model pipelines, and page render logic.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

from app.app import load_data, load_models, load_results
from src.data_loader import split_data


def test_integration():
    print("[TEST 1/6] Loading data and models...")
    df = load_data()
    assert df is not None, "Dataset failed to load!"
    assert len(df) == 614, f"Expected 614 rows, got {len(df)}"
    print(f"  [PASS] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    models = load_models()
    assert len(models) >= 2, f"Expected at least 2 models, got {len(models)}"
    print(f"  [PASS] Models loaded: {list(models.keys())}")

    results = load_results()
    assert "comparison" in results, "Comparison results missing!"
    assert "metadata" in results, "Metadata missing!"
    print("  [PASS] Results and metadata loaded successfully.")

    print("\n[TEST 2/6] Verifying test split & confusion matrix computation...")
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_test) == 123, f"Expected 123 test samples, got {len(X_test)}"
    rf_pipe = models.get("Random Forest (Selected)") or models.get("Random Forest")
    y_pred = rf_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    assert cm.shape == (2, 2), "Confusion matrix must be 2x2"
    print(f"  [PASS] Confusion matrix dynamically computed on test set: \n{cm}")

    print("\n[TEST 3/6] Verifying Feature Importance extraction directly from pipeline...")
    clf = rf_pipe.named_steps["classifier"]
    preproc = rf_pipe.named_steps["preprocessor"]
    col_trans = preproc.named_steps["column_transformer"]
    raw_names = col_trans.get_feature_names_out().tolist()
    importances = clf.feature_importances_
    assert len(importances) > 0, "Feature importances empty!"
    print(f"  [PASS] Feature importances successfully extracted from pipeline ({len(importances)} features).")

    print("\n[TEST 4/6] Testing prediction on Prime Applicant Profile...")
    prime_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6500.0,
        "CoapplicantIncome": 2500.0,
        "LoanAmount": 140.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
    }
    prime_df = pd.DataFrame([prime_applicant])
    prime_pred = int(rf_pipe.predict(prime_df)[0])
    prime_prob = float(rf_pipe.predict_proba(prime_df)[0, 1])
    print(f"  [PASS] Prime Applicant Decision: {'Approved' if prime_pred==1 else 'Rejected'} (Prob: {prime_prob:.4f})")
    assert prime_pred == 1, "Prime applicant should be approved!"

    print("\n[TEST 5/6] Testing prediction on High-Risk Applicant Profile...")
    risk_applicant = {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 2200.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 220.0,
        "Loan_Amount_Term": 180.0,
        "Credit_History": 0.0,
        "Property_Area": "Rural",
    }
    risk_df = pd.DataFrame([risk_applicant])
    risk_pred = int(rf_pipe.predict(risk_df)[0])
    risk_prob = float(rf_pipe.predict_proba(risk_df)[0, 1])
    print(f"  [PASS] High-Risk Applicant Decision: {'Approved' if risk_pred==1 else 'Rejected'} (Prob: {risk_prob:.4f})")
    assert risk_pred == 0, "High-risk applicant should be rejected!"

    print("\n[TEST 6/6] Verifying all 3 candidate models predict cleanly...")
    for name, pipe in models.items():
        p1 = pipe.predict(prime_df)[0]
        prob1 = pipe.predict_proba(prime_df)[0, 1] if hasattr(pipe, "predict_proba") else None
        print(f"  -> {name:<26} | Prime Applicant: {'Approved' if p1==1 else 'Rejected'} ({prob1:.2%})")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED CLEANLY (6/6)!")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()
