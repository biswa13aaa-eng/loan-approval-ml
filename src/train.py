"""
Model training, cross-validation, and selection pipeline for Loan Approval ML.
"""

import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data_loader import load_raw_data, split_data
from src.preprocessing import create_preprocessor
from src.evaluate import compute_metrics, print_evaluation_summary, confusion_matrix


def get_candidate_models():
    """
    Define candidate classification algorithms.
    """
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def train_and_evaluate():
    """
    Execute full cross-validation, select best model, train final pipeline, and persist artifacts.
    """
    print("[1/5] Loading and splitting data...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    preprocessor = create_preprocessor()
    models = get_candidate_models()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "f1", "accuracy", "precision", "recall"]

    cv_summary = []
    print("\n[2/5] Running 5-Fold Stratified Cross-Validation across candidate models...")

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        mean_roc_auc = float(np.mean(cv_results["test_roc_auc"]))
        mean_f1 = float(np.mean(cv_results["test_f1"]))
        mean_acc = float(np.mean(cv_results["test_accuracy"]))

        print(f"      -> {name:<20} | ROC-AUC: {mean_roc_auc:.4f} | F1: {mean_f1:.4f} | Acc: {mean_acc:.4f}")

        cv_summary.append({
            "name": name,
            "roc_auc": mean_roc_auc,
            "f1": mean_f1,
            "accuracy": mean_acc,
            "estimator": model,
        })

    # Select best model based on ROC-AUC (and F1 as secondary)
    cv_summary.sort(key=lambda x: (x["roc_auc"], x["f1"]), reverse=True)
    best_candidate = cv_summary[0]
    best_name = best_candidate["name"]
    best_estimator = best_candidate["estimator"]

    print(f"\n[3/5] Best candidate selected: {best_name} (CV ROC-AUC: {best_candidate['roc_auc']:.4f})")

    print("\n[4/5] Training final end-to-end pipeline on entire training set...")
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_estimator),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    # Evaluate on untouched holdout test set
    y_pred = final_pipeline.predict(X_test)
    y_prob = (
        final_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(final_pipeline, "predict_proba")
        else None
    )

    test_metrics = compute_metrics(y_test.values, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[Holdout Test Set Performance]")
    print_evaluation_summary(f"FINAL {best_name}", test_metrics, cm)

    # Persist model artifacts
    print("[5/5] Saving model and metadata...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, BEST_MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "cv_performance": {
            s["name"]: {"roc_auc": s["roc_auc"], "f1": s["f1"], "accuracy": s["accuracy"]}
            for s in cv_summary
        },
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "model_file": str(BEST_MODEL_PATH.name),
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"      [OK] Serialized pipeline saved to: {BEST_MODEL_PATH}")
    print(f"      [OK] Training metadata saved to:   {MODEL_METADATA_PATH}")

    return final_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()
