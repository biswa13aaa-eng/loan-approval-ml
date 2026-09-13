"""
End-to-end model training, cross-validation, hyperparameter tuning,
evaluation, and visualization pipeline for Loan Approval ML.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)
from src.data_loader import load_raw_data, split_data
from src.preprocessing import create_preprocessor

FIGURES_DIR = BASE_DIR / "reports" / "figures"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    }


def plot_confusion_matrices(models_results, y_test):
    """
    Generate and save side-by-side confusion matrix heatmaps.
    """
    fig, axes = plt.subplots(1, len(models_results), figsize=(5 * len(models_results), 4))
    if len(models_results) == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(models_results.items()):
        cm = data["confusion_matrix"]
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Rejected (N)", "Approved (Y)"],
            yticklabels=["Rejected (N)", "Approved (Y)"],
        )
        acc_val = data['metrics'].get('Accuracy', data['metrics'].get('accuracy', 0.0))
        f1_val = data['metrics'].get('F1', data['metrics'].get('f1', 0.0))
        ax.set_title(f"{name.replace('_', ' ')}\nAcc: {acc_val:.3f} | F1: {f1_val:.3f}", pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrices on Holdout Test Set (N = 123)", y=1.05, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "07_confusion_matrices.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Confusion matrices plot -> {fig_path.name}")


def plot_roc_curves(models_results, y_test):
    """
    Generate and save overlaid ROC curves for all models.
    """
    plt.figure(figsize=(8, 6))
    colors = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD"]

    for idx, (name, data) in enumerate(models_results.items()):
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val = auc(fpr, tpr)
        clean_name = name.replace("_", " ")
        plt.plot(fpr, tpr, label=f"{clean_name} (AUC = {roc_val:.3f})", color=colors[idx % len(colors)], linewidth=2.2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", pad=15, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "08_roc_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] ROC curves plot -> {fig_path.name}")


def extract_and_plot_feature_importance(best_pipeline, preprocessor, X_train):
    """
    Extract transformed feature names, calculate feature importance for Random Forest,
    and plot top features.
    """
    # Fit preprocessor on X_train to get feature names
    preprocessor.fit(X_train)
    col_transformer = preprocessor.named_steps["column_transformer"]
    try:
        raw_names = col_transformer.get_feature_names_out().tolist()
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in raw_names]
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(25)]

    rf_model = None
    if hasattr(best_pipeline.named_steps["classifier"], "feature_importances_"):
        rf_model = best_pipeline.named_steps["classifier"]
    else:
        rf_path = MODELS_DIR / "random_forest.joblib"
        if rf_path.exists():
            rf_pipe = joblib.load(rf_path)
            rf_model = rf_pipe.named_steps["classifier"]

    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        min_len = min(len(feature_names), len(importances))
        fi_df = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Importance": importances[:min_len],
        }).sort_values("Importance", ascending=False)

        # Plot top 10 features
        top_fi = fi_df.head(10)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_fi["Feature"][::-1], top_fi["Importance"][::-1], color="#27AE60", edgecolor="black")
        for bar in bars:
            w = bar.get_width()
            plt.annotate(f"{w:.3f}", xy=(w + 0.005, bar.get_y() + bar.get_height() / 2),
                         va="center", fontsize=10, fontweight="bold")

        plt.xlabel("Relative Importance (Mean Decrease in Impurity)")
        plt.title("Top 10 Feature Importances (Random Forest)", pad=15, fontweight="bold")
        plt.xlim(0, max(top_fi["Importance"]) * 1.15)
        plt.tight_layout()
        fig_path = FIGURES_DIR / "09_feature_importance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Saved] Feature importance plot -> {fig_path.name}")

        # Save feature importance to results
        fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)


def run_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Run GridSearchCV on Random Forest and compare Before vs After tuning.
    """
    print("\n[HYPERPARAMETER TUNING] Running GridSearchCV on Random Forest...")
    preprocessor = create_preprocessor()

    base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    base_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", base_rf)])
    base_pipe.fit(X_train, y_train)
    y_pred_base = base_pipe.predict(X_test)
    y_prob_base = base_pipe.predict_proba(X_test)[:, 1]

    metrics_before = {
        "accuracy": float(accuracy_score(y_test, y_pred_base)),
        "precision": float(precision_score(y_test, y_pred_base, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_base, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_base, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_base)),
    }

    param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": ["balanced", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_tuned_pipe = grid_search.best_estimator_
    y_pred_tuned = best_tuned_pipe.predict(X_test)
    y_prob_tuned = best_tuned_pipe.predict_proba(X_test)[:, 1]

    metrics_after = {
        "accuracy": float(accuracy_score(y_test, y_pred_tuned)),
        "precision": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_tuned, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_tuned)),
    }

    tuning_summary = {
        "best_params": {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()},
        "best_cv_f1": float(grid_search.best_score_),
        "before_tuning": metrics_before,
        "after_tuning": metrics_after,
    }

    with open(RESULTS_DIR / "tuning_comparison.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print("  -> Best Hyperparameters:", tuning_summary["best_params"])
    print(f"  -> Before Tuning F1: {metrics_before['f1']:.4f} | After Tuning F1: {metrics_after['f1']:.4f}")
    print(f"  -> Before Tuning Acc: {metrics_before['accuracy']:.4f} | After Tuning Acc: {metrics_after['accuracy']:.4f}")

    return tuning_summary, best_tuned_pipe


def train_and_evaluate_all():
    """
    Main training and evaluation pipeline.
    """
    print("=" * 60)
    print("STEP 1: Loading and Splitting Dataset")
    print("=" * 60)
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Total samples : {len(df)}")
    print(f"  Train samples : {len(X_train)} (80%)")
    print(f"  Test samples  : {len(X_test)} (20%)")
    print(f"  Train class distribution: Approved={sum(y_train==1)}, Rejected={sum(y_train==0)}")
    print(f"  Test class distribution : Approved={sum(y_test==1)}, Rejected={sum(y_test==0)}")

    candidate_models = get_candidate_models()
    preprocessor = create_preprocessor()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    models_results = {}
    cv_table = []
    test_table = []

    print("\n" + "=" * 60)
    print("STEP 2: 5-Fold Stratified Cross-Validation & Model Training")
    print("=" * 60)

    for name, model in candidate_models.items():
        print(f"\nEvaluating: {name}...")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        # Cross validation
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_metrics = {
            "Model": name,
            "CV_Accuracy": float(np.mean(cv_scores["test_accuracy"])),
            "CV_Precision": float(np.mean(cv_scores["test_precision"])),
            "CV_Recall": float(np.mean(cv_scores["test_recall"])),
            "CV_F1": float(np.mean(cv_scores["test_f1"])),
            "CV_ROC_AUC": float(np.mean(cv_scores["test_roc_auc"])),
        }
        cv_table.append(cv_metrics)

        # Train on full train set and evaluate on test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        test_row = {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc,
        }
        test_table.append(test_row)

        models_results[name] = {
            "pipeline": pipe,
            "metrics": test_row,
            "confusion_matrix": cm,
            "y_prob": y_prob,
        }

        # Save individual model
        model_file = MODELS_DIR / f"{name.lower()}.joblib"
        joblib.dump(pipe, model_file)
        print(f"  Saved model -> {model_file.name}")
        print(f"  Test Metrics: Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ROC-AUC={roc:.4f}")

    # -------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Hyperparameter Tuning (Random Forest)")
    print("=" * 60)
    tuning_summary, tuned_rf_pipe = run_hyperparameter_tuning(X_train, y_train, X_test, y_test)
    joblib.dump(tuned_rf_pipe, MODELS_DIR / "random_forest_tuned.joblib")

    # -------------------------------------------------------------
    # Model Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Holdout Test Set Model Comparison Table")
    print("=" * 60)
    comparison_df = pd.DataFrame(test_table)
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(test_table, f, indent=2)

    cv_df = pd.DataFrame(cv_table)
    cv_df.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)

    # -------------------------------------------------------------
    # Select Best Model
    # -------------------------------------------------------------
    # Rank models by ROC-AUC and F1
    sorted_models = sorted(test_table, key=lambda x: (x["ROC-AUC"], x["F1"]), reverse=True)
    best_model_name = sorted_models[0]["Model"]
    best_pipeline = models_results[best_model_name]["pipeline"]

    print(f"\n>>> Best Overall Performing Model: {best_model_name} (ROC-AUC: {sorted_models[0]['ROC-AUC']:.4f})")
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"  [OK] Production pipeline saved to: {BEST_MODEL_PATH}")

    # -------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Generating Evaluation Visualizations")
    print("=" * 60)
    plot_confusion_matrices(models_results, y_test)
    plot_roc_curves(models_results, y_test)
    extract_and_plot_feature_importance(best_pipeline, preprocessor, X_train)

    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "test_comparison": test_table,
        "cv_comparison": cv_table,
        "tuning_summary": tuning_summary,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Metadata saved to: {MODEL_METADATA_PATH}")
    print("\n[SUCCESS] Model training and evaluation completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_all()
