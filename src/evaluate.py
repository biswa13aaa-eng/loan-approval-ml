"""
Evaluation metrics and helpers for Loan Approval ML.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute standard binary classification performance metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probability for the positive class (optional).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def print_evaluation_summary(
    model_name: str,
    metrics: Dict[str, float],
    cm: np.ndarray = None,
):
    """
    Print a formatted evaluation report.
    """
    print(f"\n{'='*20} {model_name} {'='*20}")
    for k, v in metrics.items():
        print(f"  {k.upper():<12}: {v:.4f}")

    if cm is not None:
        print("\n  Confusion Matrix:")
        print(f"    [TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}]")
        print(f"    [FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}]")
    print(f"{'='*(42 + len(model_name))}\n")
