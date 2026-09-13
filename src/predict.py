"""
Inference module for Loan Approval ML.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH


class LoanPredictor:
    """
    Inference helper that loads the serialized pipeline and generates loan decisions.
    """

    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please train the model first using src.train."
            )
        self.pipeline = joblib.load(model_path)

    def predict_one(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a prediction for a single applicant dictionary.

        Args:
            applicant: Dictionary of applicant features.

        Returns:
            Dictionary with prediction decision, probability, and risk level.
        """
        df = pd.DataFrame([applicant])
        return self.predict_batch(df)[0]

    def predict_batch(self, df: pd.DataFrame) -> list:
        """
        Generate predictions for a DataFrame of applicants.
        """
        preds = self.pipeline.predict(df)
        probs = (
            self.pipeline.predict_proba(df)[:, 1]
            if hasattr(self.pipeline, "predict_proba")
            else [float(p) for p in preds]
        )

        results = []
        for pred, prob in zip(preds, probs):
            prob_val = float(prob)
            if prob_val >= 0.70:
                risk = "Low Risk"
            elif prob_val >= 0.50:
                risk = "Moderate Risk"
            else:
                risk = "High Risk"

            results.append({
                "approved": bool(pred == 1),
                "decision": "Approved" if pred == 1 else "Rejected",
                "approval_probability": round(prob_val, 4),
                "risk_level": risk,
            })
        return results


if __name__ == "__main__":
    # Quick sanity check with sample applicants
    predictor = LoanPredictor()

    sample_applicant_1 = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5500,
        "CoapplicantIncome": 2500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
    }

    sample_applicant_2 = {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 2000,
        "CoapplicantIncome": 0,
        "LoanAmount": 200,
        "Loan_Amount_Term": 180,
        "Credit_History": 0.0,
        "Property_Area": "Rural",
    }

    print("Sample 1 Decision:", predictor.predict_one(sample_applicant_1))
    print("Sample 2 Decision:", predictor.predict_one(sample_applicant_2))
