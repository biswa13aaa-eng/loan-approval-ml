import pandas as pd
import streamlit as st

from components.data import display_name, feature_groups
from components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("About LOANWISE AI", "Project information drawn from the project configuration and saved artifacts.")
    numerical, categorical = feature_groups(df) if df is not None else ([], [])
    metadata = results.get("metadata", {})
    st.markdown("### Project objective\nLOANWISE AI presents a loan-approval classification project through an interactive analysis and prediction interface.")
    st.markdown(f"### Dataset\nThe loaded dataset contains {len(df) if df is not None else 'an unavailable number of'} records. The target column is `Loan_Status`; the application excludes `Loan_ID` from prediction inputs.")
    st.markdown(f"### Machine learning models\nAvailable saved models: {', '.join(models) if models else 'none found'}. The metadata-selected model is {display_name(metadata.get('best_model_name', 'unavailable'))}.")
    st.markdown(f"### Preprocessing\nThe saved pipeline applies its established feature engineering, imputation, encoding, and scaling steps during inference. The raw inputs include {len(numerical)} numerical and {len(categorical)} categorical features.")
    st.markdown("### Evaluation metrics\nThe project comparison artifact reports Accuracy, Precision, Recall, F1, and ROC-AUC on the configured holdout evaluation.")
    st.markdown("### Technology stack\nPython, Pandas, scikit-learn, Plotly, joblib, and Streamlit.")
    st.markdown("### Machine learning workflow\nData → preprocessing → model training → evaluation → saved pipeline inference.")
    st.markdown("### Limitations\nThis interface reports the behavior of a historical dataset and its trained models. It is an academic project, not a production lending decision system. Feature importance and predictions are not causal claims.")
    st.markdown("### Future improvements\nAdd model monitoring, threshold validation, fairness review, and documented human-review workflows before any real-world use.")
    st.caption("Student/team information: add project-specific details here when available.")
