"""LOANWISE AI Streamlit entry point. Run: streamlit run app/app.py"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
for path in (BASE_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config import MODELS_DIR, MODEL_METADATA_PATH, RAW_DATA_FILE
from components.layout import load_css, render_sidebar
from pages import about, dataset, eda, feature_importance, models as models_view, overview, predictor

st.set_page_config(page_title="LOANWISE AI", page_icon="L", layout="wide", initial_sidebar_state="expanded")
load_css()


@st.cache_data
def load_data() -> pd.DataFrame | None:
    return pd.read_csv(RAW_DATA_FILE) if RAW_DATA_FILE.exists() else None


@st.cache_resource
def load_models() -> dict:
    registry = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Random Forest": MODELS_DIR / "random_forest.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }
    loaded = {}
    for name, path in registry.items():
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    return loaded


@st.cache_data
def load_results() -> dict:
    results_dir = BASE_DIR / "reports" / "results"
    results = {}
    for key, path in {"comparison": results_dir / "model_comparison.csv", "fi": results_dir / "feature_importance.csv"}.items():
        if path.exists():
            results[key] = pd.read_csv(path)
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, encoding="utf-8") as file:
            results["metadata"] = json.load(file)
    return results


def main() -> None:
    df, models, results = load_data(), load_models(), load_results()
    selected_page = render_sidebar()
    views = {"Overview": overview, "Dataset": dataset, "Exploratory Analysis": eda,
             "Model Performance": models_view, "Feature Importance": feature_importance,
             "Loan Predictor": predictor, "About Project": about}
    try:
        views[selected_page].render(df, models, results)
    except Exception:
        st.error("Unable to load this view. Please refresh the page or verify the project artifacts.")


if __name__ == "__main__":
    main()
