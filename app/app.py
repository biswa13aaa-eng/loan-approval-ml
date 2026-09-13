"""
LOANWISE AI — Primary Application Entry Point & Router.
Execution: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Ensure project root and app directory are on sys.path
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

for p in [str(BASE_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import joblib
import pandas as pd
import streamlit as st

from src.config import (
    RAW_DATA_FILE,
    MODELS_DIR,
    MODEL_METADATA_PATH,
)

# Import custom layout and pages with robust fallback
try:
    from components.layout import load_css, render_sidebar
    from pages import (
        overview,
        dataset,
        eda,
        models as models_view,
        feature_importance,
        predictor,
        about,
    )
except ImportError:
    from app.components.layout import load_css, render_sidebar
    from app.pages import (
        overview,
        dataset,
        eda,
        models as models_view,
        feature_importance,
        predictor,
        about,
    )

# Page configuration
st.set_page_config(
    page_title="LOANWISE AI — Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom fintech styling
load_css()


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads and caches the raw dataset.
    """
    if not RAW_DATA_FILE.exists():
        return None
    return pd.read_csv(RAW_DATA_FILE)


@st.cache_resource
def load_models() -> dict:
    """
    Loads and caches candidate model pipelines.
    """
    models = {}
    model_registry = {
        "Random Forest (Selected)": MODELS_DIR / "random_forest.joblib",
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }
    for name, path in model_registry.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.sidebar.warning(f"Note: Could not load {name} ({e})")
    return models


@st.cache_data
def load_results() -> dict:
    """
    Loads verified benchmark results and model metadata.
    """
    results = {}
    comp_csv = BASE_DIR / "reports" / "results" / "model_comparison.csv"
    if comp_csv.exists():
        results["comparison"] = pd.read_csv(comp_csv)

    fi_csv = BASE_DIR / "reports" / "results" / "feature_importance.csv"
    if fi_csv.exists():
        results["fi"] = pd.read_csv(fi_csv)

    tune_json = BASE_DIR / "reports" / "results" / "tuning_comparison.json"
    if tune_json.exists():
        with open(tune_json, "r", encoding="utf-8") as f:
            results["tuning"] = json.load(f)

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            results["metadata"] = json.load(f)

    return results


def main():
    # Load cached resources
    df = load_data()
    models = load_models()
    results = load_results()

    # Render persistent sidebar and get selected navigation page
    selected_page = render_sidebar()

    # Route to selected page module
    try:
        if "Overview" in selected_page:
            overview.render(df, models, results)
        elif "Dataset" in selected_page:
            dataset.render(df, models, results)
        elif "Exploratory" in selected_page:
            eda.render(df, models, results)
        elif "Performance" in selected_page:
            models_view.render(df, models, results)
        elif "Importance" in selected_page:
            feature_importance.render(df, models, results)
        elif "Predictor" in selected_page:
            predictor.render(df, models, results)
        elif "About" in selected_page:
            about.render(df, models, results)
        else:
            overview.render(df, models, results)
    except Exception as e:
        st.error(f"Application Error: Unable to render {selected_page}. Details: {e}")


if __name__ == "__main__":
    main()
