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
from components.layout import load_css, render_sidebar, render_top_navigation, render_editorial_footer
from pages import about, dataset, eda, feature_importance, models as models_view, overview, predictor

st.set_page_config(page_title="LOANWISE AI", page_icon="L", layout="wide", initial_sidebar_state="expanded")
load_css()


@st.cache_data
def load_data() -> pd.DataFrame | None:
    data_path = RAW_DATA_FILE if RAW_DATA_FILE.exists() else (BASE_DIR / "data" / "raw" / "loan_data.csv")
    if not data_path.exists():
        st.error(f"Dataset file not found at: {data_path}")
        return None
    df = pd.read_csv(data_path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "": None})
    return df


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
            except Exception as exc:
                st.error(f"Failed to load model {name} from {path}: {exc}")
        else:
            st.warning(f"Model file {path.name} not found in {MODELS_DIR}")
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
    render_top_navigation()
    views = {
        "Overview": overview,
        "01  Overview": overview,
        "Dataset": dataset,
        "02  Dataset": dataset,
        "Exploration": eda,
        "03  Exploration": eda,
        "Exploratory Analysis": eda,
        "Models": models_view,
        "04  Models": models_view,
        "Model Performance": models_view,
        "Features": feature_importance,
        "05  Features": feature_importance,
        "Feature Importance": feature_importance,
        "Predictor": predictor,
        "06  Predictor": predictor,
        "Loan Predictor": predictor,
        "About": about,
        "07  About": about,
        "About Project": about,
    }
    try:
        if selected_page in views:
            views[selected_page].render(df, models, results)
        else:
            st.error(f"Page '{selected_page}' not recognized. Available views: {list(views.keys())}")
    except Exception as exc:
        import traceback
        st.error(f"Error rendering view '{selected_page}': {exc}\n\n{traceback.format_exc()}")
    render_editorial_footer()


if __name__ == "__main__":
    main()
