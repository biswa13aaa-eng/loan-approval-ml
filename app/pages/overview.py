import pandas as pd
import streamlit as st

from components.cards import render_kpi_card, render_takeaway
from components.data import best_row, display_name, input_columns
from components.layout import render_app_header, render_page_header


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_app_header()
    if df is None:
        st.error("The dataset is unavailable. Check data/raw/loan_data.csv.")
        return
    comparison = results.get("comparison", pd.DataFrame())
    top = best_row(comparison, "F1")
    metadata = results.get("metadata", {})
    selected = display_name(metadata.get("best_model_name", "Unavailable"))
    cols = st.columns(4)
    values = [("Dataset rows", f"{len(df):,}", "Applications"), ("Input features", str(len(input_columns(df))), "Training inputs"),
              ("Selected model", selected, "Saved production pipeline"), ("Best F1 score", f"{top['F1']:.2%}" if top is not None else "Unavailable", "Holdout set")]
    for col, (label, value, note) in zip(cols, values):
        with col:
            render_kpi_card(label, value, note)

    render_page_header("Project pipeline", "Data flows through the saved preprocessing and model pipeline.")
    st.markdown('<div class="pipeline-line"><span>Data</span><b>→</b><span>Preprocessing</span><b>→</b><span>EDA</span><b>→</b><span>Model training</span><b>→</b><span>Evaluation</span><b>→</b><span>Prediction</span></div>', unsafe_allow_html=True)

    render_page_header("Model performance", "Metrics are loaded from the project evaluation artifact.")
    if comparison.empty:
        st.info("No model comparison artifact is available.")
    else:
        table = comparison.copy()
        table["Model"] = table["Model"].map(display_name)
        st.dataframe(table.style.format({key: "{:.2%}" for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), use_container_width=True, hide_index=True)
        if top is not None:
            render_takeaway(f"{display_name(top['Model'])} achieved the highest F1 score on the holdout set.", "Holdout result")

    render_page_header("Dataset health", "Quality checks are calculated directly from the loaded data.")
    a, b, c = st.columns(3)
    a.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    b.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c.metric("Target values", str(df["Loan_Status"].nunique()) if "Loan_Status" in df else "Unavailable")
