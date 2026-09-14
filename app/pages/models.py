import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from components.data import best_row, display_name
from components.layout import render_page_header
from src.data_loader import split_data


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("Model Performance", "Compare how different machine learning models perform on unseen data.")
    comparison = results.get("comparison", pd.DataFrame())
    if comparison.empty:
        st.warning("Model comparison results are unavailable.")
        return
    top = best_row(comparison, "F1")
    if top is not None:
        st.info(f"Highest holdout F1: {display_name(top['Model'])} ({top['F1']:.2%}).")
    table = comparison.copy(); table["Model"] = table["Model"].map(display_name)
    st.dataframe(table.style.format({key:"{:.2%}" for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    for col, metric in zip((left, right), ("F1", "Accuracy")):
        with col:
            st.plotly_chart(px.bar(table, x="Model", y=metric, color_discrete_sequence=["#2563eb"], title=f"{metric} score comparison").update_yaxes(range=[0, 1], tickformat=".0%"), use_container_width=True)
    render_page_header("Confusion matrix", "Computed on the project's reproducible stratified holdout split.")
    if df is None or not models:
        st.warning("Dataset or trained models are unavailable for this view.")
        return
    selected = st.selectbox("Select model", list(models))
    try:
        _, X_test, _, y_test = split_data(df)
        matrix = confusion_matrix(y_test, models[selected].predict(X_test), labels=[0, 1])
        st.plotly_chart(px.imshow(matrix, text_auto=True, x=["Rejected", "Approved"], y=["Rejected", "Approved"], labels={"x":"Predicted", "y":"Actual", "color":"Count"}, color_continuous_scale="Blues", title=f"{selected} confusion matrix"), use_container_width=True)
        tn, fp, fn, tp = matrix.ravel()
        st.caption(f"True negative: {tn} · False positive: {fp} · False negative: {fn} · True positive: {tp}")
    except Exception:
        st.warning("Unable to compute the confusion matrix from the saved pipeline.")
    criterion = results.get("metadata", {}).get("best_model_name")
    if criterion:
        st.info(f"The saved project metadata selects {display_name(criterion)} using its recorded evaluation procedure.")
