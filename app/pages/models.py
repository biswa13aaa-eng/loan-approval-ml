import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from components.charts import apply_editorial_theme
from components.data import best_row, display_name
from components.layout import render_section_marker
from src.data_loader import split_data


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("04 — MODEL LAB", "THREE MODELS. ONE DECISION.", "Empirical evaluation across linear, ensemble, and gradient-boosted architectures on unseen holdout records.")

    comparison = results.get("comparison", pd.DataFrame())
    if comparison.empty:
        st.warning("Model comparison results are unavailable.")
        return

    top = best_row(comparison, "F1")
    top_name = str(top["Model"]).lower() if top is not None else ""

    # 1. Three Modular Model Laboratory Cards
    model_configs = [
        ("01 — BASELINE", "Logistic Regression", "logistic_regression"),
        ("02 — ENSEMBLE", "Random Forest", "random_forest"),
        ("03 — BOOSTING", "XGBoost", "xgboost"),
    ]

    cols = st.columns(3)
    for idx, (tier_label, display_title, key_match) in enumerate(model_configs):
        # Match row from comparison table
        row_match = comparison[comparison["Model"].astype(str).str.lower().str.replace(" ", "_") == key_match]
        if row_match.empty:
            row_match = comparison[comparison["Model"].astype(str).str.contains(display_title, case=False, na=False)]

        is_top = (key_match in top_name) or (top is not None and display_title.lower() in str(top["Model"]).lower())
        rec_badge = '<div class="model-rec-badge">★ RECOMMENDED MODEL</div>' if is_top else ''
        card_class = 'recommended' if is_top else ''

        acc = f"{row_match['Accuracy'].values[0]:.1%}" if not row_match.empty else "N/A"
        f1 = f"{row_match['F1'].values[0]:.1%}" if not row_match.empty else "N/A"
        prec = f"{row_match['Precision'].values[0]:.1%}" if not row_match.empty else "N/A"
        rec = f"{row_match['Recall'].values[0]:.1%}" if not row_match.empty else "N/A"
        auc = f"{row_match['ROC-AUC'].values[0]:.1%}" if not row_match.empty and "ROC-AUC" in row_match else "N/A"

        with cols[idx]:
            st.markdown(
                f"""
                <div class="model-intel-card {card_class}">
                    {rec_badge}
                    <div class="model-tier-num">{tier_label}</div>
                    <div class="model-name">{display_title}</div>
                    <div class="metric-micro-grid">
                        <div class="metric-micro-item">
                            <div class="metric-micro-label">F1 Score</div>
                            <div class="metric-micro-val" style="color: var(--accent-lime);">{f1}</div>
                        </div>
                        <div class="metric-micro-item">
                            <div class="metric-micro-label">Accuracy</div>
                            <div class="metric-micro-val">{acc}</div>
                        </div>
                        <div class="metric-micro-item">
                            <div class="metric-micro-label">Precision</div>
                            <div class="metric-micro-val">{prec}</div>
                        </div>
                        <div class="metric-micro-item">
                            <div class="metric-micro-label">Recall</div>
                            <div class="metric-micro-val">{rec}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Comprehensive Comparison Table
    render_section_marker("04.1 — EVALUATION METRICS", "Holdout Benchmark Matrix", "Standardized metrics computed strictly on the stratified 20% test holdout split.")
    table = comparison.copy()
    table["Model"] = table["Model"].map(display_name)
    st.dataframe(
        table.style.format({key: "{:.2%}" for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] if key in table}),
        use_container_width=True,
        hide_index=True,
    )

    # 3. Metric Comparison Visuals
    render_section_marker("04.2 — PERFORMANCE BENCHMARK", "Metric Distribution", "Side-by-side performance across discrimination (ROC-AUC) and balance (F1).")
    left, right = st.columns(2)
    with left:
        fig_f1 = px.bar(
            table,
            x="Model",
            y="F1",
            text="F1",
            color="Model",
            color_discrete_sequence=["#D6BE1F", "#B8B56A", "#3B82F6"],
            title="Holdout F1-Score Comparison",
        )
        fig_f1.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_f1.update_yaxes(range=[0.6, 1.02], tickformat=".0%")
        fig_f1.update_layout(showlegend=False)
        st.plotly_chart(apply_editorial_theme(fig_f1), use_container_width=True)

    with right:
        fig_acc = px.bar(
            table,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Model",
            color_discrete_sequence=["#D6BE1F", "#B8B56A", "#3B82F6"],
            title="Holdout Accuracy Comparison",
        )
        fig_acc.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_acc.update_yaxes(range=[0.6, 1.02], tickformat=".0%")
        fig_acc.update_layout(showlegend=False)
        st.plotly_chart(apply_editorial_theme(fig_acc), use_container_width=True)

    # 4. Confusion Matrix Deep Dive
    render_section_marker("04.3 — ERROR ANALYSIS", "Holdout Confusion Matrix", "True vs predicted classification breakdown on the project's holdout partition.")
    if df is None or not models:
        st.warning("Dataset or trained models are unavailable for confusion matrix computation.")
        return

    selected_model_name = st.selectbox("Select model for confusion matrix inspection", list(models))
    try:
        _, X_test, _, y_test = split_data(df)
        matrix = confusion_matrix(y_test, models[selected_model_name].predict(X_test), labels=[0, 1])
        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            x=["Rejected (0)", "Approved (1)"],
            y=["Rejected (0)", "Approved (1)"],
            labels={"x": "Predicted Outcome", "y": "Actual Outcome", "color": "Applicant Count"},
            color_continuous_scale=[[0, "#151410"], [0.5, "#2A2922"], [1, "#D6BE1F"]],
            title=f"{selected_model_name} Confusion Matrix",
        )
        st.plotly_chart(apply_editorial_theme(fig_cm), use_container_width=True)

        tn, fp, fn, tp = matrix.ravel()
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("True Negatives (TN)", f"{tn}")
        col_m2.metric("False Positives (FP)", f"{fp}")
        col_m3.metric("False Negatives (FN)", f"{fn}")
        col_m4.metric("True Positives (TP)", f"{tp}")
    except Exception as exc:
        st.warning(f"Unable to compute confusion matrix: {exc}")

