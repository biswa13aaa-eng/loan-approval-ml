import pandas as pd
import plotly.express as px
import streamlit as st

from components.cards import render_takeaway
from components.charts import apply_editorial_theme
from components.data import feature_label
from components.layout import render_section_marker


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("05 — MODEL EXPLAINABILITY", "Why does the model make its decision?", "Deconstructing global feature importances across the trained production model.")

    importance = results.get("fi", pd.DataFrame()).copy()
    if importance.empty:
        st.info("Feature-importance output is not available for the saved project model.")
        return

    importance["Display feature"] = importance["Feature"].map(feature_label)
    sorted_fi = importance.sort_values("Importance", ascending=False).reset_index(drop=True)

    # 1. Top 3 Dominant Signals in High-Contrast Cards
    render_section_marker("05.1 — TOP EMPIRICAL SIGNALS", "Core Predictor Weights", "The highest weighted attributes driving model classifications.")
    top3 = sorted_fi.head(3)
    c1, c2, c3 = st.columns(3)
    rank_labels = [("TOP SIGNAL #1", "#D6BE1F"), ("SECOND SIGNAL #2", "#B8B56A"), ("THIRD SIGNAL #3", "#8F8D83")]

    for i, col in enumerate([c1, c2, c3]):
        if i < len(top3):
            feat_name = top3.loc[i, "Display feature"]
            feat_score = top3.loc[i, "Importance"]
            badge_title, border_color = rank_labels[i]
            with col:
                st.markdown(
                    f"""
                    <div class="feat-dom-card" style="border-top: 2px solid {border_color};">
                        <div class="feat-dom-badge">{badge_title}</div>
                        <div class="feat-dom-name">{feat_name.title()}</div>
                        <div class="feat-dom-score">{feat_score:.1%}</div>
                        <div class="feat-dom-note">Relative decision weight across trees in the calibrated ensemble pipeline.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 2. Top 10 Feature Importance Chart
    render_section_marker("05.2 — TOP 10 FEATURE RANKING", "Relative Impurity Reduction", "Normalized contribution scores across the ten most influential variables.")
    top10 = sorted_fi.head(10).sort_values("Importance", ascending=True)

    fig_bar = px.bar(
        top10,
        x="Importance",
        y="Display feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale=[[0, "#2A2922"], [0.5, "#B8B56A"], [1, "#D6BE1F"]],
        title="Top 10 Feature Contributions (Production Model)",
    )
    fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_bar.update_xaxes(tickformat=".0%")
    fig_bar.update_coloraxes(showscale=False)
    st.plotly_chart(apply_editorial_theme(fig_bar), use_container_width=True)

    # 3. Full Feature Importance Table
    render_section_marker("05.3 — COMPLETE ATTRIBUTE SPECTRUM", "Full Feature Weights", "Tabular ranking across all input dimensions.")
    table = sorted_fi[["Display feature", "Importance"]].copy()
    table.columns = ["Model Feature", "Relative Importance"]
    st.dataframe(
        table.style.format({"Relative Importance": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    # 4. Governance & Causality Disclaimer
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    render_takeaway(
        "Feature importance describes statistical associations and tree-split frequency within historic training records. It does not establish clinical or economic causation, nor does it imply that changing an attribute will guarantee loan approval.",
        "GOVERNANCE & EXPLAINABILITY NOTICE",
    )

