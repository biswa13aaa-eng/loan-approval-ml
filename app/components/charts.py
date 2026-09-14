"""
Plotly interactive chart builders for LOANWISE AI.
AI-first editorial dark theme: #0B0B09 background, #F5F2E8 cream, #B8B56A olive, #D6BE1F lime.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def apply_editorial_theme(fig: go.Figure) -> go.Figure:
    """
    Applies the LOANWISE AI editorial theme to any Plotly figure:
    transparent background, cream text, subtle hairline gridlines, and clean typography.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F5F2E8", family="'Plus Jakarta Sans', -apple-system, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="#272620",
            tickfont=dict(color="#8F8D83", size=11),
            title_font=dict(color="#F5F2E8", size=12),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="#272620",
            tickfont=dict(color="#8F8D83", size=11),
            title_font=dict(color="#F5F2E8", size=12),
        ),
        legend=dict(
            font=dict(color="#F5F2E8", size=11),
            bgcolor="rgba(24, 23, 20, 0.8)",
            bordercolor="#272620",
            borderwidth=1,
        ),
        margin=dict(l=24, r=24, t=40, b=24),
    )
    return fig


def render_target_donut(df: pd.DataFrame):
    """
    Renders an interactive Plotly donut chart of the target variable.
    """
    counts = df["Loan_Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Label"] = counts["Status"].map({"Y": "Approved", "N": "Rejected"})

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Label"],
                values=counts["Count"],
                hole=0.62,
                marker=dict(colors=["#22C55E", "#EF4444"], line=dict(color="#0B0B09", width=3)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, color="#F5F2E8"),
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=280,
    )
    st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)


def render_model_comparison_bar(comp_df: pd.DataFrame):
    """
    Renders interactive grouped bars for holdout model comparison.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    clean_df = comp_df.copy()
    clean_df["Model"] = clean_df["Model"].str.replace("_", " ")

    melted = clean_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#D6BE1F", "#B8B56A", "#3B82F6"],
        text_auto=".1%",
    )
    fig.update_layout(
        height=360,
        yaxis=dict(range=[0.6, 1.02], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)


def render_confusion_matrix_plotly(cm: list or np.ndarray, model_name: str):
    """
    Renders a styled interactive confusion matrix heatmap in dark theme.
    """
    cm_arr = np.array(cm)
    labels = ["Rejected (0)", "Approved (1)"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            colorscale=[[0, "#151410"], [0.5, "#2F2D1F"], [1, "#D6BE1F"]],
            showscale=False,
            text=[[f"TN: {cm_arr[0][0]}", f"FP: {cm_arr[0][1]}"],
                  [f"FN: {cm_arr[1][0]}", f"TP: {cm_arr[1][1]}"]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="#F5F2E8"),
        )
    )
    fig.update_layout(
        title=dict(text=f"Confusion Matrix: {model_name}", font=dict(size=14, color="#F5F2E8")),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=300,
    )
    st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)


def render_feature_importance_plotly(fi_df: pd.DataFrame, top_n: int = 10):
    """
    Renders horizontal bar chart for feature importances in dark/lime theme.
    """
    top_df = fi_df.head(top_n).copy()
    top_df = top_df.sort_values("Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],
            y=top_df["Feature"],
            orientation="h",
            marker=dict(
                color=top_df["Importance"],
                colorscale=[[0, "#424036"], [0.5, "#B8B56A"], [1, "#D6BE1F"]],
            ),
            text=top_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=380,
        xaxis_title="Relative Feature Importance",
        yaxis_title="",
        xaxis=dict(tickformat=".1%"),
    )
    st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)


def render_probability_gauge(prob: float):
    """
    Renders an interactive semi-circular gauge for approval probability.
    """
    pct = prob * 100
    color = "#22C55E" if prob >= 0.70 else ("#D6BE1F" if prob >= 0.50 else "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 32, "color": "#F5F2E8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5C5A52"},
                "bar": {"color": color, "thickness": 0.38},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [50, 70], "color": "rgba(214, 190, 31, 0.15)"},
                    {"range": [70, 100], "color": "rgba(34, 197, 94, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#F5F2E8", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)

