"""
Plotly interactive chart builders for LOANWISE AI.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
                hole=0.55,
                marker=dict(colors=["#10B981", "#EF4444"]),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, color="#FFFFFF"),
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_bar(comp_df: pd.DataFrame):
    """
    Renders interactive grouped or multi-bar charts for model comparison.
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
        color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B"],
        text_auto=".2%",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
        yaxis=dict(range=[0.7, 1.0], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix_plotly(cm: list or np.ndarray, model_name: str):
    """
    Renders a styled interactive confusion matrix heatmap.
    """
    cm_arr = np.array(cm)
    labels = ["Rejected (0)", "Approved (1)"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            colorscale=[[0, "#EFF6FF"], [1, "#1E40AF"]],
            showscale=False,
            text=[[f"TN: {cm_arr[0][0]}", f"FP: {cm_arr[0][1]}"],
                  [f"FN: {cm_arr[1][0]}", f"TP: {cm_arr[1][1]}"]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
        )
    )
    fig.update_layout(
        title=dict(text=f"Confusion Matrix: {model_name}", font=dict(size=14)),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_plotly(fi_df: pd.DataFrame, top_n: int = 10):
    """
    Renders horizontal bar chart for feature importances.
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
                colorscale=[[0, "#93C5FD"], [1, "#1D4ED8"]],
            ),
            text=top_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=40, t=20, b=20),
        height=380,
        xaxis_title="Relative Importance (Mean Decrease in Impurity)",
        yaxis_title="",
        xaxis=dict(tickformat=".1%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_probability_gauge(prob: float):
    """
    Renders an interactive semi-circular gauge for approval probability.
    """
    pct = prob * 100
    color = "#10B981" if prob >= 0.70 else ("#F59E0B" if prob >= 0.50 else "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.4},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
