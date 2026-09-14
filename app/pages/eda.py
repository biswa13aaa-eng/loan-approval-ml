from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import feature_groups
from components.layout import render_page_header

# Resolve dataset path relative to repository root using Path(__file__).resolve()
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_data.csv"


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("Exploratory Data Analysis", "Discover patterns and relationships within the loan application data.")

    # Defensive dataset fallback: ensure df is loaded from disk if None
    if (df is None or "Loan_Status" not in df) and DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            pass

    if df is None or "Loan_Status" not in df:
        st.error(f"The dataset or target column is unavailable. Checked path: {DATA_PATH}")
        return

    # Data diagnostics expander (for verification and debugging)
    with st.expander("Data diagnostics"):
        st.write("Dataset path:", str(DATA_PATH))
        st.write("Dataset exists:", DATA_PATH.exists())
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Loan_Status values:", df["Loan_Status"].value_counts(dropna=False))

    status_map = {
        "Y": "Approved",
        "N": "Rejected"
    }

    # 1. Target Distribution
    render_page_header("Target distribution", "Distribution of approved (Y) vs rejected (N) loan applications in the benchmark dataset.")
    target = df["Loan_Status"].astype(str).str.strip().str.upper()
    target_counts = (
        target.map(status_map)
        .fillna(target)
        .value_counts()
        .rename_axis("Loan Status")
        .reset_index(name="Count")
    )

    fig_target = px.bar(
        target_counts,
        x="Loan Status",
        y="Count",
        text="Count",
        title="Loan Approval Distribution",
        color="Loan Status",
        color_discrete_map={"Approved": "#16a34a", "Rejected": "#dc2626"}
    )
    fig_target.update_traces(textposition="outside")
    fig_target.update_layout(yaxis_title="Applications", showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)

    # 2. Credit History vs Approval
    render_page_header("Credit History vs Approval", "Credit history is the single strongest empirical predictor of loan outcomes.")
    if "Credit_History" in df.columns:
        ch_df = df.dropna(subset=["Credit_History", "Loan_Status"]).copy()
        ch_df["Credit_Score"] = ch_df["Credit_History"].map({
            1.0: "Meets Guidelines (1.0)",
            0.0: "Does Not Meet (0.0)"
        }).fillna("Unknown")
        ch_df["Outcome"] = ch_df["Loan_Status"].astype(str).str.strip().str.upper().map(status_map)
        ch_counts = ch_df.groupby(["Credit_Score", "Outcome"]).size().reset_index(name="Count")
        fig_ch = px.bar(
            ch_counts,
            x="Credit_Score",
            y="Count",
            color="Outcome",
            barmode="group",
            text="Count",
            title="Credit History vs Loan Approval Outcome",
            color_discrete_map={"Approved": "#16a34a", "Rejected": "#dc2626"}
        )
        fig_ch.update_traces(textposition="outside")
        fig_ch.update_layout(yaxis_title="Applications", xaxis_title="Credit History Status")
        st.plotly_chart(fig_ch, use_container_width=True)

    numerical, categorical = feature_groups(df)

    # 3. Numerical Feature Distributions
    render_page_header("Numerical analysis", "These views show associations in the dataset, not causal effects.")
    if numerical:
        feature = st.selectbox("Select numerical feature", numerical)
        chart_data = df[[feature, "Loan_Status"]].dropna().copy()
        chart_data["Outcome"] = chart_data["Loan_Status"].astype(str).str.strip().str.upper().map(status_map)
        if not chart_data.empty:
            fig_num = px.histogram(
                chart_data,
                x=feature,
                color="Outcome",
                barmode="overlay",
                title=f"{feature} by approval outcome",
                color_discrete_map={"Approved": "#16a34a", "Rejected": "#dc2626"},
            )
            st.plotly_chart(fig_num, use_container_width=True)
        else:
            st.warning(f"No valid records available for {feature}.")
    else:
        st.warning("No numerical features found in the dataset.")

    # 4. Categorical Feature Approval Rates
    render_page_header("Categorical analysis", "Approval rates are descriptive summaries of this dataset.")
    if categorical:
        feature = st.selectbox("Select categorical feature", categorical)
        cat_df = df.dropna(subset=[feature, "Loan_Status"]).copy()
        cat_df["Status_Clean"] = cat_df["Loan_Status"].astype(str).str.strip().str.upper()
        rates = (
            cat_df.groupby(feature)["Status_Clean"]
            .apply(lambda values: (values == "Y").mean())
            .reset_index(name="Approval rate")
        )
        if not rates.empty:
            fig_cat = px.bar(
                rates,
                x=feature,
                y="Approval rate",
                title=f"Approval rate by {feature}",
                range_y=[0, 1.05],
                color_discrete_sequence=["#2563eb"],
                text="Approval rate",
            )
            fig_cat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_cat.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.warning(f"No valid records available for {feature}.")
    else:
        st.warning("No categorical features found in the dataset.")

    # 5. Numerical Correlation Matrix
    render_page_header("Correlation analysis", "Pearson correlations are calculated for numeric columns and a binary target encoding.")
    if numerical:
        corr_data = df[numerical].copy()
        corr_data["Loan_Status"] = (df["Loan_Status"].astype(str).str.strip().str.upper() == "Y").astype(int)
        corr_matrix = corr_data.corr()
        if not corr_matrix.empty and corr_data.shape[1] > 1:
            st.plotly_chart(
                px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="Blues",
                    title="Numerical feature correlation with Loan Status",
                ),
                use_container_width=True,
            )
        else:
            st.warning("Insufficient numerical features to compute correlation matrix.")
    else:
        st.warning("No numerical features available for correlation analysis.")
