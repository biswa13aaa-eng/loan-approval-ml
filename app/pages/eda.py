import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import feature_groups
from components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("Exploratory Data Analysis", "Discover patterns and relationships within the loan application data.")
    if df is None or "Loan_Status" not in df:
        st.error("The dataset or target column is unavailable.")
        return

    numerical, categorical = feature_groups(df)

    # 1. Target Distribution
    status_series = df["Loan_Status"].dropna().map({"Y": "Approved", "N": "Rejected"})
    if status_series.empty:
        st.warning("No loan status records available to display target distribution.")
    else:
        counts = status_series.value_counts()
        target_df = pd.DataFrame({
            "Loan status": counts.index,
            "Applications": counts.values,
        })
        fig_target = px.bar(
            target_df,
            x="Loan status",
            y="Applications",
            color="Loan status",
            text="Applications",
            title="Target distribution",
            color_discrete_map={"Approved": "#16a34a", "Rejected": "#dc2626"},
        )
        fig_target.update_traces(textposition="outside")
        fig_target.update_layout(yaxis_title="Applications", showlegend=False)
        st.plotly_chart(fig_target, use_container_width=True)

    # 2. Numerical Feature Distributions
    render_page_header("Numerical analysis", "These views show associations in the dataset, not causal effects.")
    if numerical:
        feature = st.selectbox("Select numerical feature", numerical)
        chart_data = df[[feature, "Loan_Status"]].dropna().assign(
            Outcome=lambda x: x["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
        )
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

    # 3. Categorical Feature Approval Rates
    render_page_header("Categorical analysis", "Approval rates are descriptive summaries of this dataset.")
    if categorical:
        feature = st.selectbox("Select categorical feature", categorical)
        rates = (
            df.dropna(subset=[feature, "Loan_Status"])
            .groupby(feature)["Loan_Status"]
            .apply(lambda values: (values == "Y").mean())
            .reset_index(name="Approval rate")
        )
        if not rates.empty:
            fig_cat = px.bar(
                rates,
                x=feature,
                y="Approval rate",
                title=f"Approval rate by {feature}",
                range_y=[0, 1],
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

    # 4. Numerical Correlation Matrix
    render_page_header("Correlation analysis", "Pearson correlations are calculated for numeric columns and a binary target encoding.")
    if numerical:
        corr_data = df[numerical].copy()
        corr_data["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
        corr_matrix = corr_data.corr()
        if not corr_matrix.empty and corr_data.shape[1] > 1:
            st.plotly_chart(
                px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="Blues",
                    title="Numerical feature correlation",
                ),
                use_container_width=True,
            )
        else:
            st.warning("Insufficient numerical features to compute correlation matrix.")
    else:
        st.warning("No numerical features available for correlation analysis.")
