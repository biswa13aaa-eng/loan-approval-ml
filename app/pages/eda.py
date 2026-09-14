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
    status = df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
    st.plotly_chart(px.histogram(status, color=status, title="Target distribution", labels={"value":"Loan status", "count":"Applications"}, color_discrete_map={"Approved":"#16a34a", "Rejected":"#dc2626"}), use_container_width=True)
    render_page_header("Numerical analysis", "These views show associations in the dataset, not causal effects.")
    if numerical:
        feature = st.selectbox("Select numerical feature", numerical)
        chart_data = df[[feature, "Loan_Status"]].dropna().assign(Outcome=lambda x: x["Loan_Status"].map({"Y":"Approved", "N":"Rejected"}))
        st.plotly_chart(px.histogram(chart_data, x=feature, color="Outcome", barmode="overlay", title=f"{feature} by approval outcome"), use_container_width=True)
    render_page_header("Categorical analysis", "Approval rates are descriptive summaries of this dataset.")
    if categorical:
        feature = st.selectbox("Select categorical feature", categorical)
        rates = df.dropna(subset=[feature, "Loan_Status"]).groupby(feature)["Loan_Status"].apply(lambda values: (values == "Y").mean()).reset_index(name="Approval rate")
        st.plotly_chart(px.bar(rates, x=feature, y="Approval rate", title=f"Approval rate by {feature}", range_y=[0, 1]).update_yaxes(tickformat=".0%"), use_container_width=True)
    render_page_header("Correlation analysis", "Pearson correlations are calculated for numeric columns and a binary target encoding.")
    corr_data = df[numerical].copy()
    corr_data["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    if corr_data.shape[1] > 1:
        st.plotly_chart(px.imshow(corr_data.corr(), text_auto=".2f", color_continuous_scale="Blues", title="Numerical feature correlation"), use_container_width=True)
