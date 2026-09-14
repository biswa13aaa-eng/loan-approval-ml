import pandas as pd
import plotly.express as px
import streamlit as st

from components.cards import render_kpi_card
from components.data import feature_groups, input_columns
from components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("Dataset Explorer", "Explore the applicant data used to train and evaluate the models.")
    if df is None:
        st.error("The dataset is unavailable. Check data/raw/loan_data.csv.")
        return
    numerical, categorical = feature_groups(df)
    for col, args in zip(st.columns(4), [("Rows", len(df), "Applicant records"), ("Columns", len(df.columns), "Raw columns"), ("Numerical features", len(numerical), "Model inputs"), ("Categorical features", len(categorical), "Model inputs")]):
        with col: render_kpi_card(*map(str, args))
    render_page_header("Dataset preview", "Filter across all raw columns; no values are altered here.")
    query = st.text_input("Search records")
    shown = df if not query else df[df.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False).any(), axis=1)]
    st.dataframe(shown.head(st.slider("Rows to display", 5, min(100, len(shown)), min(20, len(shown)))), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        render_page_header("Data quality")
        missing = df.isna().sum().rename("Missing values").reset_index().rename(columns={"index": "Feature"})
        st.dataframe(missing[missing["Missing values"] > 0], use_container_width=True, hide_index=True)
        st.caption(f"Duplicate rows: {int(df.duplicated().sum())}")
    with right:
        render_page_header("Target distribution")
        if "Loan_Status" in df:
            counts = df["Loan_Status"].value_counts().rename_axis("Status").reset_index(name="Applications")
            st.plotly_chart(px.bar(counts, x="Status", y="Applications", color="Status", color_discrete_map={"Y":"#2563eb", "N":"#dc2626"}, title="Loan status distribution"), use_container_width=True)
    render_page_header("Feature dictionary", "Derived from the raw schema; names are kept as documented in the dataset.")
    dictionary = pd.DataFrame({"Feature": df.columns, "Data type": df.dtypes.astype(str), "Missing values": df.isna().sum().values, "Distinct values": df.nunique(dropna=True).values})
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
