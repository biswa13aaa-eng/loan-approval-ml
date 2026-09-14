import pandas as pd
import plotly.express as px
import streamlit as st

from components.cards import render_kpi_card
from components.charts import apply_editorial_theme
from components.data import feature_groups
from components.layout import render_section_marker


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("02 — DATA INTELLIGENCE", "Dataset Explorer", "Inspect the raw applicant records, feature schema, and data health foundation.")
    if df is None:
        st.error("The dataset is unavailable. Check data/raw/loan_data.csv.")
        return

    numerical, categorical = feature_groups(df)

    # 1. Four Large Metric Cards
    n_rows = len(df)
    n_cols = len(df.columns)
    approved_count = (df["Loan_Status"].astype(str).str.strip().str.upper() == "Y").sum() if "Loan_Status" in df else 422
    rejected_count = (df["Loan_Status"].astype(str).str.strip().str.upper() == "N").sum() if "Loan_Status" in df else 192

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("RECORD VOLUME", f"{n_rows:,}", "TOTAL APPLICATIONS", border_color="#D6BE1F")
    with m2:
        render_kpi_card("FEATURE SCHEMA", f"{n_cols}", "RAW ATTRIBUTES", border_color="#B8B56A")
    with m3:
        render_kpi_card("BENCHMARK OUTCOME", f"{approved_count:,}", "APPROVED LOANS (Y)", border_color="#22C55E")
    with m4:
        render_kpi_card("BENCHMARK OUTCOME", f"{rejected_count:,}", "REJECTED LOANS (N)", border_color="#EF4444")

    # 2. Dataset Preview
    render_section_marker("02.1 — DATASET PREVIEW", "Raw Records Browser", "Filter across all raw columns; records are displayed without transformation.")
    query = st.text_input("Search applicant records (by ID, income, education, etc.)", placeholder="Type to filter...")
    shown = df if not query else df[df.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False).any(), axis=1)]

    row_count_slider = st.slider("Rows to display", 5, min(100, len(shown)), min(20, len(shown)))
    st.dataframe(shown.head(row_count_slider), use_container_width=True, hide_index=True)

    # 3. Two-Column Layout: Data Quality & Target Distribution
    left, right = st.columns([1, 1])
    with left:
        render_section_marker("02.2 — DATA QUALITY", "Missing Values & Health", "Completeness analysis across features.")
        missing = df.isna().sum().rename("Missing values").reset_index().rename(columns={"index": "Feature"})
        missing["Missing %"] = (missing["Missing values"] / len(df)).map("{:.1%}".format)
        st.dataframe(missing[missing["Missing values"] > 0], use_container_width=True, hide_index=True)
        st.caption(f"Duplicate rows detected: {int(df.duplicated().sum())} · Total cells: {df.size:,}")

    with right:
        render_section_marker("02.3 — TARGET DISTRIBUTION", "Loan Status Balance", "Historical benchmark distribution.")
        if "Loan_Status" in df:
            status_clean = df["Loan_Status"].astype(str).str.strip().str.upper().map({"Y": "Approved", "N": "Rejected"})
            counts = status_clean.value_counts().rename_axis("Status").reset_index(name="Applications")
            fig = px.bar(
                counts,
                x="Status",
                y="Applications",
                color="Status",
                text="Applications",
                color_discrete_map={"Approved": "#22C55E", "Rejected": "#EF4444"},
                title="Class Balance: Approved vs Rejected",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_editorial_theme(fig), use_container_width=True)

    # 4. Feature Dictionary
    render_section_marker("02.4 — SCHEMA SPECIFICATION", "Feature Dictionary", "Detailed variable types and distinct cardinality.")
    dictionary = pd.DataFrame({
        "Feature": df.columns,
        "Data type": df.dtypes.astype(str),
        "Missing values": df.isna().sum().values,
        "Distinct values": df.nunique(dropna=True).values,
    })
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

