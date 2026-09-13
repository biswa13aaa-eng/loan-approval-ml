"""
Page 2: Dataset Explorer & Hygiene Audit for LOANWISE AI.
"""

import streamlit as st
import pandas as pd
try:
    from components.cards import render_kpi_card
    from components.charts import render_target_donut
    from components.layout import render_page_header
except ImportError:
    from app.components.cards import render_kpi_card
    from app.components.charts import render_target_donut
    from app.components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict):
    render_page_header(
        "Dataset Explorer & Hygiene Audit",
        "Inspect the raw loan benchmark records, data quality metrics, and feature properties.",
    )

    if df is None:
        st.error("Dataset not found at data/raw/loan_data.csv. Please verify file path.")
        return

    # Dynamic KPI cards calculated directly from dataset
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Credit_History"]
    n_duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Total Rows", f"{n_rows:,}", "Applicant Records", "#2563EB")
    with c2:
        render_kpi_card("Total Columns", f"{n_cols}", "12 Features + 1 Target", "#10B981")
    with c3:
        render_kpi_card("Numerical Features", f"{len(num_cols)}", "Financial & Term Variables", "#8B5CF6")
    with c4:
        render_kpi_card("Categorical Features", f"{len(cat_cols)}", "Demographics & Location", "#F59E0B")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Interactive Dataframe View
    st.markdown("""
    <div class="content-card">
        <h3>🔍 Raw Dataset Sample</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 12px;">
            Filter and preview applicant records. All values reflect un-transformed raw features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    row_count = st.slider("Number of records to display:", min_value=5, max_value=100, value=15, step=5)
    st.dataframe(df.head(row_count), use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Data Quality & Target Breakdown
    st.markdown("""
    <div class="content-card">
        <h3>📊 Data Quality, Missing Values & Target Balance</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 16px;">
            Auditing missingness and class balance before applying Scikit-Learn pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2 = st.columns([1.2, 1])

    with col_q1:
        st.markdown("#### 🩹 Missing Records Breakdown")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Feature": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / n_rows * 100).round(2),
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        st.dataframe(
            missing_df.style.format({"Missing %": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Duplicate Rows Identified: **{n_duplicates}**")

    with col_q2:
        st.markdown("#### 🎯 Target Distribution (`Loan_Status`)")
        render_target_donut(df)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Expandable: About the Dataset
    with st.expander("ℹ️ ABOUT THE DATASET — Feature Descriptions & Schema"):
        st.markdown("""
        | Attribute Name | Data Type | Domain Values | Business Meaning |
        | :--- | :--- | :--- | :--- |
        | **`Loan_ID`** | String | LP001002 - LP002990 | Unique applicant identifier (dropped prior to ML training). |
        | **`Gender`** | Categorical | Male, Female | Applicant gender classification. |
        | **`Married`** | Categorical | Yes, No | Marital status indicator. |
        | **`Dependents`** | Categorical | 0, 1, 2, 3+ | Number of financially dependent family members. |
        | **`Education`** | Categorical | Graduate, Not Graduate | Highest academic attainment level. |
        | **`Self_Employed`** | Categorical | Yes, No | Independent business owner / freelancer flag. |
        | **`ApplicantIncome`** | Continuous | $150 – $81,000 | Primary borrower's monthly gross earnings (USD). |
        | **`CoapplicantIncome`** | Continuous | $0 – $41,667 | Secondary borrower's monthly gross earnings (USD). |
        | **`LoanAmount`** | Continuous | $9 – $700 ($'000) | Total requested loan principal in thousands. |
        | **`Loan_Amount_Term`** | Discrete | 12 – 480 Months | Scheduled repayment period (mode = 360 months / 30 years). |
        | **`Credit_History`** | Binary Flag | 1.0 (Good), 0.0 (Default) | Past compliance with credit guidelines. |
        | **`Property_Area`** | Categorical | Urban, Semiurban, Rural | Geographic classification of pledged collateral property. |
        | **`Loan_Status`** | Binary Target | Y (Approved), N (Rejected) | Ground-truth historical underwriting outcome. |
        """)
