from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_editorial_theme
from components.data import feature_groups
from components.layout import render_section_marker

# Resolve dataset path relative to repository root using Path(__file__).resolve()
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_data.csv"


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("03 — EXPLORATION", "WHAT DOES THE DATA TELL US?", "Find the statistical signals and empirical patterns that separate approved and rejected applications.")

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

    # 1. Approval Distribution
    render_section_marker("03.1 — APPROVAL DISTRIBUTION", "Loan Approval Balance", "Class distribution of historical loan applications in the benchmark dataset.")
    target = df["Loan_Status"].astype(str).str.strip().str.upper()
    target_counts = (
        target.map(status_map)
        .fillna(target)
        .value_counts()
        .rename_axis("Loan Status")
        .reset_index(name="Count")
    )
    target_counts["Count"] = target_counts["Count"].astype(int)

    fig_target = px.bar(
        target_counts,
        x="Loan Status",
        y="Count",
        text="Count",
        title="Historical Loan Status Outcome (Approved: 422, Rejected: 192)",
        color="Loan Status",
        color_discrete_map={"Approved": "#22C55E", "Rejected": "#EF4444"}
    )
    fig_target.update_traces(textposition="outside")
    fig_target.update_layout(yaxis_title="Applications", showlegend=False)
    st.plotly_chart(apply_editorial_theme(fig_target), use_container_width=True)

    # 2. Credit History Impact
    render_section_marker("03.2 — CREDIT HISTORY", "Credit History Guidelines", "Empirically the single strongest historical determinant of loan approval.")
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
            title="Loan Status Distribution by Credit History Compliance",
            color_discrete_map={"Approved": "#22C55E", "Rejected": "#EF4444"}
        )
        fig_ch.update_traces(textposition="outside")
        fig_ch.update_layout(yaxis_title="Applications", xaxis_title="Credit History Compliance")
        st.plotly_chart(apply_editorial_theme(fig_ch), use_container_width=True)

    numerical, categorical = feature_groups(df)

    # 3. Income Patterns
    render_section_marker("03.3 — INCOME PATTERNS", "Applicant Financial Profile", "Comparing applicant and combined household income distributions across approval outcomes.")
    if numerical:
        income_features = [col for col in ["ApplicantIncome", "CoapplicantIncome"] if col in df.columns]
        if income_features:
            selected_income = st.selectbox("Select income dimension", income_features)
            chart_data = df[[selected_income, "Loan_Status"]].dropna().copy()
            chart_data["Outcome"] = chart_data["Loan_Status"].astype(str).str.strip().str.upper().map(status_map)
            fig_income = px.histogram(
                chart_data,
                x=selected_income,
                color="Outcome",
                barmode="overlay",
                title=f"{selected_income} Distribution by Approval Outcome",
                color_discrete_map={"Approved": "#22C55E", "Rejected": "#EF4444"},
            )
            st.plotly_chart(apply_editorial_theme(fig_income), use_container_width=True)

    # 4. Loan Amount & Terms
    render_section_marker("03.4 — LOAN AMOUNT & TERMS", "Financing Request Scale", "Distribution of requested financing and repayment durations by outcome.")
    loan_num_features = [col for col in ["LoanAmount", "Loan_Amount_Term"] if col in df.columns]
    if loan_num_features:
        selected_loan_col = st.selectbox("Select loan feature", loan_num_features)
        chart_data_loan = df[[selected_loan_col, "Loan_Status"]].dropna().copy()
        chart_data_loan["Outcome"] = chart_data_loan["Loan_Status"].astype(str).str.strip().str.upper().map(status_map)
        fig_loan = px.histogram(
            chart_data_loan,
            x=selected_loan_col,
            color="Outcome",
            barmode="overlay",
            title=f"{selected_loan_col} by Approval Outcome",
            color_discrete_map={"Approved": "#22C55E", "Rejected": "#EF4444"},
        )
        st.plotly_chart(apply_editorial_theme(fig_loan), use_container_width=True)

    # 5. Categorical Demographics Approval Rates
    render_section_marker("03.5 — APPLICANT PROFILE", "Categorical Approval Rates", "Empirical approval rate variations across demographic, household, and property tiers.")
    if categorical:
        cat_feature = st.selectbox("Select categorical attribute", categorical)
        cat_df = df.dropna(subset=[cat_feature, "Loan_Status"]).copy()
        cat_df["Status_Clean"] = cat_df["Loan_Status"].astype(str).str.strip().str.upper()
        rates = (
            cat_df.groupby(cat_feature)["Status_Clean"]
            .apply(lambda values: (values == "Y").mean())
            .reset_index(name="Approval rate")
        )
        if not rates.empty:
            fig_cat = px.bar(
                rates,
                x=cat_feature,
                y="Approval rate",
                title=f"Approval Rate by {cat_feature}",
                range_y=[0, 1.05],
                color_discrete_sequence=["#D6BE1F"],
                text="Approval rate",
            )
            fig_cat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_cat.update_yaxes(tickformat=".0%")
            st.plotly_chart(apply_editorial_theme(fig_cat), use_container_width=True)

    # 6. Feature Correlation
    render_section_marker("03.6 — FEATURE CORRELATION", "Numerical Matrix & Target Association", "Pearson correlation matrix computed for numeric columns with binary target encoding.")
    if numerical:
        corr_data = df[numerical].copy()
        corr_data["Loan_Status"] = (df["Loan_Status"].astype(str).str.strip().str.upper() == "Y").astype(int)
        corr_matrix = corr_data.corr()
        if not corr_matrix.empty and corr_data.shape[1] > 1:
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale=[[0, "#151410"], [0.5, "#2A2922"], [1, "#D6BE1F"]],
                title="Correlation Heatmap: Numeric Inputs & Binary Loan Status",
            )
            st.plotly_chart(apply_editorial_theme(fig_corr), use_container_width=True)

