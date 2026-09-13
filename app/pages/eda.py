"""
Page 3: Exploratory Data Analysis (EDA) for LOANWISE AI.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
try:
    from components.cards import render_takeaway
    from components.layout import render_page_header
except ImportError:
    from app.components.cards import render_takeaway
    from app.components.layout import render_page_header

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"


def render(df: pd.DataFrame, models: dict, results: dict):
    render_page_header(
        "Exploratory Data Analysis",
        "Discovering statistical patterns and associations that correlate with loan approval decisions.",
    )

    eda_sections = [
        "1. Credit History vs Loan Approval",
        "2. Financial Income & Loan Amount vs Decision",
        "3. Categorical Factors Impact",
        "4. Numerical Distributions & Skewness",
        "5. Correlation Matrix Heatmap",
        "6. Target Distribution Analysis",
    ]

    selected_eda = st.selectbox("Select Analytical Section:", eda_sections)

    if selected_eda.startswith("1"):
        st.markdown("""
        <div class="content-card">
            <h3>💳 Credit History Compliance vs. Approval Outcome</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Evaluating loan approval proportions conditioned on past credit compliance (Credit_History = 1.0 vs. 0.0).
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_path = FIG_DIR / "02_credit_history_vs_approval.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        render_takeaway(
            "Applicants with documented adherence to credit guidelines (Credit_History = 1.0) exhibit an approval rate "
            "of approximately 79.6%. Conversely, applicants with documented default history (Credit_History = 0.0) drop to "
            "an approval rate of only 8.0%. Note: This pattern shows a powerful statistical association within this dataset "
            "and does not establish direct causation.",
            title="Empirical Observation",
        )

    elif selected_eda.startswith("2"):
        st.markdown("""
        <div class="content-card">
            <h3>💰 Total Household Income & Loan Request vs. Decision</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Comparing log-scaled total income (Applicant + Coapplicant) and requested loan principal between approved and rejected applicants.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_path = FIG_DIR / "04_income_vs_loan_approval.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        render_takeaway(
            "The median household income for approved applications closely mirrors rejected applications. High standalone income "
            "does not guarantee approval if the applicant possesses an adverse credit history or an excessive loan-to-income ratio. "
            "This highlights the multi-dimensional nature of credit underwriting.",
            title="The Income Paradox",
        )

    elif selected_eda.startswith("3"):
        st.markdown("""
        <div class="content-card">
            <h3>🏘️ Categorical Demographics & Location Impact</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Evaluating approval rate percentages across property location, education, marital status, and dependents.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic interactive feature explorer
        cat_feature = st.selectbox(
            "Choose a Categorical Factor to Inspect Dynamically:",
            ["Property_Area", "Education", "Married", "Dependents", "Self_Employed", "Gender"],
        )

        clean_df = df.dropna(subset=[cat_feature, "Loan_Status"]).copy()
        rates = clean_df.groupby(cat_feature)["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100).reset_index()
        rates.columns = [cat_feature, "Approval Rate (%)"]

        fig = px.bar(
            rates,
            x=cat_feature,
            y="Approval Rate (%)",
            color=cat_feature,
            color_discrete_sequence=px.colors.qualitative.Prism,
            text=rates["Approval Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        )
        fig.add_hline(y=68.7, line_dash="dash", line_color="gray", annotation_text="Dataset Avg (68.7%)")
        fig.update_layout(
            height=320,
            yaxis=dict(range=[0, 100]),
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_all_path = FIG_DIR / "05_categorical_vs_approval.png"
        if fig_all_path.exists():
            st.image(str(fig_all_path), use_container_width=True)

        render_takeaway(
            "Semiurban applicants exhibit the highest approval frequency (~76.8%), exceeding Urban (~65.8%) and Rural (~61.5%). "
            "Graduates also demonstrate modestly higher approval rates than non-graduates. These demographic differences reflect "
            "historical lending correlations in the training sample and must be audited to prevent algorithmic disparity.",
            title="Demographic Insights",
        )

    elif selected_eda.startswith("4"):
        st.markdown("""
        <div class="content-card">
            <h3>📈 Numerical Feature Distributions & Skewness Audit</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Inspecting probability density and positive skewness across applicant earnings and requested loan amounts.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_path = FIG_DIR / "03_numerical_distributions.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        render_takeaway(
            "ApplicantIncome and CoapplicantIncome display severe positive skewness with high-income outliers ($81,000 max vs $3,812 median). "
            "Consequently, feature scaling (StandardScaler) and median imputation are essential to prevent linear models from being "
            "disproportionately influenced by extreme values.",
            title="Distribution Properties",
        )

    elif selected_eda.startswith("5"):
        st.markdown("""
        <div class="content-card">
            <h3>🔥 Correlation Matrix Heatmap</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Pairwise Pearson correlation coefficients among continuous financials, loan terms, and binary loan status.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_path = FIG_DIR / "06_correlation_heatmap.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        render_takeaway(
            "Credit_History exhibits the strongest linear correlation with Loan_Status (r ≈ 0.54). LoanAmount shows moderate "
            "correlation with Total_Income (r ≈ 0.62), confirming that banks generally calibrate loan limits according to household earning capacity.",
            title="Correlation Structure",
        )

    elif selected_eda.startswith("6"):
        st.markdown("""
        <div class="content-card">
            <h3>🎯 Target Class Distribution</h3>
            <p style="color: #64748B; font-size: 0.9rem;">
                Breakdown of approved versus rejected historical loan applications.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig_path = FIG_DIR / "01_target_distribution.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        render_takeaway(
            "Approximately 68.7% of all applications in the dataset were approved. Due to this class imbalance, accuracy alone is "
            "an insufficient performance metric. The model must be evaluated on Precision, Recall, F1-Score, and ROC-AUC.",
            title="Class Imbalance Considerations",
        )
