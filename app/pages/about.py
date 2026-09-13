"""
Page 7: About LOANWISE AI & Project Documentation.
"""

import streamlit as st
import pandas as pd
try:
    from components.cards import render_takeaway
    from components.layout import render_page_header
except ImportError:
    from app.components.cards import render_takeaway
    from app.components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict):
    render_page_header(
        "About LOANWISE AI",
        "Project architecture, machine learning methodology, limitations, and viva voce defense summary.",
    )

    # Project Objective & Overview
    st.markdown("""
    <div class="content-card">
        <h3>🎯 Project Objective & Problem Statement</h3>
        <p style="color: #334155; line-height: 1.6;">
            <b>LOANWISE AI</b> is an academic Machine Learning project developed by a second-year B.Tech student to automate 
            commercial retail loan underwriting. The objective is to build a robust, reproducible binary classification model 
            that evaluates applicant creditworthiness, predicting whether an application will be <b>Approved (Y)</b> or 
            <b>Rejected (N)</b> while minimizing bad debt default risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="content-card">
            <h3>🔬 Machine Learning Methodology</h3>
            <ul style="color: #334155; line-height: 1.7;">
                <li><b>Supervised Binary Classification:</b> Target is discrete (Loan_Status = 1 or 0).</li>
                <li><b>Zero Data Leakage:</b> Stratified 80/20 train/test split executed prior to fitting transformers.</li>
                <li><b>Scikit-Learn Pipelines:</b> Median imputation for skewed numericals; Mode for categoricals; One-Hot Encoding; StandardScaler.</li>
                <li><b>5-Fold Stratified Cross-Validation:</b> Verifies out-of-fold generalization stability.</li>
                <li><b>Hyperparameter Tuning:</b> GridSearchCV performed on Random Forest across 72 configurations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="content-card">
            <h3>💻 Technology Stack</h3>
            <ul style="color: #334155; line-height: 1.7;">
                <li><b>Core Language:</b> Python 3.14</li>
                <li><b>ML Algorithms:</b> Scikit-Learn 1.9.1, XGBoost 3.4.1</li>
                <li><b>Data Processing:</b> Pandas 3.0.5, NumPy 2.5.3</li>
                <li><b>Visualizations:</b> Plotly 7.0.0, Matplotlib 3.11.2, Seaborn 0.13.2</li>
                <li><b>Frontend Application:</b> Streamlit 1.63.0</li>
                <li><b>Academic Reporting:</b> ReportLab 5.0.1 (15-Page PDF Report)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <h3>⚠️ System Limitations & Governance Considerations</h3>
        <p style="color: #334155; line-height: 1.6;">
            While LOANWISE AI demonstrates strong discriminative performance (83.74% Accuracy, 87.55% ROC-AUC), retail deployment 
            in an enterprise banking environment would require addressing the following boundaries:
        </p>
        <ol style="color: #334155; line-height: 1.7;">
            <li><b>Sample Scale:</b> The benchmark dataset contains 614 rows. Industrial underwriting models utilize hundreds of thousands of historical multi-year records.</li>
            <li><b>Granular Financial Omissions:</b> Key underwriting variables such as exact numerical FICO credit scores, total revolving credit utilization, existing asset collateral, and debt-to-income limits were absent from the raw dataset.</li>
            <li><b>Class Imbalance Management:</b> The training sample exhibits a 68.7% approval baseline, requiring probability threshold calibration to avoid over-approvals.</li>
            <li><b>Regulatory Compliance:</b> Automated credit scoring models are subject to FCRA (Fair Credit Reporting Act) and ECOA (Equal Credit Opportunity Act) standards requiring detailed adverse action reporting.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <h3>🚀 Future Roadmap</h3>
        <ul style="color: #334155; line-height: 1.7;">
            <li><b>SHAP Explainability:</b> Integration of SHAP TreeExplainer to produce individual waterfall plots for adverse action letters.</li>
            <li><b>Cost-Sensitive Thresholding:</b> Calibrate the decision threshold based on asymmetric banking costs ($10,000 default cost vs. $500 lost interest margin).</li>
            <li><b>Cloud Containerization:</b> Package the complete application into Docker for deployment on cloud container engines.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    render_takeaway(
        "LOANWISE AI was developed for academic evaluation. All benchmark scores are documented in reports/results/ "
        "and reproducible via 'python -m src.train'.",
        title="Academic Integrity Notice",
    )
