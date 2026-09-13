"""
Loan Approval Classification — Interactive Streamlit Dashboard & Live Prediction Interface.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    RAW_DATA_FILE,
    MODELS_DIR,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)

# Page configuration
st.set_page_config(
    page_title="Loan Approval Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .decision-approved {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    .decision-rejected {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(RAW_DATA_FILE)
    return df


@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Random Forest (Best)": MODELS_DIR / "random_forest.joblib",
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }
    for name, path in model_files.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_results():
    results = {}
    comp_path = BASE_DIR / "reports" / "results" / "model_comparison.csv"
    if comp_path.exists():
        results["comparison"] = pd.read_csv(comp_path)

    fi_path = BASE_DIR / "reports" / "results" / "feature_importance.csv"
    if fi_path.exists():
        results["fi"] = pd.read_csv(fi_path)

    tune_path = BASE_DIR / "reports" / "results" / "tuning_comparison.json"
    if tune_path.exists():
        with open(tune_path) as f:
            results["tuning"] = json.load(f)

    meta_path = MODEL_METADATA_PATH
    if meta_path.exists():
        with open(meta_path) as f:
            results["metadata"] = json.load(f)

    return results


df = load_data()
models = load_models()
results = load_results()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.markdown("## **Loan Approval ML**")
    st.markdown("*Second-Year College Machine Learning Project*")
    st.markdown("---")
    st.markdown("**Dataset**: Kaggle Loan Prediction")
    st.markdown(f"**Total Samples**: {len(df):,} applicants")
    st.markdown("**Target**: `Loan_Status` (Approved / Rejected)")
    st.markdown("---")
    st.markdown("### Quick Navigation")
    st.markdown("- 📌 Overview\n- 📊 Dataset\n- 📈 Exploratory Analysis\n- 🏆 Model Performance\n- 🔍 Feature Importance\n- 🚀 Live Predictor")
    st.markdown("---")
    st.caption("Built with Python, Scikit-Learn & Streamlit")

# Main Header
st.markdown('<div class="main-title">🏦 Loan Approval Classification & Risk Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An end-to-end Machine Learning system for evaluating applicant creditworthiness and predicting loan approval outcomes.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Overview",
    "📊 Dataset Explorer",
    "📈 Exploratory Analysis",
    "🏆 Model Performance",
    "🔍 Feature Importance",
    "🚀 Live Loan Predictor",
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Executive Summary")
    st.markdown("""
    This project provides a robust, machine-learning-driven framework to assist financial institutions in evaluating credit risk. 
    Using demographic, employment, and past repayment features, our algorithms learn decision boundaries that differentiate 
    creditworthy applicants from potential default risks.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", f"{len(df):,}")
    with col2:
        st.metric("Approval Rate", f"{(df['Loan_Status']=='Y').mean()*100:.1f}%")
    with col3:
        st.metric("Best Model Test ROC-AUC", "87.55%")
    with col4:
        st.metric("Primary Predictive Factor", "Credit History")

    st.markdown("---")
    st.markdown("### 🏗️ ML Architecture & Pipeline Workflow")
    st.markdown("""
    ```text
    [Raw Loan Dataset (614 Rows)]
                  │
                  ▼
    [Stratified 80/20 Train-Test Split] ── (Prevents Data Leakage)
                  │
                  ├── Feature Engineering: Total Income, EMI, Loan-to-Income, EMI-to-Income
                  ├── Missing Value Imputation: Median (Numerical) & Mode (Categorical)
                  ├── Categorical Encoding: One-Hot Encoding (sparse_output=False)
                  └── Feature Scaling: StandardScaler
                  │
                  ▼
    [Candidate Model Training & 5-Fold Stratified Cross-Validation]
       ├── Logistic Regression (Interpretable Baseline)
       ├── Random Forest Classifier (Non-linear Ensemble)
       └── XGBoost (Gradient Boosting)
                  │
                  ▼
    [Hyperparameter Optimization: GridSearchCV on Random Forest]
                  │
                  ▼
    [Holdout Test Set Evaluation & Model Serialization]
                  │
                  ▼
    [Streamlit Interactive Decision Dashboard & Live Inference]
    ```
    """)

# -----------------------------------------------------------------------------
# TAB 2: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔍 Dataset Explorer & Hygiene Inspection")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📐 Dataset Dimensions & Target Split")
        st.write(f"- **Rows**: {df.shape[0]}")
        st.write(f"- **Columns**: {df.shape[1]}")
        st.write(f"- **Duplicate Rows**: {df.duplicated().sum()}")
        st.write(f"- **Approved (`Y`)**: {(df['Loan_Status']=='Y').sum()} ({(df['Loan_Status']=='Y').mean()*100:.1f}%)")
        st.write(f"- **Rejected (`N`)**: {(df['Loan_Status']=='N').sum()} ({(df['Loan_Status']=='N').mean()*100:.1f}%)")

    with c2:
        st.markdown("#### 🩹 Missing Values Breakdown")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Missing Records": missing,
            "Missing %": (missing / len(df) * 100).round(2),
        })
        st.dataframe(missing_df[missing_df["Missing Records"] > 0], use_container_width=True)

    st.markdown("#### 📊 Summary Statistics for Numerical Attributes")
    st.dataframe(df.describe().round(2), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📈 Exploratory Data Analysis & Statistical Takeaways")
    fig_dir = BASE_DIR / "reports" / "figures"

    eda_choice = st.selectbox(
        "Select an Analytical Visualization to View:",
        [
            "1. Target Class Distribution",
            "2. Credit History vs Approval Rate",
            "3. Numerical Distributions & Skewness",
            "4. Total Income & Loan Amount vs Decision",
            "5. Categorical Factors Impact",
            "6. Feature Correlation Heatmap",
        ]
    )

    if eda_choice.startswith("1"):
        img = fig_dir / "01_target_distribution.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: Approximately 68.7% of all applications are approved, indicating a moderate class imbalance.
        **Why It Matters**: Standard accuracy can be deceptive; a naive model guessing all 'Approved' would achieve 68.7% accuracy. We must monitor Precision, Recall, F1-Score, and ROC-AUC.
        """)

    elif eda_choice.startswith("2"):
        img = fig_dir / "02_credit_history_vs_approval.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: Applicants meeting credit guidelines (1.0) have an approval rate of ~79.6%, whereas applicants with past defaults (0.0) drop to only ~8.0%.
        **Why It Matters**: Credit history is overwhelmingly the single most powerful predictor in the dataset.
        """)

    elif eda_choice.startswith("3"):
        img = fig_dir / "03_numerical_distributions.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: ApplicantIncome, CoapplicantIncome, and LoanAmount exhibit strong right-skewness with extreme high-income outliers.
        **Why It Matters**: Linear models like Logistic Regression benefit heavily from feature scaling (StandardScaler) and logarithmic transformation to prevent high-income outliers from skewing decision hyperplanes.
        """)

    elif eda_choice.startswith("4"):
        img = fig_dir / "04_income_vs_loan_approval.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: Counterintuitively, income distributions between approved and rejected applicants have significant overlap. High income alone does not guarantee approval if past credit history is compromised.
        """)

    elif eda_choice.startswith("5"):
        img = fig_dir / "05_categorical_vs_approval.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: Semiurban applicants exhibit the highest approval rate (~76.8%), compared to Rural (~61.5%). Graduates and married applicants also exhibit modestly higher approval rates.
        """)

    elif eda_choice.startswith("6"):
        img = fig_dir / "06_correlation_heatmap.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        st.info("""
        **Key Observation**: `LoanAmount` correlates strongly with `Total_Income` (r ≈ 0.62). `Credit_History` has the highest positive correlation with binary `Loan_Status` (r ≈ 0.54).
        """)

# -----------------------------------------------------------------------------
# TAB 4: MODEL PERFORMANCE & COMPARISON
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🏆 Comprehensive Model Evaluation & Benchmark")
    st.markdown("All performance numbers are derived from testing on an untouched holdout test set (123 applicants, 20% stratified split):")

    if "comparison" in results:
        comp_df = results["comparison"]
        st.dataframe(
            comp_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#D1FAE5"),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        cm_img = fig_dir / "07_confusion_matrices.png"
        if cm_img.exists():
            st.markdown("#### 🎯 Confusion Matrices")
            st.image(str(cm_img), use_container_width=True)

    with col2:
        roc_img = fig_dir / "08_roc_curves.png"
        if roc_img.exists():
            st.markdown("#### 📉 ROC Curves Comparison")
            st.image(str(roc_img), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚖️ The Precision vs Recall Trade-off in Credit Decisioning")
    st.markdown("""
    In financial lending:
    - **Precision (Approved Class)**: Answers *"Out of all applicants predicted to be approved, how many were actually creditworthy?"* 
      A high precision prevents the bank from issuing loans to high-risk borrowers who default.
    - **Recall (Approved Class)**: Answers *"Out of all truly creditworthy applicants, how many did we successfully identify?"* 
      A high recall ensures legitimate borrowers aren't wrongfully turned away.
    - **Final Verdict**: **Random Forest** provides the most balanced trade-off, achieving **89.16% Precision**, **87.06% Recall**, and the top **ROC-AUC of 87.55%**.
    """)

    st.markdown("---")
    st.markdown("### 🔧 Hyperparameter Tuning (GridSearchCV on Random Forest)")
    if "tuning" in results:
        t = results["tuning"]
        st.write(f"- **Best Parameters Selected**: `{t.get('best_params')}`")
        b_f1 = t.get("before_tuning", {}).get("f1", 0)
        a_f1 = t.get("after_tuning", {}).get("f1", 0)
        b_acc = t.get("before_tuning", {}).get("accuracy", 0)
        a_acc = t.get("after_tuning", {}).get("accuracy", 0)
        st.write(f"- **Accuracy Before Tuning**: `{b_acc*100:.2f}%` ➡️ **After Tuning**: `{a_acc*100:.2f}%`")
        st.write(f"- **F1-Score Before Tuning**: `{b_f1*100:.2f}%` ➡️ **After Tuning**: `{a_f1*100:.2f}%`")
        st.caption("Note: Hyperparameter pruning constrained max_depth to 4 to combat variance, resulting in virtually equivalent test generalization.")

# -----------------------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 🔍 Model Interpretability & Feature Significance")
    fi_img = fig_dir / "09_feature_importance.png"
    if fi_img.exists():
        st.image(str(fi_img), use_container_width=True)

    if "fi" in results:
        st.markdown("#### 📋 Feature Importance Rankings (Random Forest)")
        st.dataframe(results["fi"].head(10), use_container_width=True)

    st.markdown("""
    > [!NOTE]
    > **Interpretability Takeaway**:
    > 1. **Credit History**: Dominates the model's split decisions (>42% aggregate importance). Without clean repayment history, approval probability plummets.
    > 2. **Financial Capacity (Total Income & EMI)**: Ranked second, indicating that loan sizing relative to total household earnings is vital for determining affordability.
    > 3. **Location (Property Area)**: Semiurban residency provides a noticeable positive uplift over rural applicants.
    """)

# -----------------------------------------------------------------------------
# TAB 6: LIVE PREDICTOR
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("### 🚀 Real-Time Loan Approval Predictor")
    st.markdown("Enter applicant details below to evaluate approval likelihood using our trained machine learning models:")

    selected_model_name = st.selectbox("Select Model Architecture:", list(models.keys()))
    selected_pipeline = models[selected_model_name]

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Applicant Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        with col2:
            st.markdown("##### 💰 Financial Earnings ($/month)")
            applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=250)
            coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, max_value=50000, value=1500, step=250)
            property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

        with col3:
            st.markdown("##### 📄 Loan Request & Credit Record")
            loan_amount = st.number_input("Loan Amount Requested ($ in Thousands)", min_value=1, max_value=1000, value=150, step=5)
            loan_term = st.selectbox("Loan Term (Months)", [360, 180, 240, 300, 480, 120, 84, 60, 36, 12], index=0)
            credit_history = st.selectbox(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x: "Meets Guidelines (Good - 1.0)" if x == 1.0 else "Defaults / Delinquent (Bad - 0.0)",
            )

        submit_btn = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)

    if submit_btn:
        applicant_dict = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area,
        }

        applicant_df = pd.DataFrame([applicant_dict])

        # Inference
        pred = selected_pipeline.predict(applicant_df)[0]
        prob = (
            selected_pipeline.predict_proba(applicant_df)[0, 1]
            if hasattr(selected_pipeline, "predict_proba")
            else float(pred)
        )

        st.markdown("---")
        st.markdown("### 📝 Decision Verdict & Risk Analysis")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div class="decision-approved">
                    ✅ LOAN APPROVED<br>
                    <span style="font-size: 1.05rem; font-weight: normal;">Approval Probability: <b>{prob*100:.1f}%</b></span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-rejected">
                    ❌ LOAN REJECTED<br>
                    <span style="font-size: 1.05rem; font-weight: normal;">Approval Probability: <b>{prob*100:.1f}%</b></span>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Risk Assessment Tier")
            if prob >= 0.70:
                st.success(f"🟢 **Low Risk Tier** ({prob*100:.1f}% approval confidence)")
                st.write("Applicant has strong repayment likelihood based on credit guidelines and stable income-to-loan ratios.")
            elif prob >= 0.50:
                st.warning(f"🟡 **Moderate Risk Tier** ({prob*100:.1f}% approval confidence)")
                st.write("Marginal qualification. Additional collateral or secondary guarantor may be advisable.")
            else:
                st.error(f"🔴 **High Risk Tier** ({prob*100:.1f}% approval confidence)")
                st.write("Application presents significant default risk, primarily driven by credit history record or elevated debt obligation.")

        # Key applicant ratios
        tot_inc = applicant_income + coapplicant_income
        monthly_emi = (loan_amount * 1000) / loan_term if loan_term else 0
        dti = (monthly_emi / tot_inc * 100) if tot_inc > 0 else 0

        st.markdown("#### 💡 Applicant Financial Ratios")
        r1, r2, r3 = st.columns(3)
        r1.metric("Total Household Income", f"${tot_inc:,.0f} / mo")
        r2.metric("Estimated Monthly EMI", f"${monthly_emi:,.2f} / mo")
        r3.metric("Estimated Debt-to-Income (DTI)", f"{dti:.1f}%")
