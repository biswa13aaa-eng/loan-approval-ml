import pandas as pd
import streamlit as st

from components.cards import render_prediction_result, render_takeaway
from components.data import best_row, display_name, feature_groups
from components.layout import render_section_marker


def _default(series: pd.Series):
    clean = series.dropna()
    return clean.median() if pd.api.types.is_numeric_dtype(series) else (clean.mode().iloc[0] if not clean.empty else "")


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("06 — ELIGIBILITY ENGINE", "LET THE MODEL ANALYZE YOUR PROFILE.", "Enter applicant profile, financial capacity, and credit details to generate an instant, calibrated machine learning approval assessment.")

    if df is None or not models:
        st.error("A dataset and at least one saved model are required for prediction.")
        return

    numerical, categorical = feature_groups(df)

    # Determine recommended model for default selection
    metadata = results.get("metadata", {})
    best_name = metadata.get("best_model_name", "")
    model_list = list(models.keys())
    default_idx = 0
    for i, m in enumerate(model_list):
        if best_name and (best_name.lower() in m.lower() or m.lower() in best_name.lower()):
            default_idx = i
            break

    model_col1, model_col2 = st.columns([1.5, 2.5])
    with model_col1:
        selected = st.selectbox("Scoring Model Architecture", model_list, index=default_idx)
    with model_col2:
        st.markdown(
            f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1.5rem;">
                <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-lime); text-transform: uppercase;">ACTIVE INFERENCE ENGINE:</span>
                <span style="font-size: 0.88rem; color: var(--text-cream); margin-left: 0.5rem; font-weight: 600;">{selected} Pipeline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    values = {}
    with st.form("loan_predictor_form"):
        # SECTION 01 — YOUR PROFILE
        render_section_marker("SECTION 01", "Your Profile", "Demographic, household, and educational background.")
        c1, c2, c3 = st.columns(3)
        profile_cols = [col for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed"] if col in df.columns]
        for index, column in enumerate(profile_cols):
            options = df[column].dropna().unique().tolist()
            default = _default(df[column])
            target_col = c1 if index % 3 == 0 else (c2 if index % 3 == 1 else c3)
            with target_col:
                values[column] = st.selectbox(
                    column.replace("_", " "),
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"pred_prof_{column}",
                )

        # SECTION 02 — YOUR FINANCES
        render_section_marker("SECTION 02", "Your Finances", "Monthly incomes, requested loan principal, and amortization horizon.")
        f1, f2 = st.columns(2)
        finance_cols = [col for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"] if col in df.columns]
        for index, column in enumerate(finance_cols):
            default = float(_default(df[column]))
            target_col = f1 if index % 2 == 0 else f2
            with target_col:
                step_val = max(1.0, round(default / 20, 2)) if default > 0 else 100.0
                values[column] = st.number_input(
                    column.replace("_", " "),
                    min_value=0.0,
                    value=default,
                    step=step_val,
                    key=f"pred_fin_{column}",
                )

        # SECTION 03 — CREDIT & PROPERTY
        render_section_marker("SECTION 03", "Credit & Property Context", "Credit guideline history compliance and geographic property area.")
        cp1, cp2 = st.columns(2)
        credit_property_cols = [col for col in ["Credit_History", "Property_Area"] if col in df.columns]
        for index, column in enumerate(credit_property_cols):
            options = df[column].dropna().unique().tolist()
            default = _default(df[column])
            target_col = cp1 if index % 2 == 0 else cp2
            with target_col:
                if column == "Credit_History":
                    # Clean presentation of credit history options
                    ch_options = [1.0, 0.0] if set([1.0, 0.0]).issubset(set(options)) else options
                    values[column] = st.selectbox(
                        "Credit History (1.0 = Meets Guidelines, 0.0 = Does Not Meet)",
                        ch_options,
                        index=0 if 1.0 in ch_options else 0,
                        key=f"pred_ch_{column}",
                    )
                else:
                    values[column] = st.selectbox(
                        column.replace("_", " "),
                        options,
                        index=options.index(default) if default in options else 0,
                        key=f"pred_pa_{column}",
                    )

        # Any remaining columns from the dataset schema
        remaining = [c for c in categorical + numerical if c not in values and c not in ["Loan_Status", "Loan_ID"]]
        for rem in remaining:
            default = _default(df[rem])
            values[rem] = float(default) if pd.api.types.is_numeric_dtype(df[rem]) else str(default)

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("ANALYZE MY ELIGIBILITY →", type="primary", use_container_width=True)

    if not submitted:
        return

    # Input Validation: zero household income guard
    household_income = values.get("ApplicantIncome", 0) + values.get("CoapplicantIncome", 0)
    if household_income <= 0:
        st.warning("Unable to generate prediction. Household income (Applicant + Co-applicant) must be greater than zero.")
        return

    try:
        applicant = pd.DataFrame([values])
        pipeline = models[selected]
        with st.spinner("Evaluating profile against trained model pipeline..."):
            prediction = int(pipeline.predict(applicant)[0])
            probabilities = pipeline.predict_proba(applicant)[0] if hasattr(pipeline, "predict_proba") else None

        approved = (prediction == 1)

        if probabilities is not None:
            rejected_prob = float(probabilities[0])
            approval_prob = float(probabilities[1])
            risk = "Low" if approval_prob >= 0.70 else ("Moderate" if approval_prob >= 0.50 else "High")
        else:
            approval_prob = 1.0 if approved else 0.0
            rejected_prob = 0.0 if approved else 1.0
            risk = "Low" if approved else "High"

        # 1. Render Flagship AI Result Card
        render_prediction_result(approved, approval_prob, risk, selected)

        # 2. How to interpret this assessment
        with st.expander("How should I interpret this AI assessment?", expanded=True):
            st.markdown(
                f"""
                - **Approval Probability ({approval_prob:.2%}):** Represents the empirical likelihood of loan approval predicted by the **{selected}** model, trained on historic Kaggle lending benchmark records.
                - **Assessed Risk Tier ({risk} Risk):**
                  - **Low Risk (≥ 70%):** Strong alignment with historical approval profiles, typically characterized by compliant credit history and viable debt-to-income balance.
                  - **Moderate Risk (50% – 69.9%):** Marginal profile where certain counter-signals (e.g., higher requested loan amount or single-income dependency) increase underwriting sensitivity.
                  - **High Risk (< 50%):** Profile exhibits significant risk indicators (e.g. non-compliant credit history or insufficient income relative to loan size).
                - **Decision Transparency:** Credit history accounts for approximately ~48% of feature importance in tree-based benchmarks. Maintaining an on-time credit record is the single most influential variable.
                """
            )

        render_takeaway(
            "This prediction is an automated statistical estimate generated for academic and benchmarking purposes. It does not constitute a legally binding credit decision or clinical lending commitment.",
            "MODEL GOVERNANCE & ETHICS NOTICE",
        )

    except Exception as exc:
        st.error(f"Unable to generate prediction. Please verify the entered parameters: {exc}")

