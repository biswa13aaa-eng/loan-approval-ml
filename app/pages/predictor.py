import traceback
import pandas as pd
import streamlit as st

from components.cards import render_prediction_result, render_takeaway
from components.data import display_name, feature_groups
from components.layout import render_section_marker

TRAINING_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


def _default(series: pd.Series):
    clean = series.dropna()
    return clean.median() if pd.api.types.is_numeric_dtype(series) else (clean.mode().iloc[0] if not clean.empty else "")


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker(
        "06 — ELIGIBILITY ENGINE",
        "LET THE MODEL ANALYZE YOUR PROFILE.",
        "Enter applicant profile, financial capacity, and credit details to generate an instant, calibrated machine learning approval assessment.",
    )

    if df is None or not models:
        st.error("A dataset and at least one saved model are required for prediction.")
        return

    # Determine recommended model for default selection
    metadata = results.get("metadata", {})
    best_name = metadata.get("best_model_name", "Random_Forest")
    model_list = list(models.keys())
    default_idx = 0
    for i, m in enumerate(model_list):
        if best_name and (best_name.lower() in m.lower().replace(" ", "_") or m.lower() in best_name.lower()):
            default_idx = i
            break

    # Model architecture selector
    model_col1, model_col2 = st.columns([1.5, 2.5])
    with model_col1:
        selected_model = st.selectbox(
            "Scoring Model Architecture",
            model_list,
            index=default_idx,
            key="scoring_model_selector",
        )
    with model_col2:
        st.markdown(
            f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1.5rem;">
                <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-lime); text-transform: uppercase;">ACTIVE INFERENCE ENGINE:</span>
                <span style="font-size: 0.88rem; color: var(--text-cream); margin-left: 0.5rem; font-weight: 600;">{selected_model} Pipeline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    values = {}
    with st.form("loan_predictor_form"):
        # SECTION 01 — YOUR PROFILE
        render_section_marker("SECTION 01", "Your Profile", "Demographic, household, and educational background.")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender_opts = [str(x) for x in df["Gender"].dropna().unique()] if "Gender" in df else ["Male", "Female"]
            values["Gender"] = st.selectbox("Gender", gender_opts, index=0, key="pred_gender")

            married_opts = [str(x) for x in df["Married"].dropna().unique()] if "Married" in df else ["Yes", "No"]
            values["Married"] = st.selectbox("Married", married_opts, index=0, key="pred_married")

        with c2:
            dep_opts = ["0", "1", "2", "3+"]
            values["Dependents"] = st.selectbox("Dependents", dep_opts, index=0, key="pred_dependents")

            edu_opts = [str(x) for x in df["Education"].dropna().unique()] if "Education" in df else ["Graduate", "Not Graduate"]
            values["Education"] = st.selectbox("Education", edu_opts, index=0, key="pred_education")

        with c3:
            emp_opts = [str(x) for x in df["Self_Employed"].dropna().unique()] if "Self_Employed" in df else ["No", "Yes"]
            values["Self_Employed"] = st.selectbox("Self Employed", emp_opts, index=0, key="pred_self_employed")

        # SECTION 02 — YOUR FINANCES
        render_section_marker("SECTION 02", "Your Finances", "Monthly incomes, requested loan principal, and amortization horizon.")
        f1, f2 = st.columns(2)
        with f1:
            def_inc = float(df["ApplicantIncome"].median()) if "ApplicantIncome" in df else 5000.0
            values["ApplicantIncome"] = st.number_input(
                "Applicant Monthly Income ($)",
                min_value=0.0,
                value=def_inc,
                step=100.0,
                key="pred_applicant_income",
            )

            def_coinc = float(df["CoapplicantIncome"].median()) if "CoapplicantIncome" in df else 0.0
            values["CoapplicantIncome"] = st.number_input(
                "Co-applicant Monthly Income ($)",
                min_value=0.0,
                value=def_coinc,
                step=100.0,
                key="pred_coapplicant_income",
            )

        with f2:
            def_amt = float(df["LoanAmount"].median()) if "LoanAmount" in df else 128.0
            values["LoanAmount"] = st.number_input(
                "Loan Amount ($ in thousands, e.g. 150 = $150,000)",
                min_value=1.0,
                value=def_amt,
                step=5.0,
                key="pred_loan_amount",
            )

            def_term = float(df["Loan_Amount_Term"].median()) if "Loan_Amount_Term" in df else 360.0
            values["Loan_Amount_Term"] = st.number_input(
                "Loan Term (in months, e.g. 360 = 30 years)",
                min_value=12.0,
                value=def_term,
                step=12.0,
                key="pred_loan_term",
            )

        # SECTION 03 — CREDIT & PROPERTY
        render_section_marker("SECTION 03", "Credit & Property Context", "Credit guideline history compliance and geographic property area.")
        cp1, cp2 = st.columns(2)
        with cp1:
            ch_options = [1.0, 0.0]
            values["Credit_History"] = st.selectbox(
                "Credit History Compliance",
                ch_options,
                format_func=lambda x: "Meets Credit Guidelines (1.0)" if x == 1.0 else "Does Not Meet Guidelines (0.0)",
                index=0,
                key="pred_credit_history",
            )

        with cp2:
            prop_opts = [str(x) for x in df["Property_Area"].dropna().unique()] if "Property_Area" in df else ["Urban", "Semiurban", "Rural"]
            values["Property_Area"] = st.selectbox("Property Area", prop_opts, index=0, key="pred_property_area")

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("ANALYZE MY ELIGIBILITY →", type="primary", use_container_width=True)

    # Clean applicant data builder
    def build_applicant_df(raw_inputs: dict) -> pd.DataFrame:
        clean = {
            "Gender": str(raw_inputs["Gender"]).strip(),
            "Married": str(raw_inputs["Married"]).strip(),
            "Dependents": str(raw_inputs["Dependents"]).strip(),
            "Education": str(raw_inputs["Education"]).strip(),
            "Self_Employed": str(raw_inputs["Self_Employed"]).strip(),
            "ApplicantIncome": float(raw_inputs["ApplicantIncome"]),
            "CoapplicantIncome": float(raw_inputs["CoapplicantIncome"]),
            "LoanAmount": float(raw_inputs["LoanAmount"]),
            "Loan_Amount_Term": float(raw_inputs["Loan_Amount_Term"]),
            "Credit_History": float(raw_inputs["Credit_History"]),
            "Property_Area": str(raw_inputs["Property_Area"]).strip(),
        }
        return pd.DataFrame([clean])[TRAINING_COLS]

    # Run inference helper
    def run_inference(applicant_df: pd.DataFrame, model_name: str) -> dict:
        pipeline = models[model_name]
        prediction = int(pipeline.predict(applicant_df)[0])
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(applicant_df)[0]
            rejected_prob = float(probs[0])
            approval_prob = float(probs[1])
        else:
            approval_prob = 1.0 if prediction == 1 else 0.0
            rejected_prob = 0.0 if prediction == 1 else 1.0

        approved = (prediction == 1)
        if approval_prob >= 0.70:
            risk_tier = "Low"
        elif approval_prob >= 0.50:
            risk_tier = "Moderate"
        else:
            risk_tier = "High"

        return {
            "approved": approved,
            "approval_prob": approval_prob,
            "rejected_prob": rejected_prob,
            "risk_tier": risk_tier,
            "model_name": model_name,
        }

    # Handle Form Submission
    if submitted:
        total_income = float(values.get("ApplicantIncome", 0)) + float(values.get("CoapplicantIncome", 0))
        if total_income <= 0:
            st.warning("Household income (Applicant + Co-applicant) must be greater than zero.")
            return

        if float(values.get("LoanAmount", 0)) <= 0:
            st.warning("Requested loan amount must be greater than zero.")
            return

        try:
            applicant_df = build_applicant_df(values)
            result = run_inference(applicant_df, selected_model)
            st.session_state["latest_applicant_inputs"] = values
            st.session_state["latest_prediction_result"] = result
        except Exception as exc:
            st.error(f"Prediction Pipeline Error: {exc}")
            with st.expander("Diagnostic Error Details", expanded=True):
                st.code(traceback.format_exc(), language="python")
            return

    # If not submitted in this rerun, check if user changed model selector with an existing submission
    elif "latest_applicant_inputs" in st.session_state and "latest_prediction_result" in st.session_state:
        prev_result = st.session_state["latest_prediction_result"]
        if prev_result.get("model_name") != selected_model:
            try:
                applicant_df = build_applicant_df(st.session_state["latest_applicant_inputs"])
                new_result = run_inference(applicant_df, selected_model)
                st.session_state["latest_prediction_result"] = new_result
            except Exception as exc:
                st.error(f"Prediction Update Error: {exc}")
                with st.expander("Diagnostic Error Details", expanded=True):
                    st.code(traceback.format_exc(), language="python")
                return

    # If there is a prediction result available, render it
    if "latest_prediction_result" in st.session_state:
        res = st.session_state["latest_prediction_result"]
        render_prediction_result(
            approved=res["approved"],
            prob=res["approval_prob"],
            risk_tier=res["risk_tier"],
            model_name=res["model_name"],
        )

        with st.expander("How should I interpret this AI assessment?", expanded=True):
            st.markdown(
                f"""
                - **Approval Probability ({res['approval_prob']:.2%}):** Represents the empirical likelihood of loan approval predicted by the **{res['model_name']}** model pipeline, evaluated on historic Kaggle lending records.
                - **Assessed Risk Tier ({res['risk_tier']} Risk):**
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
