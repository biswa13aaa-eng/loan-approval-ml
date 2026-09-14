import pandas as pd
import streamlit as st

from components.data import feature_groups
from components.layout import render_page_header


def _default(series: pd.Series):
    clean = series.dropna()
    return clean.median() if pd.api.types.is_numeric_dtype(series) else (clean.mode().iloc[0] if not clean.empty else "")


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("Loan Approval Predictor", "Enter applicant information to estimate the likelihood of loan approval.")
    if df is None or not models:
        st.error("A dataset and at least one saved model are required for prediction.")
        return
    numerical, categorical = feature_groups(df)
    selected = st.selectbox("Model used", list(models))
    values = {}
    with st.form("loan_predictor"):
        st.markdown("#### Applicant information")
        columns = st.columns(2)
        for index, column in enumerate(categorical):
            options = df[column].dropna().unique().tolist()
            default = _default(df[column])
            with columns[index % 2]:
                values[column] = st.selectbox(column.replace("_", " "), options, index=options.index(default) if default in options else 0)
        st.markdown("#### Financial and loan information")
        columns = st.columns(2)
        for index, column in enumerate(numerical):
            default = float(_default(df[column]))
            with columns[index % 2]:
                values[column] = st.number_input(column.replace("_", " "), min_value=0.0, value=default, step=max(1.0, round(default / 20, 2)))
        submitted = st.form_submit_button("PREDICT LOAN APPROVAL", use_container_width=True)
    if not submitted:
        return
    if values.get("ApplicantIncome", 1) + values.get("CoapplicantIncome", 0) <= 0:
        st.warning("Unable to generate prediction. Household income must be greater than zero.")
        return
    try:
        applicant = pd.DataFrame([values])
        pipeline = models[selected]
        with st.spinner("Generating prediction..."):
            prediction = int(pipeline.predict(applicant)[0])
            probabilities = pipeline.predict_proba(applicant)[0] if hasattr(pipeline, "predict_proba") else None
        approved = prediction == 1
        st.success("LOAN LIKELY TO BE APPROVED" if approved else "LOAN LIKELY TO BE REJECTED")
        st.caption(f"Model used: {selected}")
        if probabilities is not None:
            rejected, approval = float(probabilities[0]), float(probabilities[1])
            a, b = st.columns(2); a.metric("Approved", f"{approval:.1%}"); b.metric("Rejected", f"{rejected:.1%}")
            risk = "Low" if approval >= .70 else "Moderate" if approval >= .50 else "High"
            st.info(f"Risk: {risk}. Thresholds: low ≥70%, moderate 50–69.9%, high <50% approval probability.")
        else:
            st.caption("This saved model does not expose prediction probabilities.")
        st.caption("The result reflects model behavior in the training data and is not a causal explanation or lending decision.")
    except Exception:
        st.error("Unable to generate prediction. Please check the entered information.")
