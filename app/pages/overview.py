import pandas as pd
import streamlit as st

from components.cards import (
    render_editorial_hero,
    render_kpi_card,
    render_process_step,
    render_story_card,
    render_system_diagram,
    render_takeaway,
)
from components.data import best_row, display_name, input_columns
from components.layout import navigate_to, render_section_marker


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    if df is None:
        st.error("The dataset is unavailable. Check data/raw/loan_data.csv.")
        return

    # 1. Editorial Hero Section
    render_editorial_hero(
        eyebrow="AI-POWERED LOAN INTELLIGENCE",
        title='LOAN <span class="accent-text">INTELLIGENCE</span>,<br>BUILT FOR BETTER<br>DECISIONS.',
        description="LoanWise AI analyzes applicant information and historical lending patterns to estimate loan approval outcomes using machine learning.",
        status_text="AI MODEL READY",
    )

    # Interactive CTA Row
    cta_col1, cta_col2, cta_col_space = st.columns([1.4, 1.4, 3.2])
    with cta_col1:
        st.button("CHECK ELIGIBILITY →", key="hero_cta_predict", on_click=navigate_to, args=("06  Predictor",), type="primary", use_container_width=True)
    with cta_col2:
        st.button("EXPLORE THE DATA", key="hero_cta_dataset", on_click=navigate_to, args=("02  Dataset",), use_container_width=True)

    # 2. Dynamic Real Statistics Bar
    n_rows = len(df)
    n_features = len(df.columns)
    n_models = max(3, len(models))
    approval_rate = (df["Loan_Status"].astype(str).str.strip().str.upper() == "Y").mean() if "Loan_Status" in df else 0.69

    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("01 · DATASET VOLUME", f"{n_rows:,}", "APPLICATIONS ANALYZED", border_color="#D6BE1F")
    with c2:
        render_kpi_card("02 · SCHEMA BREADTH", f"{n_features}", "INPUT FEATURES", border_color="#B8B56A")
    with c3:
        render_kpi_card("03 · MODEL BENCHMARK", f"{n_models}", "EVALUATED ALGORITHMS", border_color="#D6BE1F")
    with c4:
        render_kpi_card("04 · BASELINE APPROVAL", f"{approval_rate:.0%}", "HISTORICAL APPROVAL RATE", border_color="#B8B56A")

    # 3. Abstract AI / Data Visual (System Architecture Flow)
    render_section_marker("01 — ARCHITECTURE", "HOW LOANWISE THINKS", "End-to-end transformation from raw applicant submission to risk-stratified decision.")
    render_system_diagram()

    # 4. Product Story (Editorial 3-Column Narrative)
    render_section_marker("02 — PRODUCT STORY", "WHY INTELLIGENT UNDERWRITING?", "Bridging the gap between historic lending records and instant, objective credit intelligence.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        render_story_card(
            "01",
            "THE PROBLEM",
            "Loan decisions are data decisions.",
            "Traditional underwriting relies on manual audits and rigid thresholds that slow down processing and miss non-linear financial patterns across applicant portfolios.",
        )
    with sc2:
        render_story_card(
            "02",
            "THE APPROACH",
            "Turn applicant information into actionable intelligence.",
            "LoanWise AI cleanses, imputes, and standardizes demographic, income, and debt-to-loan ratios, passing them into an ensemble of supervised learning classifiers.",
        )
    with sc3:
        render_story_card(
            "03",
            "THE RESULT",
            "From raw data to an explainable prediction.",
            "Instant calibrated probability scores and transparent risk tiers allow credit officers and applicants to understand the empirical basis behind every outcome.",
        )

    # 5. How LoanWise Works (4-Step Process)
    render_section_marker("03 — WORKFLOW", "FROM APPLICATION TO DECISION", "A transparent four-phase pipeline powering every prediction.")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_process_step(
            "01",
            "01 PROFILE",
            "Captures core applicant information: income, co-applicant contribution, loan term, credit history, and property location.",
        )
    with p2:
        render_process_step(
            "02",
            "02 FINANCIALS",
            "Applies median numerical imputation, modal categorical imputation, and one-hot encoding without lookahead bias.",
        )
    with p3:
        render_process_step(
            "03",
            "03 CREDIT",
            "Scores the applicant against trained pipelines (Logistic Regression, Random Forest, and XGBoost) tuned via cross-validation.",
        )
    with p4:
        render_process_step(
            "04",
            "04 AI DECISION",
            "Classifies the probability into calibrated tiers (Low, Moderate, High) with top contributing feature explanations.",
        )

    # 6. Performance Snapshot
    comparison = results.get("comparison", pd.DataFrame())
    top = best_row(comparison, "F1")
    if top is not None:
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        render_takeaway(
            f"Production pipeline benchmark selected <strong>{display_name(top['Model'])}</strong> achieving an F1 score of <strong>{top['F1']:.2%}</strong> and Accuracy of <strong>{top['Accuracy']:.2%}</strong> on unseen holdout data.",
            "MODEL BENCHMARK SNAPSHOT",
        )

    # 7. Bottom Editorial Call to Action
    st.markdown(
        """
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px; padding: 2.5rem; margin-top: 3rem; text-align: center;">
            <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--accent-lime); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.6rem;">READY TO TEST YOUR ELIGIBILITY?</div>
            <div style="font-family: var(--font-display); font-size: 2.2rem; font-weight: 800; color: var(--text-cream); margin-bottom: 0.8rem;">Evaluate Your Loan Profile in Seconds</div>
            <div style="font-size: 0.95rem; color: var(--text-muted); max-width: 540px; margin: 0 auto 1.8rem;">
                Enter your demographic and financial parameters to generate instant machine learning approval assessments across three calibrated models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_cta_btn_l, col_cta_btn, col_cta_btn_r = st.columns([1.5, 2, 1.5])
    with col_cta_btn:
        st.button("ANALYZE MY APPLICATION →", key="bottom_cta_predict", on_click=navigate_to, args=("06  Predictor",), type="primary", use_container_width=True)

