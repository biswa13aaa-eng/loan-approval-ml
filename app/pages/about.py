import pandas as pd
import streamlit as st

from components.cards import render_takeaway
from components.data import display_name, feature_groups
from components.layout import render_section_marker


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_section_marker("07 — ABOUT LOANWISE AI", "The Architecture Behind Intelligent Lending", "Engineering an end-to-end decision intelligence platform on empirical financial benchmarks.")

    numerical, categorical = feature_groups(df) if df is not None else ([], [])
    metadata = results.get("metadata", {})

    # 1. The Idea / Objective
    render_section_marker("07.1 — THE VISION", "Objective & Rationale", "Modernizing retail credit assessment through calibrated machine learning.")
    st.markdown(
        """
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1.8rem; margin-bottom: 1.5rem; line-height: 1.7; color: var(--text-cream);">
            <strong>LoanWise AI</strong> is conceived as a decision-intelligence engine that shifts lending evaluation from opaque, manual heuristics toward empirical, reproducible probability scoring. By validating multi-paradigm classifiers on standardized historical lending benchmarks, the system provides both underwriting officers and applicants with instant transparency into approval likelihoods, key risk drivers, and holdout performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. The Data & Preprocessing
    render_section_marker("07.2 — THE DATASET", "Benchmark Data & Feature Pipeline", "Handling raw applicant signals without data leakage.")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.4rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-lime); text-transform: uppercase; margin-bottom: 0.4rem;">BENCHMARK REPOSITORY</div>
                <div style="font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-cream); margin-bottom: 0.5rem;">Kaggle Loan Approval Classification</div>
                <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.55;">
                    The dataset encompasses <strong>{len(df) if df is not None else 614}</strong> historical applicant files with <strong>{len(df.columns) if df is not None else 13}</strong> attributes. Target variable <code>Loan_Status</code> is distributed between 422 Approved (Y) and 192 Rejected (N) records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.4rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-olive); text-transform: uppercase; margin-bottom: 0.4rem;">TRANSFORMATION PIPELINE</div>
                <div style="font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-cream); margin-bottom: 0.5rem;">Leakage-Free Feature Engineering</div>
                <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.55;">
                    Numerical attributes (<code>{len(numerical)}</code>) are imputed via median strategy and scaled; categorical variables (<code>{len(categorical)}</code>) receive modal imputation followed by one-hot encoding, fitted strictly on training folds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. The Technology Stack
    render_section_marker("07.3 — TECHNOLOGY STACK", "Engineered with Open Source ML", "Core libraries and frameworks powering the modeling and interface layer.")
    tech_items = [
        ("Python 3.10+", "Core Language", "High-performance scientific computing and modeling."),
        ("Streamlit", "Frontend Engine", "Reactive, stateful decision dashboard without client JS."),
        ("Scikit-learn", "ML Architecture", "Pipelines, imputation, cross-validation, and metrics."),
        ("XGBoost", "Gradient Boosting", "High-efficiency decision tree ensemble modeling."),
        ("Pandas & NumPy", "Data Operations", "Vectorized dataset munging and quality assurance."),
        ("Plotly Express", "Interactive Charts", "Vector visualization conforming to the editorial design system."),
    ]
    tcols = st.columns(3)
    for idx, (tech_name, tech_role, tech_desc) in enumerate(tech_items):
        with tcols[idx % 3]:
            st.markdown(
                f"""
                <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem;">
                    <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--accent-lime); text-transform: uppercase;">{tech_role}</div>
                    <div style="font-family: var(--font-display); font-size: 1.15rem; font-weight: 700; color: var(--text-cream); margin: 0.2rem 0 0.4rem;">{tech_name}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.45;">{tech_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Limitations & Future Directions
    render_section_marker("07.4 — GOVERNANCE & ETHICS", "Academic Limitations & Roadmap", "Responsible AI considerations for automated financial scoring.")
    render_takeaway(
        "<strong>Academic Research Scope:</strong> This application serves as a benchmark and educational decision-intelligence tool. Model outputs reflect patterns in historical training data and must not be used as an unmonitored automated credit determination system.",
        "FAIR LENDING & STATISTICAL NOTICE",
    )

    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-top: 1.2rem;">
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-olive); text-transform: uppercase;">PLANNED ENHANCEMENT #1</div>
                <div style="color: var(--text-cream); font-weight: 600; font-size: 0.92rem; margin: 0.3rem 0;">Disparate Impact & Bias Audit</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Integrating fairness metrics (equalized odds, demographic parity) across demographic cohorts.</div>
            </div>
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-olive); text-transform: uppercase;">PLANNED ENHANCEMENT #2</div>
                <div style="color: var(--text-cream); font-weight: 600; font-size: 0.92rem; margin: 0.3rem 0;">Counterfactual Recourse Engine</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Generating minimum actionable parameter changes required for a rejected applicant to reach approval.</div>
            </div>
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-olive); text-transform: uppercase;">PLANNED ENHANCEMENT #3</div>
                <div style="color: var(--text-cream); font-weight: 600; font-size: 0.92rem; margin: 0.3rem 0;">Real-Time Concept Drift Monitoring</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Continuous population stability index (PSI) tracking to detect macro-economic credit shifts.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

