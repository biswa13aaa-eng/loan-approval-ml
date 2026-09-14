"""Shared layout primitives for the LOANWISE AI application."""

from pathlib import Path
import streamlit as st


def load_css() -> None:
    """Loads the central editorial design system CSS."""
    css_path = Path(__file__).resolve().parent.parent / "styles" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    """Legacy compatibility helper."""
    pass


def render_page_header(title: str, subtitle: str = "") -> None:
    """Renders a standard section header."""
    st.markdown(
        f"""
        <div class="section-marker">
            <div class="section-marker-title">{title}</div>
            {f'<div class="section-marker-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_marker(number: str, title: str, subtitle: str = "") -> None:
    """Renders an editorial numbered section marker (e.g. 01 — OVERVIEW)."""
    st.markdown(
        f"""
        <div class="section-marker">
            <div class="section-marker-num">{number}</div>
            <div class="section-marker-title">{title}</div>
            {f'<div class="section-marker-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_navigation() -> None:
    """Renders a sleek top brand bar."""
    st.markdown(
        """
        <div class="top-nav-bar">
            <div class="top-nav-brand">LOANWISE AI</div>
            <div class="top-nav-tag">AI LOAN INTELLIGENCE · READY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


NAV_LABELS = [
    "01  Overview",
    "02  Dataset",
    "03  Exploration",
    "04  Models",
    "05  Features",
    "06  Predictor",
    "07  About",
]

LABEL_FROM_ALIAS = {
    "01  Overview": "01  Overview",
    "Overview": "01  Overview",
    "02  Dataset": "02  Dataset",
    "Dataset": "02  Dataset",
    "03  Exploration": "03  Exploration",
    "Exploration": "03  Exploration",
    "Exploratory Analysis": "03  Exploration",
    "EDA": "03  Exploration",
    "04  Models": "04  Models",
    "Models": "04  Models",
    "Model Performance": "04  Models",
    "05  Features": "05  Features",
    "Features": "05  Features",
    "Feature Importance": "05  Features",
    "06  Predictor": "06  Predictor",
    "Predictor": "06  Predictor",
    "Loan Predictor": "06  Predictor",
    "Eligibility": "06  Predictor",
    "07  About": "07  About",
    "About": "07  About",
    "About Project": "07  About",
}

VIEW_FROM_LABEL = {
    "01  Overview": "Overview",
    "02  Dataset": "Dataset",
    "03  Exploration": "Exploration",
    "04  Models": "Models",
    "05  Features": "Features",
    "06  Predictor": "Predictor",
    "07  About": "About",
}

PAGE_MAP = {**LABEL_FROM_ALIAS, **VIEW_FROM_LABEL}


def navigate_to(target: str) -> None:
    """Sets navigation state to target page across all navigation state holders."""
    canonical_label = LABEL_FROM_ALIAS.get(target, "01  Overview")
    st.session_state["active_page"] = canonical_label
    st.session_state["sidebar_nav_radio"] = canonical_label


def render_sidebar() -> str:
    """Renders the dark editorial sidebar navigation with monospace numbered items."""
    if "sidebar_nav_radio" not in st.session_state:
        st.session_state["sidebar_nav_radio"] = "01  Overview"

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">LOANWISE AI</div>
                <div class="sidebar-brand-sub">Smarter decisions. Powered by intelligence.</div>
            </div>
            <div class="sidebar-status-pill">
                <span class="sidebar-status-dot"></span>
                <span>AI LOAN INTELLIGENCE · READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio(
            "Navigation",
            NAV_LABELS,
            key="sidebar_nav_radio",
            label_visibility="collapsed",
        )

        st.session_state["active_page"] = selected

        st.markdown(
            """
            <div class="sidebar-footer">
                <strong>LOANWISE AI v2.0</strong><br>
                Production Benchmark<br>
                Stratified 80/20 Holdout Split<br>
                XGBoost · Random Forest · LogReg
            </div>
            """,
            unsafe_allow_html=True,
        )

        return VIEW_FROM_LABEL.get(selected, selected)


def render_editorial_footer() -> None:
    """Renders the closing editorial footer with call-to-action."""
    st.markdown(
        """
        <div class="editorial-footer">
            <div class="hero-eyebrow" style="justify-content: center;">07 — NEXT GENERATION LENDING</div>
            <div class="footer-cta-title">Ready to make a smarter decision?</div>
            <div class="footer-cta-sub">
                Evaluate individual borrower profiles with trained, holdout-calibrated machine learning models in seconds.
            </div>
            <div class="footer-disclaimer">
                LOANWISE AI · AI-POWERED LOAN INTELLIGENCE<br>
                Educational & research benchmark project. Predictions and feature contributions reflect historical patterns in training data and are not guaranteed causal lending decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

