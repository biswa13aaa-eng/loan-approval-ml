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


def render_sidebar() -> str:
    """Renders the dark editorial sidebar navigation synchronized with session_state."""
    pages = [
        "Overview",
        "Dataset",
        "Exploratory Analysis",
        "Model Performance",
        "Feature Importance",
        "Loan Predictor",
        "About Project",
    ]

    # Initialize active page in session state if not set
    if "active_page" not in st.session_state or st.session_state["active_page"] not in pages:
        st.session_state["active_page"] = "Overview"

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">LOANWISE AI</div>
                <div class="sidebar-brand-sub">Smarter decisions. Powered by intelligence.</div>
            </div>
            <div class="sidebar-status-pill">
                <span class="sidebar-status-dot"></span>
                <span>AI MODEL READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_idx = pages.index(st.session_state["active_page"])
        selected = st.radio(
            "Navigation",
            pages,
            index=current_idx,
            label_visibility="collapsed",
            key="sidebar_nav_radio",
        )

        if selected != st.session_state["active_page"]:
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

        return st.session_state["active_page"]


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

