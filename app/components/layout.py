"""Shared layout primitives for the LOANWISE AI application."""

from pathlib import Path
import streamlit as st


def load_css() -> None:
    css_path = Path(__file__).resolve().parent.parent / "styles" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    st.markdown("""<div class="app-header"><p class="eyebrow">LOANWISE AI</p><h1>Loan approval intelligence</h1><p>Loan Approval Prediction using Machine Learning</p><span>Understand loan approval patterns. Compare machine learning models. Predict approval outcomes.</span></div>""", unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-header"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><div>LOANWISE AI</div><span>Loan Approval Intelligence</span></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["Overview", "Dataset", "Exploratory Analysis", "Model Performance", "Feature Importance", "Loan Predictor", "About Project"], label_visibility="collapsed")
        st.markdown('<div class="sidebar-footer"><b>Machine Learning Project</b><br>Second Year</div>', unsafe_allow_html=True)
        return page
