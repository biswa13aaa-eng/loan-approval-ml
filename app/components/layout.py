"""
Layout, header, and persistent sidebar components for LOANWISE AI.
"""

from pathlib import Path
import streamlit as st
import pandas as pd


def load_css():
    """
    Loads and injects the custom design system CSS into the Streamlit app.
    """
    css_path = Path(__file__).resolve().parent.parent / "styles" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_app_header(
    title: str = "LOANWISE AI",
    subtitle: str = "Loan Approval Prediction using Machine Learning",
    tagline: str = "Understand loan approval patterns. Compare machine learning models. Predict approval outcomes.",
):
    """
    Renders the unified top brand header.
    """
    html = f"""
    <div class="app-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
        <div class="tagline">“{tagline}”</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = ""):
    """
    Renders section headers on each page.
    """
    html = f"""
    <div class="section-header">
        <h2>{title}</h2>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar():
    """
    Renders the persistent branded sidebar and returns the selected navigation view.
    """
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-title">🏦 LOANWISE AI</div>
            <div class="sidebar-subtitle">Credit Risk Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            "🏠 Overview",
            "📊 Dataset",
            "🔎 Exploratory Analysis",
            "🤖 Model Performance",
            "🧠 Feature Importance",
            "💳 Loan Predictor",
            "ℹ️ About Project",
        ]

        selected_page = st.radio(
            "Navigation",
            pages,
            label_visibility="collapsed",
        )

        st.markdown("""
        <div class="sidebar-footer">
            <b>Machine Learning Project</b><br/>
            Second Year B.Tech • Academic Project<br/>
            <span style="color: #64748B;">Kaggle Loan Benchmark (614 rows)</span>
        </div>
        """, unsafe_allow_html=True)

        return selected_page
