"""
Modular editorial card and UI primitives for LOANWISE AI.
"""

import streamlit as st


def render_kpi_card(label: str, value: str, subtext: str = "", border_color: str = "#D6BE1F") -> None:
    """
    Renders an editorial high-contrast stat card.
    """
    st.markdown(
        f"""
        <div class="stat-card" style="border-top: 2px solid {border_color};">
            <div class="stat-card-num">{label}</div>
            <div class="stat-card-val">{value}</div>
            <div class="stat-card-label">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_takeaway(text: str, title: str = "KEY INSIGHT") -> None:
    """
    Renders an editorial takeaway insight box in dark card style.
    """
    st.markdown(
        f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-left: 3px solid var(--accent-lime); border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-lime); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem;">{title}</div>
            <div style="color: var(--text-cream); font-size: 0.95rem; line-height: 1.5;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_editorial_hero(
    eyebrow: str = "AI-POWERED LOAN INTELLIGENCE",
    title: str = "Make smarter<br>loan decisions.",
    description: str = "LoanWise AI analyzes applicant profiles and historical lending patterns to estimate loan approval outcomes using calibrated machine learning ensembles.",
    status_text: str = "AI MODEL READY",
) -> None:
    """
    Renders the large editorial hero section.
    """
    st.markdown(
        f"""
        <div class="editorial-hero">
            <div class="hero-eyebrow">
                <span class="sidebar-status-dot"></span>
                <span>{eyebrow}</span>
                <span style="color: var(--text-dim);">·</span>
                <span style="color: var(--accent-lime);">{status_text}</span>
            </div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_diagram() -> None:
    """
    Renders the abstract AI system architecture diagram:
    APPLICANT -> DATA -> PREPROCESSING -> ML MODELS -> PREDICTION
    """
    nodes = [
        ("01", "APPLICANT", "Demographic & household submission"),
        ("02", "DATA", "13 Normalized benchmark attributes"),
        ("03", "PREPROCESSING", "Leakage-free imputation & scaling"),
        ("04", "ML MODELS", "XGBoost, Random Forest & LogReg"),
        ("05", "PREDICTION", "Calibrated probability & risk tier"),
    ]
    html = ['<div class="system-diagram">']
    for i, (num, title, detail) in enumerate(nodes):
        html.append(
            f"""
            <div class="diagram-node">
                <div class="diagram-node-tag">{num} — SYSTEM NODE</div>
                <div class="diagram-node-title">{title}</div>
                <div class="diagram-node-detail">{detail}</div>
            </div>
            """
        )
        if i < len(nodes) - 1:
            html.append('<div class="diagram-arrow">→</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_story_card(number: str, eyebrow: str, headline: str, copy: str) -> None:
    """
    Renders an editorial story card for Problem / Approach / Result.
    """
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-eyebrow">{number} — {eyebrow}</div>
            <div class="story-headline">{headline}</div>
            <div class="story-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_process_step(number: str, title: str, description: str) -> None:
    """
    Renders a 4-step workflow process card.
    """
    st.markdown(
        f"""
        <div class="process-step-card">
            <div class="step-badge">STEP {number}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(approved: bool, prob: float, risk_tier: str, model_name: str) -> None:
    """
    Renders the flagship AI Predictor result card with probability gauge and risk tier.
    """
    card_class = "approved" if approved else "rejected"
    headline = "LIKELY TO BE APPROVED" if approved else "LIKELY TO BE REJECTED"
    headline_color = "var(--status-success)" if approved else "var(--status-danger)"
    approval_pct = prob * 100
    rejection_pct = (1.0 - prob) * 100
    risk_color = (
        "var(--status-success)" if risk_tier == "Low" else ("var(--accent-lime)" if risk_tier == "Moderate" else "var(--status-danger)")
    )

    card_html = (
        f'<div class="pred-result-card {card_class}">'
        f'<div class="pred-header-tag">AI ASSESSMENT REPORT · {model_name.upper()}</div>'
        f'<div class="pred-decision-headline" style="color: {headline_color};">{headline}</div>'
        f'<div class="pred-prob-large">{approval_pct:.1f}%</div>'
        f'<div class="pred-prob-label">ESTIMATED APPROVAL PROBABILITY</div>'
        f'<div class="prob-meter-track">'
        f'<div class="prob-meter-fill {card_class}" style="width: {approval_pct:.2f}%;"></div>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-top: 1.5rem;">'
        f'<div style="background: rgba(0,0,0,0.4); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem 1rem;">'
        f'<div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Approval Probability</div>'
        f'<div style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; color: var(--text-cream); margin-top: 0.2rem;">{approval_pct:.2f}%</div>'
        f'</div>'
        f'<div style="background: rgba(0,0,0,0.4); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem 1rem;">'
        f'<div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Rejection Probability</div>'
        f'<div style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; color: var(--text-cream); margin-top: 0.2rem;">{rejection_pct:.2f}%</div>'
        f'</div>'
        f'<div style="background: rgba(0,0,0,0.4); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem 1rem;">'
        f'<div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Assessed Risk Tier</div>'
        f'<div style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; color: {risk_color}; margin-top: 0.2rem;">{risk_tier.upper()} RISK</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

