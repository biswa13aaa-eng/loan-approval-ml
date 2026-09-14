"""Small reusable display components."""

import streamlit as st


def render_kpi_card(label: str, value: str, subtext: str = "", border_color: str = "#2563eb") -> None:
    st.markdown(f'<div class="kpi-card" style="border-left:3px solid {border_color}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-subtext">{subtext}</div></div>', unsafe_allow_html=True)


def render_takeaway(text: str, title: str = "Insight") -> None:
    st.markdown(f'<div class="takeaway-box"><strong>{title}:</strong> {text}</div>', unsafe_allow_html=True)
