import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import feature_label
from components.layout import render_page_header


def render(df: pd.DataFrame, models: dict, results: dict) -> None:
    render_page_header("What drives the model?", "Understand which applicant features contribute most to the model's predictions.")
    importance = results.get("fi", pd.DataFrame()).copy()
    if importance.empty:
        st.info("Feature-importance output is not available for the saved project model.")
        return
    importance["Display feature"] = importance["Feature"].map(feature_label)
    top = importance.nlargest(10, "Importance").sort_values("Importance")
    st.plotly_chart(px.bar(top, x="Importance", y="Display feature", orientation="h", title="Top 10 model feature importances", color_discrete_sequence=["#2563eb"]).update_xaxes(tickformat=".1%"), use_container_width=True)
    table = importance[["Display feature", "Importance"]].sort_values("Importance", ascending=False)
    st.dataframe(table.style.format({"Importance":"{:.2%}"}), use_container_width=True, hide_index=True)
    st.info("Feature importance describes how the trained model used available inputs. It does not establish that a feature causes an approval outcome.")
