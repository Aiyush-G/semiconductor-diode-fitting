import streamlit as st

st.set_page_config(page_title="Diode Fitting", page_icon="☀️", layout="wide")

st.title("Diode Fitting")
st.markdown(
    """
    Solar cell IV curve modelling and fitting — TacOSPV internship project.

    Use the sidebar to navigate:
    - **Single Diode** — Phases A-B: IV/JV curve plotting and fitting
    - **Tandem** — Phase C: 2-terminal tandem (two sub-cells in series,
      current-matched), 10-parameter modelling and fitting
    """
)
