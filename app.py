import streamlit as st

st.set_page_config(page_title="Diode Fitting", page_icon="", layout="wide")

st.title("Diode Fitting")
st.markdown(
    """
    Solar cell IV curve modelling and fitting.

    Use the sidebar to navigate:
    - **Single Diode** - IV/JV curve plotting
    - **Tandem** 
    
    """
)
