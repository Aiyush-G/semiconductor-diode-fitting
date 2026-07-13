"""Accessible Streamlit page for exploring the single-diode model."""

import streamlit as st

from src.models.single_diode import DiodeParams, iv_curve, key_metrics
from src.models.temperature import TemperatureCoefficients, adjust_params_for_temperature
from ui.plotting import iv_curve_figure


REFERENCE_TEMP_K = 298.15
SATURATION_CURRENT_OPTIONS = [10 ** p for p in range(-14, -5)]


# Page metadata and opening copy are kept concise so keyboard and screen-reader
# users reach the controls quickly.
st.set_page_config(page_title="Single Diode", page_icon="☀️", layout="wide")
st.title("Single Diode Model - Phase A")

st.markdown(
    "Explore how equivalent-circuit parameters change a solar-cell IV curve "
    "under light and optional dark conditions."
)

# Keep inputs and outputs in separate columns so the interaction flow is
# predictable: set reference values first, then read the computed result.
col_controls, col_results = st.columns([1, 2], gap="large")

with col_controls:
    st.header("Reference parameters")
    st.caption("Circuit values at the 25 deg C reference condition.")

    i_ph = st.slider(
        "Photo-current I_ph (A)",
        0.0,
        10.0,
        8.0,
        step=0.1,
        help=(
            "Light-generated current in amps. Higher values raise the "
            "short-circuit current of the light IV curve."
        ),
    )
    i_0 = st.select_slider(
        "Saturation current I_0 (A)",
        options=SATURATION_CURRENT_OPTIONS,
        value=1e-10,
        format_func=lambda x: f"{x:.0e}",
        help=(
            "Reverse saturation current in amps. This controls the diode "
            "recombination current and strongly affects open-circuit voltage."
        ),
    )
    n = st.slider(
        "Ideality factor n",
        1.0,
        2.5,
        1.2,
        step=0.05,
        help=(
            "Dimensionless diode ideality factor. Values near 1 represent a "
            "more ideal diode; larger values indicate stronger recombination."
        ),
    )
    r_s = st.slider(
        "Series resistance R_s (Ohm)",
        0.0,
        2.0,
        0.05,
        step=0.01,
        help=(
            "Internal series resistance in ohms from contacts, bulk material, "
            "and wiring. Larger values reduce current at high voltage."
        ),
    )
    r_sh = st.slider(
        "Shunt resistance R_sh (Ohm)",
        10.0,
        5000.0,
        500.0,
        step=10.0,
        help=(
            "Leakage-path resistance in ohms. Higher values generally mean "
            "less leakage near short circuit."
        ),
    )

    st.header("Operating conditions")
    temp_c = st.slider(
        "Cell temperature (deg C)",
        -20,
        85,
        25,
        help=(
            "Cell temperature used to adjust the reference photocurrent and "
            "saturation current before the IV curve is calculated."
        ),
    )
    show_dark = st.checkbox(
        "Overlay dark IV curve",
        value=False,
        help=(
            "Show a second IV curve with photo-current set to zero while "
            "keeping the other adjusted diode parameters unchanged."
        ),
    )

# Convert the visible controls into model parameters before applying the
# temperature adjustment. The reference values remain anchored to 25 deg C.
ref_params = DiodeParams(
    i_ph=i_ph,
    i_0=i_0,
    n=n,
    r_s=r_s,
    r_sh=r_sh,
    temp_k=REFERENCE_TEMP_K,
)
target_temp_k = temp_c + 273.15

if abs(target_temp_k - REFERENCE_TEMP_K) > 0.01:
    params = adjust_params_for_temperature(
        ref_params,
        target_temp_k,
        TemperatureCoefficients(),
    )
else:
    params = ref_params

# Model evaluation is kept outside the rendering blocks so UI layout changes do
# not affect the physics path.
voltage, current = iv_curve(params)
metrics = key_metrics(voltage, current)

with col_results:
    st.header("Results")

    metric_isc, metric_voc, metric_pmax, metric_ff = st.columns(4)
    metric_isc.metric("Isc (A)", f"{metrics['isc']:.3f}")
    metric_voc.metric("Voc (V)", f"{metrics['voc']:.3f}")
    metric_pmax.metric("Pmax (W)", f"{metrics['pmax']:.3f}")
    metric_ff.metric("Fill factor", f"{metrics['fill_factor']:.3f}")

    fig = iv_curve_figure(voltage, current, metrics=metrics, title="Light IV Curve")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Light IV curve with the maximum-power point marked and generated "
        "power shown on the secondary axis."
    )

    if show_dark:
        v_dark, i_dark = iv_curve(params, dark=True)
        fig_dark = iv_curve_figure(v_dark, i_dark, title="Dark IV Curve")
        st.plotly_chart(fig_dark, width="stretch")
        st.caption(
            "Dark IV curve calculated with the photo-current set to zero."
        )
