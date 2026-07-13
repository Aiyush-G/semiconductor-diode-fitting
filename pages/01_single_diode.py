"""Accessible Streamlit page for exploring the single-diode model."""

import streamlit as st

from src.models.single_diode import DiodeParams, iv_curve, key_metrics
from src.models.temperature import TemperatureCoefficients, adjust_params_for_temperature
from ui.plotting import iv_curve_figure


REFERENCE_TEMP_K = 298.15
# Saturation current density J_0 options in A/cm^2 (PV Lighthouse convention).
SATURATION_CURRENT_OPTIONS = [10 ** p for p in range(-15, -5)]


# Page metadata and opening copy are kept concise so keyboard and screen-reader
# users reach the controls quickly.
st.set_page_config(page_title="Single Diode", layout="wide")
st.title("Single Diode Model")

st.markdown(
    "Explore how equivalent-circuit parameters change a solar-cell JV "
    "(current-density) curve under light and optional dark conditions."
)

# Keep inputs and outputs in separate columns so the interaction flow is
# predictable: set reference values first, then read the computed result.
col_controls, col_results = st.columns([1, 2], gap="large")

with col_controls:
    st.header("Reference parameters")
    st.caption(
        "Area-normalised circuit values at the 25 deg C reference condition "
        "(PV Lighthouse convention)."
    )

    j_ph_ma = st.slider(
        "Photo-current density J_ph (mA/cm²)",
        0.0,
        50.0,
        40.0,
        step=0.5,
        help=(
            "Light-generated current density in mA/cm². Higher values raise "
            "the short-circuit current density of the light JV curve."
        ),
    )
    j_0 = st.select_slider(
        "Saturation current density J_0 (A/cm²)",
        options=SATURATION_CURRENT_OPTIONS,
        value=1e-13,
        format_func=lambda x: f"{x:.0e}",
        help=(
            "Reverse saturation current density in A/cm². This controls the "
            "diode recombination current and strongly affects open-circuit "
            "voltage."
        ),
    )
    n = st.slider(
        "Ideality factor n",
        1.0,
        2.0,
        1.0,
        step=0.05,
        help=(
            "Dimensionless diode ideality factor. Values near 1 represent a "
            "more ideal diode; larger values indicate stronger recombination."
        ),
    )
    r_s = st.slider(
        "Series resistance R_s (Ω·cm²)",
        0.0,
        5.0,
        0.5,
        step=0.05,
        help=(
            "Area-normalised series resistance in Ω·cm² from contacts, bulk "
            "material, and wiring. Larger values reduce current at high voltage."
        ),
    )
    r_sh = st.slider(
        "Shunt resistance R_sh (Ω·cm²)",
        100.0,
        100000.0,
        1000.0,
        step=100.0,
        help=(
            "Area-normalised leakage-path resistance in Ω·cm². Higher values "
            "generally mean less leakage near short circuit."
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
# J_ph is entered in mA/cm² but the model works internally in A/cm².
ref_params = DiodeParams(
    j_ph=j_ph_ma * 1e-3,
    j_0=j_0,
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

    metric_jsc, metric_voc, metric_pmax, metric_ff, metric_eff = st.columns(5)
    metric_jsc.metric("Jsc (mA/cm²)", f"{metrics['jsc'] * 1e3:.2f}")
    metric_voc.metric("Voc (V)", f"{metrics['voc']:.3f}")
    metric_pmax.metric("Pmax (mW/cm²)", f"{metrics['pmax'] * 1e3:.2f}")
    metric_ff.metric("Fill factor", f"{metrics['fill_factor']:.3f}")
    metric_eff.metric("Efficiency (%)", f"{metrics['efficiency'] * 1e2:.2f}")

    fig = iv_curve_figure(voltage, current, metrics=metrics, title="Light JV Curve")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Light JV curve with the maximum-power point marked and generated "
        "power density shown on the secondary axis."
    )

    if show_dark:
        v_dark, i_dark = iv_curve(params, dark=True)
        fig_dark = iv_curve_figure(v_dark, i_dark, title="Dark JV Curve")
        st.plotly_chart(fig_dark, width="stretch")
        st.caption(
            "Dark JV curve calculated with the photo-current set to zero."
        )
