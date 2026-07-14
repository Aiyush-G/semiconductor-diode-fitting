"""Accessible Streamlit page for exploring the single-diode model."""

import math

import streamlit as st

from src.models.single_diode import (
    DiodeParams,
    iv_curve,
    key_metrics,
    local_ideality_factor,
)
from src.models.temperature import TemperatureCoefficients, adjust_params_for_temperature
from ui.inputs import slider_with_number
from ui.plotting import ideality_factor_figure, iv_curve_figure, log_jv_figure


REFERENCE_TEMP_K = 298.15
# Saturation current density J_0 options in A/cm^2 (PV Lighthouse convention).
SATURATION_CURRENT_OPTIONS = [10 ** p for p in range(-15, -5)]
# Bounds for the free-form J_0 number box (span the decade selector, with a
# little headroom on either side).
J0_MIN, J0_MAX = 1e-16, 1e-5


def saturation_current_input() -> float:
    """Decade select-slider + synced number box for the saturation current J_0.

    J_0 spans many orders of magnitude, so the select-slider gives a quick
    decade sweep while the number box allows an exact value (e.g. 3e-14). The
    number box is the source of truth for the model; when it changes, the decade
    selector snaps to the nearest option so the two stay visually consistent.
    """
    if "j_0_num" not in st.session_state:
        st.session_state["j_0_num"] = 1e-13
        st.session_state["j_0_sel"] = 1e-13

    def _sync_from_select() -> None:
        st.session_state["j_0_num"] = st.session_state["j_0_sel"]

    def _sync_from_number() -> None:
        value = st.session_state["j_0_num"]
        # Snap the decade selector to the option closest in log10 space.
        st.session_state["j_0_sel"] = min(
            SATURATION_CURRENT_OPTIONS,
            key=lambda option: abs(math.log10(option) - math.log10(value)),
        )

    help_text = (
        "Reverse saturation current density in A/cm². This controls the diode "
        "recombination current and strongly affects open-circuit voltage."
    )
    col_select, col_number = st.columns([3, 1], gap="small")
    with col_select:
        st.select_slider(
            "Saturation current density J_0 (A/cm²)",
            options=SATURATION_CURRENT_OPTIONS,
            format_func=lambda x: f"{x:.0e}",
            key="j_0_sel",
            help=help_text,
            on_change=_sync_from_select,
        )
    with col_number:
        st.number_input(
            "Saturation current density J_0 (A/cm²)",
            min_value=J0_MIN,
            max_value=J0_MAX,
            step=1e-14,
            format="%.2e",
            key="j_0_num",
            on_change=_sync_from_number,
            label_visibility="collapsed",
        )

    return st.session_state["j_0_num"]


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

    st.caption("Drag a slider for a quick sweep, or type an exact value in the box.")

    j_ph_ma = slider_with_number(
        "Photo-current density J_ph (mA/cm²)",
        min_value=0.0,
        max_value=50.0,
        value=40.0,
        step=0.5,
        key="j_ph_ma",
        fmt="%.1f",
        help=(
            "Light-generated current density in mA/cm². Higher values raise "
            "the short-circuit current density of the light JV curve."
        ),
    )
    j_0 = saturation_current_input()
    n = slider_with_number(
        "Ideality factor n",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
        key="n",
        fmt="%.2f",
        help=(
            "Dimensionless diode ideality factor. Values near 1 represent a "
            "more ideal diode; larger values indicate stronger recombination."
        ),
    )
    r_s = slider_with_number(
        "Series resistance R_s (Ω·cm²)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.05,
        key="r_s",
        fmt="%.2f",
        help=(
            "Area-normalised series resistance in Ω·cm² from contacts, bulk "
            "material, and wiring. Larger values reduce current at high voltage."
        ),
    )
    r_sh = slider_with_number(
        "Shunt resistance R_sh (Ω·cm²)",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0,
        key="r_sh",
        fmt="%.0f",
        help=(
            "Area-normalised leakage-path resistance in Ω·cm². Higher values "
            "generally mean less leakage near short circuit."
        ),
    )

    st.header("Operating conditions")
    temp_c = slider_with_number(
        "Cell temperature (deg C)",
        min_value=-20,
        max_value=85,
        value=25,
        step=1,
        key="temp_c",
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

    # Compute the dark curve once (when enabled) so the linear, log JV and m(V)
    # plots all share the same arrays.
    dark_curve = iv_curve(params, dark=True) if show_dark else None
    v_dark, i_dark = dark_curve if dark_curve is not None else (None, None)

    fig = iv_curve_figure(
        voltage, current, metrics=metrics, title="JV Curve",
        dark_voltage=v_dark, dark_current=i_dark,
    )
    st.plotly_chart(fig, width="stretch")
    if dark_curve is not None:
        st.caption(
            "Light JV curve with the maximum-power point marked and generated "
            "power density shown on the secondary axis. The dashed blue trace "
            "is the dark current density (photo-current set to zero)."
        )
    else:
        st.caption(
            "Light JV curve with the maximum-power point marked and generated "
            "power density shown on the secondary axis."
        )

    # Diagnostic pair: semilog JV and the local ideality factor, overlaying the
    # light and (when enabled) dark curves. Click a legend entry to hide a series.
    jv_series = [("Light", voltage, current)]
    m_series = [
        (
            "Light",
            voltage,
            local_ideality_factor(voltage, current, params.temp_k, j_ph=params.j_ph),
        )
    ]
    if dark_curve is not None:
        jv_series.append(("Dark", v_dark, i_dark))
        # Dark current already encodes zero photocurrent, so j_ph stays 0.
        m_series.append(
            ("Dark", v_dark, local_ideality_factor(v_dark, i_dark, params.temp_k))
        )

    st.plotly_chart(log_jv_figure(jv_series), width="stretch")
    st.caption(
        "Semilog JV curve: |current density| on a log axis reveals the "
        "exponential diode region across several decades. Click a legend entry "
        "to toggle the light or dark series."
    )

    st.plotly_chart(ideality_factor_figure(m_series), width="stretch")
    st.caption(
        "Local ideality factor m(V) = (1/Vt)·dV/d(ln|J|). It sits near the diode "
        "ideality factor n in the exponential region and departs where series/"
        "shunt resistance dominate; gaps appear near the J→0 crossing (Voc)."
    )
