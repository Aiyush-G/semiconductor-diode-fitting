"""Accessible Streamlit page for exploring the single-diode model."""

import math

import streamlit as st

from src.models.data_import import DataImportError, build_dataset
from src.models.examples import EXAMPLE_DATASETS
from src.models.fitting import (
    PARAM_NAMES,
    default_specs,
    fit_diode,
)
from src.models.single_diode import (
    DiodeParams,
    iv_curve,
    key_metrics,
    local_ideality_factor,
)
from src.models.temperature import TemperatureCoefficients, adjust_params_for_temperature
from ui.inputs import slider_with_number
from ui.plotting import (
    ideality_factor_figure,
    iv_curve_figure,
    log_jv_figure,
    residual_figure,
)


REFERENCE_TEMP_K = 298.15
# Saturation current density J_0 options in A/cm^2 (PV Lighthouse convention).
SATURATION_CURRENT_OPTIONS = [10 ** p for p in range(-15, -5)]
# Bounds for the free-form J_0 number box (span the decade selector, with a
# little headroom on either side).
J0_MIN, J0_MAX = 1e-16, 1e-5


def saturation_current_input(
    fit_key: str | None = None,
    fit_disabled: bool = False,
) -> float:
    """Decade select-slider + synced number box for the saturation current J_0.

    J_0 spans many orders of magnitude, so the select-slider gives a quick
    decade sweep while the number box allows an exact value (e.g. 3e-14). The
    number box is the source of truth for the model; when it changes, the decade
    selector snaps to the nearest option so the two stay visually consistent.

    When ``fit_key`` is given, a "Fit" checkbox is rendered as a third column
    (matching ``slider_with_number``); the caller reads ``st.session_state[fit_key]``.
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
    if fit_key is not None:
        col_select, col_number, col_fit = st.columns([3, 1, 1], gap="small")
    else:
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
    if fit_key is not None:
        if fit_key not in st.session_state:
            st.session_state[fit_key] = True
        with col_fit:
            st.checkbox("Fit", key=fit_key, disabled=fit_disabled)

    return st.session_state["j_0_num"]




@st.dialog("Custom Data")
def data_load_dialog() -> None:
    """Modal for loading measured J-V data. Loading only — no fit controls here.

    On a successful load (example or imported), the dataset is stored in
    session_state and the modal is closed via ``st.rerun`` so the main page shows
    the loaded state and enables the fit controls.
    """
    st.caption(
        "Load measured J-V data to overlay on the graph and fit. Light data can "
        "fit J_L, J_0, n, R_s, R_sh; dark data fits J_0, n, R_s, R_sh (no photocurrent)."
    )

    example_choice = st.selectbox(
        "Example dataset",
        ["None", *EXAMPLE_DATASETS.keys()],
        key="fit_example_choice",
    )
    if st.button(
        "Load example",
        key="fit_load_example",
        disabled=example_choice == "None",
    ):
        st.session_state["imported_dataset"] = EXAMPLE_DATASETS[example_choice]
        st.session_state.pop("fit_result", None)
        st.rerun()

    st.markdown("**Or import your own data**")
    dataset_name = st.text_input("Dataset name", value="My dataset", key="fit_name")
    kind_label = st.radio(
        "Data type", ["Light JV", "Dark JV"], key="fit_kind", horizontal=True
    )
    v_unit_label = st.radio(
        "Voltage units", ["V", "mV"], key="fit_v_units", horizontal=True
    )
    i_unit_label = st.radio(
        "Current units", ["A/cm²", "mA/cm²"], key="fit_i_units", horizontal=True
    )
    pasted = st.text_area(
        "Paste two columns (voltage, current)",
        key="fit_paste",
        height=130,
        placeholder="0.0, 0.036\n0.1, 0.0358\n...",
    )
    uploaded = st.file_uploader(
        "...or upload a file",
        type=["csv", "txt", "tsv", "dat"],
        key="fit_upload",
    )
    if st.button("Import dataset", key="fit_import", type="primary"):
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8", errors="replace")
        else:
            text = pasted
        try:
            dataset = build_dataset(
                text,
                label=dataset_name or "My dataset",
                kind="light" if kind_label == "Light JV" else "dark",
                voltage_units="V" if v_unit_label == "V" else "mV",
                current_units="A/cm2" if i_unit_label == "A/cm²" else "mA/cm2",
            )
            st.session_state["imported_dataset"] = dataset
            st.session_state.pop("fit_result", None)
            st.rerun()
        except DataImportError as exc:
            st.error(str(exc))


@st.dialog("Fit results", width="large")
def fit_results_dialog() -> None:
    """Modal showing the most recent fit's summary, metrics, and residuals."""
    imported_dataset = st.session_state.get("imported_dataset")
    fit_result = st.session_state.get("fit_result")
    if fit_result is None or imported_dataset is None:
        st.caption("No fit has been run yet.")
        return

    status = "converged" if fit_result.success else "did not converge"
    st.caption(
        f"Fitted {', '.join(fit_result.free_names)} to '{imported_dataset.label}' "
        f"({fit_result.n_points} points, {fit_result.residual_space} residuals) — {status}."
    )

    p = fit_result.params
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    pcol1.metric("J_L (mA/cm²)", f"{p.j_ph * 1e3:.3f}")
    pcol2.metric("J_0 (A/cm²)", f"{p.j_0:.3e}")
    pcol3.metric("n", f"{p.n:.3f}")
    pcol4.metric("R_s (Ω·cm²)", f"{p.r_s:.3f}")
    pcol5.metric("R_sh (Ω·cm²)", f"{p.r_sh:.4g}")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("RMSE (mA/cm²)", f"{fit_result.rmse * 1e3:.4f}")
    mcol2.metric("R²", f"{fit_result.r_squared:.5f}")
    mcol3.metric("Max |resid| (mA/cm²)", f"{fit_result.max_abs_residual * 1e3:.4f}")
    if fit_result.rmse_log is not None:
        mcol4.metric("RMSE (log₁₀|J|)", f"{fit_result.rmse_log:.4f}")

    st.caption(f"Optimizer message: {fit_result.message}")

    st.plotly_chart(
        residual_figure(
            imported_dataset.voltage,
            fit_result.residual,
            residual_space=fit_result.residual_space,
        ),
        width="stretch",
    )
    st.caption(
        "Residual = fitted − measured current density at each point. A "
        "structureless scatter about zero indicates a good fit; systematic "
        "curvature points to a model/parameter mismatch."
    )


# Page metadata and opening copy are kept concise so keyboard and screen-reader
# users reach the controls quickly.
st.set_page_config(page_title="Single Diode", layout="wide")

st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        div[data-testid="stColumn"]:has(.single-diode-control-rail-marker),
        div[data-testid="column"]:has(.single-diode-control-rail-marker) {
            position: sticky;
            top: 4.25rem;
            align-self: flex-start;
            max-height: calc(100vh - 5rem);
            overflow-y: auto;
            padding-right: 0.35rem;
        }

        div[data-testid="stColumn"]:has(.single-diode-control-rail-marker)
            > div[data-testid="stVerticalBlock"],
        div[data-testid="column"]:has(.single-diode-control-rail-marker)
            > div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="stColumn"]:has(.single-diode-control-rail-marker),
        div[data-testid="column"]:has(.single-diode-control-rail-marker) {
            position: static;
            max-height: none;
            overflow: visible;
            padding-right: 0;
        }
    }

    .single-diode-control-rail-marker {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Single Diode Model")

st.markdown(
    "Explore how equivalent-circuit parameters change a solar-cell JV "
    "(current-density) curve under light and optional dark conditions."
)

# Keep inputs and outputs in separate columns so the interaction flow is
# predictable: set reference values first, then read the computed result.
col_controls, col_results = st.columns([1, 2], gap="large")

with col_controls:
    st.markdown(
        '<div class="single-diode-control-rail-marker"></div>',
        unsafe_allow_html=True,
    )

    # --- Custom fitting: load data (modal) + fit controls ------------------
    with st.expander("Custom Fitting", expanded=True):
        # Place Load, Fit and Clear on a single row for compact controls.
        col_load_btn, col_fit_btn, col_clear_btn = st.columns([1, 1, 1])

        with col_load_btn:
            if st.button("Load data", key="fit_open_dialog"):
                data_load_dialog()

        dataset = st.session_state.get("imported_dataset")
        if dataset is not None:
            st.caption(
                f"Loaded: {dataset.label} — {dataset.kind}, {dataset.voltage.size} points"
            )

        # Fit / Clear sit side by side on the same row and are only clickable once data is loaded.
        fit_clicked = col_fit_btn.button(
            "Fit dataset", key="fit_run", type="primary", disabled=dataset is None,
        )
        clear_clicked = col_clear_btn.button(
            "Clear dataset", key="fit_clear", disabled=dataset is None,
        )
        if dataset is not None:
            residual_space = st.radio(
                "Residual space",
                ["auto", "linear", "log"],
                key="fit_residual_space",
                horizontal=True,
                help=(
                    "How points are weighted: 'log' suits dark data spanning many "
                    "decades; 'linear' suits light data. 'auto' chooses per data type."
                ),
            )
        else:
            residual_space = "auto"

    with st.expander("Reference parameters", expanded=True):
        st.caption(
            "Area-normalised circuit values at the 25 deg C reference condition "
            "(PV Lighthouse convention)."
        )

        st.caption("Drag a slider for a quick sweep, or type an exact value in the box.")
        st.caption(
            "The value shown for each parameter is used by the fit: it is the initial "
            "guess when the parameter is ticked to fit, or held constant when unticked."
        )

        st.caption("Once a custom dataset has been loaded, tick a parameter to fit it; unticked parameters stay fixed at the value shown.")

        # J_ph is a light-only parameter, so its Fit checkbox is disabled for dark data.
        dark_loaded = dataset is not None and dataset.kind == "dark"

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
            fit_key="fit_free_j_ph",
            fit_disabled=dark_loaded,
        )
        j_0 = saturation_current_input(fit_key="fit_free_j_0")
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
            fit_key="fit_free_n",
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
            fit_key="fit_free_r_s",
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
            fit_key="fit_free_r_sh",
        )

    with st.expander("Operating conditions", expanded=True):
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

    # The fit temperature is taken from the cell-temperature control above (a
    # known, fixed input) rather than being fitted.
    fit_temp_k = temp_c + 273.15

    # Execute the Fit / Clear actions captured at the top of the rail now that the
    # reference-parameter values, Fit checkboxes, and temperature are all resolved.
    # The reference-parameter values (converted to model units) are the fit's
    # initial guesses; only ticked parameters are freed.
    if dataset is not None and clear_clicked:
        st.session_state.pop("imported_dataset", None)
        st.session_state.pop("fit_result", None)
        st.rerun()
    if dataset is not None and fit_clicked:
        fit_initial = {
            "j_ph": j_ph_ma * 1e-3,  # mA/cm² -> A/cm²
            "j_0": j_0,
            "n": n,
            "r_s": r_s,
            "r_sh": r_sh,
        }
        fit_free = {
            name for name in PARAM_NAMES
            if st.session_state.get(f"fit_free_{name}", False)
        }
        # default_specs drops j_ph for dark data, so a stray tick can't fit it.
        specs = default_specs(dataset.kind, free=fit_free, initial=fit_initial)
        st.session_state["fit_result"] = fit_diode(
            dataset.voltage,
            dataset.current,
            fit_temp_k,
            specs,
            kind=dataset.kind,
            residual_space=residual_space,
        )
        fit_results_dialog()

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

    # Overlay imported measurements and, once fitted, the fitted curve onto the
    # existing JV graph. Both are pulled from session_state so reruns preserve them.
    imported_dataset = st.session_state.get("imported_dataset")
    fit_result = st.session_state.get("fit_result")
    measured_voltage = imported_dataset.voltage if imported_dataset is not None else None
    measured_current = imported_dataset.current if imported_dataset is not None else None
    measured_label = imported_dataset.label if imported_dataset is not None else "Measured data"
    fitted_voltage = imported_dataset.voltage if fit_result is not None else None
    fitted_current = fit_result.model_current if fit_result is not None else None

    fig = iv_curve_figure(
        voltage, current, metrics=metrics, title="JV Curve",
        dark_voltage=v_dark, dark_current=i_dark,
        measured_voltage=measured_voltage, measured_current=measured_current,
        measured_label=measured_label,
        fitted_voltage=fitted_voltage, fitted_current=fitted_current,
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

    # --- Fit summary and residuals (shown in a modal) ----------------------
    if fit_result is not None:
        status = "converged" if fit_result.success else "did not converge"
        st.caption(f"Last fit {status}. ")
        if st.button("View fit results", key="fit_view_results"):
            fit_results_dialog()
