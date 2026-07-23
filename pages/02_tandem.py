"""Streamlit page for the 2-terminal tandem model (two sub-cells in series).

Mirrors the single-diode page: control rail on the left (fitting, per-sub-cell
reference parameters with Fit checkboxes, operating conditions) and results on
the right. Every session-state key is prefixed ``td_`` so the tandem page and
the single-diode page never leak datasets, fit results, or widget values into
each other (Streamlit session state is shared across pages).
"""

import streamlit as st

from src.models.data_import import DataImportError, build_dataset
from src.models.examples_tandem import TANDEM_EXAMPLE_DATASETS
from src.models.single_diode import (
    DiodeParams,
    key_metrics,
    local_ideality_factor,
)
from src.models.tandem import TandemParams, tandem_iv_curve, tandem_subcell_curves
from src.models.tandem_fitting import (
    TANDEM_PARAM_NAMES,
    default_tandem_specs,
    fit_tandem,
)
from src.models.temperature import TemperatureCoefficients, adjust_params_for_temperature
from ui.inputs import saturation_current_input, slider_with_number
from ui.plotting import (
    ideality_factor_figure,
    iv_curve_figure,
    log_jv_figure,
    residual_figure,
)


REFERENCE_TEMP_K = 298.15

# Freeing all 10 parameters against one curve is hopelessly degenerate, so the
# Fit checkboxes default to a small, identifiable subset; the rest start fixed.
_DEFAULT_FREE = {"top_j_0", "top_n", "bot_j_0", "bot_n"}
for _name in TANDEM_PARAM_NAMES:
    _key = f"td_fit_free_{_name}"
    if _key not in st.session_state:
        st.session_state[_key] = _name in _DEFAULT_FREE


@st.dialog("Custom Data")
def data_load_dialog() -> None:
    """Modal for loading measured tandem J-V data. Loading only — no fit controls.

    On a successful load (example or imported), the dataset is stored in
    session_state and the modal is closed via ``st.rerun`` so the main page shows
    the loaded state and enables the fit controls.
    """
    st.caption(
        "Load a measured terminal J-V curve of the tandem stack to overlay on "
        "the graph and fit. Light data can fit any of the 10 sub-cell "
        "parameters; dark data excludes both photocurrents."
    )

    example_choice = st.selectbox(
        "Example dataset",
        ["None", *TANDEM_EXAMPLE_DATASETS.keys()],
        key="td_fit_example_choice",
    )
    if st.button(
        "Load example",
        key="td_fit_load_example",
        disabled=example_choice == "None",
    ):
        st.session_state["td_imported_dataset"] = TANDEM_EXAMPLE_DATASETS[example_choice]
        st.session_state.pop("td_fit_result", None)
        st.rerun()

    st.markdown("**Or import your own data**")
    dataset_name = st.text_input("Dataset name", value="My dataset", key="td_fit_name")
    kind_label = st.radio(
        "Data type", ["Light JV", "Dark JV"], key="td_fit_kind", horizontal=True
    )
    v_unit_label = st.radio(
        "Voltage units", ["V", "mV"], key="td_fit_v_units", horizontal=True
    )
    i_unit_label = st.radio(
        "Current units", ["A/cm²", "mA/cm²"], key="td_fit_i_units", horizontal=True
    )
    pasted = st.text_area(
        "Paste two columns (voltage, current)",
        key="td_fit_paste",
        height=130,
        placeholder="0.0, 0.0196\n0.1, 0.0195\n...",
    )
    uploaded = st.file_uploader(
        "...or upload a file",
        type=["csv", "txt", "tsv", "dat"],
        key="td_fit_upload",
    )
    if st.button("Import dataset", key="td_fit_import", type="primary"):
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
            st.session_state["td_imported_dataset"] = dataset
            st.session_state.pop("td_fit_result", None)
            st.rerun()
        except DataImportError as exc:
            st.error(str(exc))


@st.dialog("Fit results", width="large")
def fit_results_dialog() -> None:
    """Modal showing the most recent tandem fit's summary, metrics, and residuals."""
    imported_dataset = st.session_state.get("td_imported_dataset")
    fit_result = st.session_state.get("td_fit_result")
    if fit_result is None or imported_dataset is None:
        st.caption("No fit has been run yet.")
        return

    status = "converged" if fit_result.success else "did not converge"
    st.caption(
        f"Fitted {', '.join(fit_result.free_names)} to '{imported_dataset.label}' "
        f"({fit_result.n_points} points, {fit_result.residual_space} residuals) — {status}."
    )

    for cell_label, p in (
        ("Top", fit_result.params.top),
        ("Bottom", fit_result.params.bottom),
    ):
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        pcol1.metric(f"{cell_label} J_L (mA/cm²)", f"{p.j_ph * 1e3:.3f}")
        pcol2.metric(f"{cell_label} J_0 (A/cm²)", f"{p.j_0:.3e}")
        pcol3.metric(f"{cell_label} n", f"{p.n:.3f}")
        pcol4.metric(f"{cell_label} R_s (Ω·cm²)", f"{p.r_s:.3f}")
        pcol5.metric(f"{cell_label} R_sh (Ω·cm²)", f"{p.r_sh:.4g}")

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
        "curvature points to a model/parameter mismatch. With 10 tandem "
        "parameters, an excellent residual does not guarantee unique "
        "parameters — fit small subsets and sanity-check the values."
    )


# Page metadata and opening copy are kept concise so keyboard and screen-reader
# users reach the controls quickly.
st.set_page_config(page_title="Tandem", layout="wide")

st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        div[data-testid="stColumn"]:has(.tandem-control-rail-marker),
        div[data-testid="column"]:has(.tandem-control-rail-marker) {
            position: sticky;
            top: 4.25rem;
            align-self: flex-start;
            max-height: calc(100vh - 5rem);
            overflow-y: auto;
            padding-right: 0.35rem;
        }

        div[data-testid="stColumn"]:has(.tandem-control-rail-marker)
            > div[data-testid="stVerticalBlock"],
        div[data-testid="column"]:has(.tandem-control-rail-marker)
            > div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="stColumn"]:has(.tandem-control-rail-marker),
        div[data-testid="column"]:has(.tandem-control-rail-marker) {
            position: static;
            max-height: none;
            overflow: visible;
            padding-right: 0;
        }
    }

    .tandem-control-rail-marker {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tandem Model (2-Terminal)")

st.markdown(
    "Two single-diode sub-cells in series share one current density "
    "(current matching), so the terminal voltage is the sum of the sub-cell "
    "voltages: V(J) = V_top(J) + V_bottom(J). Explore how the 10 sub-cell "
    "parameters shape the tandem JV curve."
)

# Keep inputs and outputs in separate columns so the interaction flow is
# predictable: set reference values first, then read the computed result.
col_controls, col_results = st.columns([1, 2], gap="large")

with col_controls:
    st.markdown(
        '<div class="tandem-control-rail-marker"></div>',
        unsafe_allow_html=True,
    )

    # --- Custom fitting: load data (modal) + fit controls ------------------
    with st.expander("Custom Fitting", expanded=True):
        # Place Load, Fit and Clear on a single row for compact controls.
        col_load_btn, col_fit_btn, col_clear_btn = st.columns([1, 1, 1])

        with col_load_btn:
            if st.button("Load data", key="td_fit_open_dialog"):
                data_load_dialog()

        dataset = st.session_state.get("td_imported_dataset")
        if dataset is not None:
            st.caption(
                f"Loaded: {dataset.label} — {dataset.kind}, {dataset.voltage.size} points"
            )

        # Fit / Clear sit side by side and are only clickable once data is loaded.
        fit_clicked = col_fit_btn.button(
            "Fit dataset", key="td_fit_run", type="primary", disabled=dataset is None,
        )
        clear_clicked = col_clear_btn.button(
            "Clear dataset", key="td_fit_clear", disabled=dataset is None,
        )
        if dataset is not None:
            residual_space = st.radio(
                "Residual space",
                ["auto", "linear", "log"],
                key="td_fit_residual_space",
                horizontal=True,
                help=(
                    "How points are weighted: 'log' suits dark data spanning many "
                    "decades; 'linear' suits light data. 'auto' chooses per data type."
                ),
            )
        else:
            residual_space = "auto"
        st.caption(
            "10 parameters against one curve is strongly degenerate: many "
            "parameter sets fit equally well. Free a small subset (a few "
            "checkboxes) and keep the rest fixed at known values."
        )

    st.caption(
        "Reference parameters are area-normalised circuit values at the 25 °C "
        "reference condition (PV Lighthouse convention). Drag a slider for a "
        "quick sweep, or type an exact value in the box. Once a dataset is "
        "loaded, tick a parameter to fit it; unticked parameters stay fixed "
        "at the value shown (the shown value is the fit's initial guess)."
    )

    # Photocurrents are light-only parameters, so their Fit checkboxes are
    # disabled for dark data.
    dark_loaded = dataset is not None and dataset.kind == "dark"

    with st.expander("Top cell — perovskite", expanded=True):
        top_j_ph_ma = slider_with_number(
            "Photo-current density J_ph (mA/cm²)",
            min_value=0.0,
            max_value=30.0,
            value=20.0,
            step=0.25,
            key="td_top_j_ph_ma",
            fmt="%.2f",
            help=(
                "Light generated in the wide-bandgap top sub-cell. In a "
                "2-terminal stack the smaller sub-cell photocurrent limits the "
                "tandem short-circuit current."
            ),
            fit_key="td_fit_free_top_j_ph",
            fit_disabled=dark_loaded,
        )
        top_j_0 = saturation_current_input(
            "Saturation current density J_0 (A/cm²)",
            key="td_top_j_0",
            default=1e-16,
            decade_range=(-20, -8),
            min_value=1e-21,
            max_value=1e-7,
            help=(
                "Reverse saturation current density of the top sub-cell. A "
                "wide-bandgap junction sits many decades below silicon's, "
                "which is what gives the tandem its high Voc."
            ),
            fit_key="td_fit_free_top_j_0",
        )
        top_n = slider_with_number(
            "Ideality factor n",
            min_value=1.0,
            max_value=2.5,
            value=1.5,
            step=0.05,
            key="td_top_n",
            fmt="%.2f",
            help=(
                "Dimensionless diode ideality factor of the top sub-cell. "
                "Perovskite junctions commonly sit between 1 and 2."
            ),
            fit_key="td_fit_free_top_n",
        )
        top_r_s = slider_with_number(
            "Series resistance R_s (Ω·cm²)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.05,
            key="td_top_r_s",
            fmt="%.2f",
            help=(
                "Area-normalised series resistance of the top sub-cell, "
                "including its transparent contact. Sub-cell resistances add "
                "in the series stack."
            ),
            fit_key="td_fit_free_top_r_s",
        )
        top_r_sh = slider_with_number(
            "Shunt resistance R_sh (Ω·cm²)",
            min_value=100.0,
            max_value=100000.0,
            value=2000.0,
            step=100.0,
            key="td_top_r_sh",
            fmt="%.0f",
            help=(
                "Area-normalised leakage-path resistance of the top sub-cell. "
                "It also sets how much extra current a current-limited "
                "sub-cell can pass in reverse bias."
            ),
            fit_key="td_fit_free_top_r_sh",
        )
        top_eg = st.number_input(
            "Bandgap E_g (eV)",
            min_value=0.5,
            max_value=3.0,
            value=1.68,
            step=0.01,
            key="td_top_eg",
            help=(
                "Bandgap used only in the temperature model for this "
                "sub-cell's saturation current; it is never fitted."
            ),
        )
        top_alpha = st.number_input(
            "Jsc temperature coefficient α (1/K)",
            min_value=-0.005,
            max_value=0.005,
            value=0.0002,
            step=0.0001,
            format="%.4f",
            key="td_top_alpha",
            help=(
                "Fractional photocurrent temperature coefficient of this "
                "sub-cell (0.0002 = +0.02 %/K); temperature model only."
            ),
        )

    with st.expander("Bottom cell — silicon", expanded=True):
        bot_j_ph_ma = slider_with_number(
            "Photo-current density J_ph (mA/cm²)",
            min_value=0.0,
            max_value=30.0,
            value=19.5,
            step=0.25,
            key="td_bot_j_ph_ma",
            fmt="%.2f",
            help=(
                "Light generated in the silicon bottom sub-cell from the "
                "spectrum transmitted through the top cell. Slightly below "
                "the top cell's value here, so the bottom cell limits."
            ),
            fit_key="td_fit_free_bot_j_ph",
            fit_disabled=dark_loaded,
        )
        bot_j_0 = saturation_current_input(
            "Saturation current density J_0 (A/cm²)",
            key="td_bot_j_0",
            default=1e-13,
            decade_range=(-16, -6),
            min_value=1e-17,
            max_value=1e-5,
            help=(
                "Reverse saturation current density of the silicon bottom "
                "sub-cell (typically ~1e-13 A/cm² at 25 °C)."
            ),
            fit_key="td_fit_free_bot_j_0",
        )
        bot_n = slider_with_number(
            "Ideality factor n",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="td_bot_n",
            fmt="%.2f",
            help=(
                "Dimensionless diode ideality factor of the bottom sub-cell. "
                "Good silicon junctions sit near 1."
            ),
            fit_key="td_fit_free_bot_n",
        )
        bot_r_s = slider_with_number(
            "Series resistance R_s (Ω·cm²)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.05,
            key="td_bot_r_s",
            fmt="%.2f",
            help=(
                "Area-normalised series resistance of the bottom sub-cell and "
                "the recombination/tunnel junction between the sub-cells."
            ),
            fit_key="td_fit_free_bot_r_s",
        )
        bot_r_sh = slider_with_number(
            "Shunt resistance R_sh (Ω·cm²)",
            min_value=100.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0,
            key="td_bot_r_sh",
            fmt="%.0f",
            help=(
                "Area-normalised leakage-path resistance of the bottom "
                "sub-cell."
            ),
            fit_key="td_fit_free_bot_r_sh",
        )
        bot_eg = st.number_input(
            "Bandgap E_g (eV)",
            min_value=0.5,
            max_value=3.0,
            value=1.121,
            step=0.01,
            format="%.3f",
            key="td_bot_eg",
            help=(
                "Bandgap used only in the temperature model for this "
                "sub-cell's saturation current; it is never fitted."
            ),
        )
        bot_alpha = st.number_input(
            "Jsc temperature coefficient α (1/K)",
            min_value=-0.005,
            max_value=0.005,
            value=0.0005,
            step=0.0001,
            format="%.4f",
            key="td_bot_alpha",
            help=(
                "Fractional photocurrent temperature coefficient of this "
                "sub-cell (0.0005 = +0.05 %/K); temperature model only."
            ),
        )

    with st.expander("Operating conditions", expanded=True):
        temp_c = slider_with_number(
            "Cell temperature (deg C)",
            min_value=-20,
            max_value=85,
            value=25,
            step=1,
            key="td_temp_c",
            help=(
                "Device temperature applied to both sub-cells. Each sub-cell's "
                "photocurrent and saturation current are adjusted with its own "
                "bandgap and α before the tandem curve is calculated."
            ),
        )
        show_dark = st.checkbox(
            "Overlay dark IV curve",
            value=False,
            key="td_show_dark",
            help=(
                "Show a second IV curve with both sub-cell photo-currents set "
                "to zero while keeping the other adjusted parameters unchanged."
            ),
        )
        show_subcells = st.checkbox(
            "Show sub-cell JV curves",
            value=False,
            key="td_show_subcells",
            help=(
                "Overlay each sub-cell's own JV curve, evaluated on the shared "
                "series current — their voltages sum to the tandem curve."
            ),
        )

    # The fit temperature is taken from the cell-temperature control above (a
    # known, fixed input) rather than being fitted.
    fit_temp_k = temp_c + 273.15

    # Execute the Fit / Clear actions captured at the top of the rail now that
    # the reference-parameter values, Fit checkboxes, and temperature are all
    # resolved. The reference-parameter values (converted to model units) are
    # the fit's initial guesses; only ticked parameters are freed.
    if dataset is not None and clear_clicked:
        st.session_state.pop("td_imported_dataset", None)
        st.session_state.pop("td_fit_result", None)
        st.rerun()
    if dataset is not None and fit_clicked:
        fit_initial = {
            "top_j_ph": top_j_ph_ma * 1e-3,  # mA/cm² -> A/cm²
            "top_j_0": top_j_0,
            "top_n": top_n,
            "top_r_s": top_r_s,
            "top_r_sh": top_r_sh,
            "bot_j_ph": bot_j_ph_ma * 1e-3,
            "bot_j_0": bot_j_0,
            "bot_n": bot_n,
            "bot_r_s": bot_r_s,
            "bot_r_sh": bot_r_sh,
        }
        fit_free = {
            name for name in TANDEM_PARAM_NAMES
            if st.session_state.get(f"td_fit_free_{name}", False)
        }
        # default_tandem_specs drops both j_ph terms for dark data, so a stray
        # tick can't fit a photocurrent.
        specs = default_tandem_specs(dataset.kind, free=fit_free, initial=fit_initial)
        st.session_state["td_fit_result"] = fit_tandem(
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
ref_stack = TandemParams(
    top=DiodeParams(
        j_ph=top_j_ph_ma * 1e-3, j_0=top_j_0, n=top_n,
        r_s=top_r_s, r_sh=top_r_sh, temp_k=REFERENCE_TEMP_K,
    ),
    bottom=DiodeParams(
        j_ph=bot_j_ph_ma * 1e-3, j_0=bot_j_0, n=bot_n,
        r_s=bot_r_s, r_sh=bot_r_sh, temp_k=REFERENCE_TEMP_K,
    ),
)
target_temp_k = temp_c + 273.15

if abs(target_temp_k - REFERENCE_TEMP_K) > 0.01:
    # Each sub-cell is adjusted with its own bandgap and Jsc coefficient, so
    # temperature can change which sub-cell limits the tandem current.
    params = TandemParams(
        top=adjust_params_for_temperature(
            ref_stack.top, target_temp_k,
            TemperatureCoefficients(alpha_isc=top_alpha, e_g_ev=top_eg),
        ),
        bottom=adjust_params_for_temperature(
            ref_stack.bottom, target_temp_k,
            TemperatureCoefficients(alpha_isc=bot_alpha, e_g_ev=bot_eg),
        ),
    )
else:
    params = ref_stack

# Model evaluation is kept outside the rendering blocks so UI layout changes do
# not affect the physics path.
voltage, current = tandem_iv_curve(params)
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
    dark_curve = tandem_iv_curve(params, dark=True) if show_dark else None
    v_dark, i_dark = dark_curve if dark_curve is not None else (None, None)

    subcell_curves = tandem_subcell_curves(params) if show_subcells else None

    # Overlay imported measurements and, once fitted, the fitted curve onto the
    # existing JV graph. Both are pulled from session_state so reruns preserve them.
    imported_dataset = st.session_state.get("td_imported_dataset")
    fit_result = st.session_state.get("td_fit_result")
    measured_voltage = imported_dataset.voltage if imported_dataset is not None else None
    measured_current = imported_dataset.current if imported_dataset is not None else None
    measured_label = imported_dataset.label if imported_dataset is not None else "Measured data"
    fitted_voltage = imported_dataset.voltage if fit_result is not None else None
    fitted_current = fit_result.model_current if fit_result is not None else None

    fig = iv_curve_figure(
        voltage, current, metrics=metrics, title="Tandem JV Curve",
        dark_voltage=v_dark, dark_current=i_dark,
        measured_voltage=measured_voltage, measured_current=measured_current,
        measured_label=measured_label,
        fitted_voltage=fitted_voltage, fitted_current=fitted_current,
        extra_series=subcell_curves,
    )
    st.plotly_chart(fig, width="stretch")
    caption = (
        "Terminal JV curve of the series stack with the maximum-power point "
        "marked and generated power density on the secondary axis. Jsc sits "
        "just above the smaller sub-cell photocurrent: the current-limited "
        "sub-cell is pushed into reverse bias and its shunt passes the excess."
    )
    if dark_curve is not None:
        caption += " The dashed blue trace is the dark current density."
    if subcell_curves is not None:
        caption += (
            " Thin dashed traces are the sub-cell JV curves on the shared "
            "current; their voltages sum to the tandem curve."
        )
    st.caption(caption)

    # Diagnostic pair: semilog JV and the local ideality factor, overlaying the
    # light and (when enabled) dark curves. Click a legend entry to hide a series.
    j_ph_limit = min(params.top.j_ph, params.bottom.j_ph)
    jv_series = [("Light", voltage, current)]
    m_series = [
        (
            "Light",
            voltage,
            local_ideality_factor(
                voltage, current, params.top.temp_k, j_ph=j_ph_limit
            ),
        )
    ]
    if dark_curve is not None:
        jv_series.append(("Dark", v_dark, i_dark))
        # Dark current already encodes zero photocurrent, so j_ph stays 0.
        m_series.append(
            ("Dark", v_dark, local_ideality_factor(v_dark, i_dark, params.top.temp_k))
        )

    st.plotly_chart(log_jv_figure(jv_series), width="stretch")
    st.caption(
        "Semilog JV curve: |current density| on a log axis reveals the "
        "exponential diode region across several decades. Click a legend entry "
        "to toggle the light or dark series."
    )

    st.plotly_chart(ideality_factor_figure(m_series), width="stretch")
    st.caption(
        "Local ideality factor m(V) = (1/Vt)·dV/d(ln|J|). Because the sub-cell "
        "voltages add at shared current, the tandem's exponential region reads "
        "m ≈ n_top + n_bottom rather than a single sub-cell's n; it departs "
        "where series/shunt resistance dominate."
    )

    # --- Fit summary and residuals (shown in a modal) ----------------------
    if fit_result is not None:
        status = "converged" if fit_result.success else "did not converge"
        st.caption(f"Last fit {status}. ")
        if st.button("View fit results", key="td_fit_view_results"):
            fit_results_dialog()
