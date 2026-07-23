"""
Plotly figure builders for JV (current-density vs voltage) curves.

Current is handled internally as a density in A/cm^2 (PV Lighthouse
convention); it is converted to mA/cm^2 for display, and power to mW/cm^2.
"""

import numpy as np
import plotly.graph_objects as go

# Shared per-series colours so the light/dark traces match the linear JV plot.
SERIES_COLORS = {"Light": "#1f77b4", "Dark": "#ff7f0e"}


def iv_curve_figure(
    voltage: np.ndarray,
    current: np.ndarray,
    metrics: dict | None = None,
    title: str = "JV Curve",
    dark_voltage: np.ndarray | None = None,
    dark_current: np.ndarray | None = None,
    measured_voltage: np.ndarray | None = None,
    measured_current: np.ndarray | None = None,
    measured_label: str = "Measured data",
    fitted_voltage: np.ndarray | None = None,
    fitted_current: np.ndarray | None = None,
    extra_series: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> go.Figure:
    """Build a Plotly figure for a JV curve, with an optional power-density
    overlay on a secondary axis and key metrics marked (Jsc, Voc, MPP).

    When ``dark_current`` is supplied, the dark current density is overlaid on
    the same primary axis as a dashed blue trace so light and dark share one
    figure. Colour encodes the quantity (blue = current, orange = power) while
    line style encodes light (solid) vs dark (dashed).

    Imported measurements and a fitted curve can also be overlaid: measurements
    are drawn as green markers and the fit as a solid red line, both on the
    primary current axis (converted to mA/cm²).

    Args:
        voltage: voltage points (V)
        current: current density (A/cm^2)
        metrics: optional dict from ``key_metrics`` (vmp in V, jmp in A/cm^2)
        title: figure title
        dark_voltage: optional dark-curve voltage points (V)
        dark_current: optional dark current density (A/cm^2); when given, drawn
            as a dashed blue overlay on the primary axis
        measured_voltage: optional imported-measurement voltage points (V)
        measured_current: optional imported-measurement current density (A/cm^2),
            drawn as markers on the primary axis
        measured_label: legend label for the measured markers
        fitted_voltage: optional fitted-curve voltage points (V)
        fitted_current: optional fitted current density (A/cm^2), drawn as a
            solid red line on the primary axis
        extra_series: optional list of (label, voltage, current) auxiliary
            curves (e.g. tandem sub-cell JV curves), drawn as thin dashed
            lines on the primary axis
    """
    has_dark = dark_current is not None
    has_measured = measured_current is not None
    has_fit = fitted_current is not None
    # Convert to display units: mA/cm^2 for current density, mW/cm^2 for power.
    current_ma = current * 1e3
    power_mw = voltage * current * 1e3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=voltage, y=current_ma, mode="lines",
        name="Light current density (mA/cm²)" if has_dark else "Current density (mA/cm²)",
        line=dict(color="#1f77b4", width=2),
    ))
    if has_dark:
        fig.add_trace(go.Scatter(
            x=dark_voltage, y=dark_current * 1e3, mode="lines",
            name="Dark current density (mA/cm²)",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=voltage, y=power_mw, mode="lines", name="Power density (mW/cm²)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        yaxis="y2",
    ))

    if has_measured:
        fig.add_trace(go.Scatter(
            x=measured_voltage, y=measured_current * 1e3, mode="markers",
            name=measured_label,
            marker=dict(color="#2ca02c", size=6, symbol="circle-open"),
        ))
    if has_fit:
        fig.add_trace(go.Scatter(
            x=fitted_voltage, y=fitted_current * 1e3, mode="lines",
            name="Fitted curve (mA/cm²)",
            line=dict(color="#d62728", width=2),
        ))

    if extra_series:
        aux_palette = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for index, (label, aux_voltage, aux_current) in enumerate(extra_series):
            fig.add_trace(go.Scatter(
                x=aux_voltage, y=aux_current * 1e3, mode="lines",
                name=f"{label} (mA/cm²)",
                line=dict(
                    color=aux_palette[index % len(aux_palette)],
                    width=1.5, dash="dash",
                ),
            ))

    if metrics:
        fig.add_trace(go.Scatter(
            x=[metrics["vmp"]], y=[metrics["jmp"] * 1e3], mode="markers",
            name="Max Power Point", marker=dict(color="red", size=10, symbol="x"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title="Current density (mA/cm²)",
        yaxis2=dict(title="Power density (mW/cm²)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _series_color(label: str, index: int) -> str:
    """Pick a colour for a named series, falling back to the Plotly cycle."""
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    return SERIES_COLORS.get(label, palette[index % len(palette)])


def log_jv_figure(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    title: str = "Log JV Curve",
    measured_voltage: np.ndarray | None = None,
    measured_current: np.ndarray | None = None,
    measured_label: str = "Measured data",
) -> go.Figure:
    """Build a semilog JV figure: |J| (mA/cm^2) on a log y-axis vs voltage.

    Overlays each named series (e.g. "Light" and "Dark") as its own trace;
    click a legend entry to toggle that series off. Imported measurements can
    also be overlaid as green markers, matching ``iv_curve_figure``.

    Args:
        series: list of (label, voltage [V], current_density [A/cm^2]) tuples
        title: figure title
        measured_voltage: optional imported-measurement voltage points (V)
        measured_current: optional imported-measurement current density
            (A/cm^2), drawn as markers
        measured_label: legend label for the measured markers
    """
    fig = go.Figure()
    for index, (label, voltage, current) in enumerate(series):
        # Log axis needs positive values; take |J| and let Plotly log-scale it.
        current_ma = np.abs(current) * 1e3
        fig.add_trace(go.Scatter(
            x=voltage, y=current_ma, mode="lines", name=label,
            line=dict(color=_series_color(label, index), width=2),
        ))

    if measured_current is not None:
        fig.add_trace(go.Scatter(
            x=measured_voltage, y=np.abs(measured_current) * 1e3, mode="markers",
            name=measured_label,
            marker=dict(color="#2ca02c", size=6, symbol="circle-open"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title="|Current density| (mA/cm²)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(type="log")
    return fig


def ideality_factor_figure(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    title: str = "Local Ideality Factor m(V)",
    measured_voltage: np.ndarray | None = None,
    measured_m: np.ndarray | None = None,
    measured_label: str = "Measured data",
) -> go.Figure:
    """Build a local-ideality-factor figure: m vs voltage.

    The m arrays are computed upstream by the model
    (``single_diode.local_ideality_factor``); this builder only draws them, so
    NaN entries (near the J -> 0 crossing) render as gaps. Imported
    measurements can also be overlaid as green markers, matching
    ``iv_curve_figure``.

    Args:
        series: list of (label, voltage [V], m [dimensionless]) tuples
        title: figure title
        measured_voltage: optional imported-measurement voltage points (V)
        measured_m: optional local ideality factor computed from the imported
            measurements, drawn as markers
        measured_label: legend label for the measured markers
    """
    fig = go.Figure()
    for index, (label, voltage, m) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=voltage, y=m, mode="lines", name=label,
            line=dict(color=_series_color(label, index), width=2),
        ))

    if measured_m is not None:
        fig.add_trace(go.Scatter(
            x=measured_voltage, y=measured_m, mode="markers",
            name=measured_label,
            marker=dict(color="#2ca02c", size=6, symbol="circle-open"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title="Local ideality factor m",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def residual_figure(
    voltage: np.ndarray,
    residual: np.ndarray,
    residual_space: str = "linear",
    title: str = "Fit residuals",
) -> go.Figure:
    """Build a residual plot for a fit: (fit - measured) vs voltage.

    The residual passed in is always the *linear* current residual (A/cm^2). When
    the fit used log space, that is noted in the axis label but the plotted
    quantity stays in mA/cm^2 so it reads in the same units as the JV plot.

    Args:
        voltage: measured voltage points (V)
        residual: linear current residual (A/cm^2), model minus measured
        residual_space: "linear" or "log"; only affects the axis annotation
        title: figure title
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=voltage, y=residual * 1e3, mode="markers",
        name="Residual", marker=dict(color="#d62728", size=6),
    ))
    # Zero reference line so systematic bias is easy to see.
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))

    space_note = " (fit minimised in log space)" if residual_space == "log" else ""
    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title=f"Residual (mA/cm²){space_note}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
