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
) -> go.Figure:
    """Build a Plotly figure for a JV curve, with an optional power-density
    overlay on a secondary axis and key metrics marked (Jsc, Voc, MPP).

    Args:
        voltage: voltage points (V)
        current: current density (A/cm^2)
        metrics: optional dict from ``key_metrics`` (vmp in V, jmp in A/cm^2)
        title: figure title
    """
    # Convert to display units: mA/cm^2 for current density, mW/cm^2 for power.
    current_ma = current * 1e3
    power_mw = voltage * current * 1e3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=voltage, y=current_ma, mode="lines", name="Current density (mA/cm²)",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=voltage, y=power_mw, mode="lines", name="Power density (mW/cm²)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        yaxis="y2",
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
) -> go.Figure:
    """Build a semilog JV figure: |J| (mA/cm^2) on a log y-axis vs voltage.

    Overlays each named series (e.g. "Light" and "Dark") as its own trace;
    click a legend entry to toggle that series off.

    Args:
        series: list of (label, voltage [V], current_density [A/cm^2]) tuples
        title: figure title
    """
    fig = go.Figure()
    for index, (label, voltage, current) in enumerate(series):
        # Log axis needs positive values; take |J| and let Plotly log-scale it.
        current_ma = np.abs(current) * 1e3
        fig.add_trace(go.Scatter(
            x=voltage, y=current_ma, mode="lines", name=label,
            line=dict(color=_series_color(label, index), width=2),
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
) -> go.Figure:
    """Build a local-ideality-factor figure: m vs voltage.

    The m arrays are computed upstream by the model
    (``single_diode.local_ideality_factor``); this builder only draws them, so
    NaN entries (near the J -> 0 crossing) render as gaps.

    Args:
        series: list of (label, voltage [V], m [dimensionless]) tuples
        title: figure title
    """
    fig = go.Figure()
    for index, (label, voltage, m) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=voltage, y=m, mode="lines", name=label,
            line=dict(color=_series_color(label, index), width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title="Local ideality factor m",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
