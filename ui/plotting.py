"""
Plotly figure builders for JV (current-density vs voltage) curves.

Current is handled internally as a density in A/cm^2 (PV Lighthouse
convention); it is converted to mA/cm^2 for display, and power to mW/cm^2.
"""

import numpy as np
import plotly.graph_objects as go


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
