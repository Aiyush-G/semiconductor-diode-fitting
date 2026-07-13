"""
Plotly figure builders for IV/JV curves. 
"""

import numpy as np
import plotly.graph_objects as go


def iv_curve_figure(
    voltage: np.ndarray,
    current: np.ndarray,
    metrics: dict | None = None,
    title: str = "IV Curve",
) -> go.Figure:
    """Build a Plotly figure for an IV curve, with an optional power overlay
    on a secondary axis and key metrics marked (Isc, Voc, MPP).
    """
    power = voltage * current

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=voltage, y=current, mode="lines", name="Current (A)",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=voltage, y=power, mode="lines", name="Power (W)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        yaxis="y2",
    ))

    if metrics:
        fig.add_trace(go.Scatter(
            x=[metrics["vmp"]], y=[metrics["imp"]], mode="markers",
            name="Max Power Point", marker=dict(color="red", size=10, symbol="x"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Voltage (V)",
        yaxis_title="Current (A)",
        yaxis2=dict(title="Power (W)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
