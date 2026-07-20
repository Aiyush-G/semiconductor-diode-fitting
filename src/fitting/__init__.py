"""Uncertainty, profiling, and joint-fitting tools.

The original single-diode optimiser remains in :mod:`src.models.fitting` for
backwards compatibility with the Streamlit application.  This package grows
the inference-facing fitting tools beside it.
"""

from src.fitting.uncertainty import (
    JacobianUncertainty,
    covariance_from_jacobian,
    estimate_fit_uncertainty,
)

__all__ = [
    "JacobianUncertainty",
    "covariance_from_jacobian",
    "estimate_fit_uncertainty",
]