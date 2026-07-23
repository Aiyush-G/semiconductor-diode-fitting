"""
Bounded nonlinear least-squares fitting of the 2-terminal tandem model.

The tandem stack has 10 parameters — the five single-diode parameters per
sub-cell, prefixed ``top_`` / ``bot_`` — fitted against a measured terminal
J-V curve via the current-matched forward model
(``tandem.solve_tandem_current``). All of the fitting machinery (per-parameter
free/fixed ``ParamSpec``s, log10 fit space for multi-decade parameters,
residual spaces, penalty handling, metrics) is shared with the single-diode
fit in ``fitting.py``; only the parameter names, the container, and the
forward model differ.

A note on degeneracy: freeing all 10 parameters against a single terminal
curve is hopeless — the sub-cell voltages add at shared current, so many
parameter combinations produce near-identical terminal curves. The intended
workflow (mirroring the single-diode page) is to fix most parameters at
physically motivated values and free a small subset.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.models.fitting import (
    DEFAULT_BOUNDS,
    LOG_PARAMS,
    PARAM_NAMES,
    PENALTY,
    FitResult,
    ParamSpec,
    ResidualSpace,
    _fit_generic,
    resolve_residual_space,
    unpack_values,
)
from src.models.single_diode import DiodeParams
from src.models.tandem import TandemParams, solve_tandem_current

# The 10 tandem parameters: single-diode names prefixed per sub-cell.
TANDEM_PARAM_NAMES: tuple[str, ...] = tuple(
    f"{prefix}_{name}" for prefix in ("top", "bot") for name in PARAM_NAMES
)

# Multi-decade parameters fitted in log10 space, as in the single-diode fit.
TANDEM_LOG_PARAMS = frozenset(
    f"{prefix}_{name}" for prefix in ("top", "bot") for name in LOG_PARAMS
)

# Per-parameter bounds: the single-diode defaults, except the saturation
# current lower bound is extended — a wide-bandgap top cell's j_0 sits many
# decades below silicon's.
TANDEM_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    f"{prefix}_{name}": ((1e-22, hi) if name == "j_0" else (lo, hi))
    for prefix in ("top", "bot")
    for name, (lo, hi) in DEFAULT_BOUNDS.items()
}

# Default starting values for a plausible perovskite (top) / silicon (bottom)
# tandem at 25 degC (A/cm^2, Ohm.cm^2).
TANDEM_DEFAULT_INITIAL: dict[str, float] = {
    "top_j_ph": 0.020,
    "top_j_0": 1e-16,
    "top_n": 1.5,
    "top_r_s": 1.0,
    "top_r_sh": 2000.0,
    "bot_j_ph": 0.0195,
    "bot_j_0": 1e-13,
    "bot_n": 1.0,
    "bot_r_s": 0.5,
    "bot_r_sh": 5000.0,
}


def default_tandem_specs(
    model: Literal["light", "dark"],
    free: set[str],
    initial: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, ParamSpec]:
    """Build the full ``{name: ParamSpec}`` map for a tandem light or dark fit.

    Mirrors ``fitting.default_specs``: for ``model="dark"`` both sub-cell
    photocurrents are structurally excluded — ``top_j_ph`` and ``bot_j_ph``
    are injected as fixed specs at 0 and dropped from ``free``.

    Args:
        model: "light" (all ten parameters available) or "dark" (both j_ph
            forced to 0).
        free: set of parameter names (``TANDEM_PARAM_NAMES``) to fit.
        initial: optional per-parameter starting/fixed values overriding
            ``TANDEM_DEFAULT_INITIAL``.
        bounds: optional per-parameter (lower, upper) overrides of
            ``TANDEM_DEFAULT_BOUNDS``.

    Returns:
        Ordered dict (``TANDEM_PARAM_NAMES`` order) of ``ParamSpec``.
    """
    if model not in ("light", "dark"):
        raise ValueError(f"Unknown model {model!r}; expected 'light' or 'dark'.")

    initial = {**TANDEM_DEFAULT_INITIAL, **(initial or {})}
    bounds = {**TANDEM_DEFAULT_BOUNDS, **(bounds or {})}
    free = set(free)

    if model == "dark":
        # Photocurrent terms are meaningless for dark data; never fit them.
        free.discard("top_j_ph")
        free.discard("bot_j_ph")

    specs: dict[str, ParamSpec] = {}
    for name in TANDEM_PARAM_NAMES:
        if model == "dark" and name in ("top_j_ph", "bot_j_ph"):
            specs[name] = ParamSpec(
                name=name, free=False, value=0.0,
                lower=0.0, upper=0.0, log=False,
            )
            continue
        lower, upper = bounds[name]
        specs[name] = ParamSpec(
            name=name,
            free=name in free,
            value=float(initial[name]),
            lower=float(lower),
            upper=float(upper),
            log=name in TANDEM_LOG_PARAMS,
        )
    return specs


def unpack_tandem(
    theta: np.ndarray, specs: dict[str, ParamSpec], temp_k: float
) -> TandemParams:
    """Rebuild ``TandemParams`` from a fit-space vector plus the fixed specs."""
    values = unpack_values(theta, specs, TANDEM_PARAM_NAMES)
    return TandemParams(
        top=DiodeParams(
            j_ph=values["top_j_ph"], j_0=values["top_j_0"], n=values["top_n"],
            r_s=values["top_r_s"], r_sh=values["top_r_sh"], temp_k=temp_k,
        ),
        bottom=DiodeParams(
            j_ph=values["bot_j_ph"], j_0=values["bot_j_0"], n=values["bot_n"],
            r_s=values["bot_r_s"], r_sh=values["bot_r_sh"], temp_k=temp_k,
        ),
    )


def fit_tandem(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    *,
    kind: str = "light",
    residual_space: ResidualSpace = "auto",
    loss: str = "linear",
    penalty: float = PENALTY,
    max_nfev: int | None = None,
) -> FitResult:
    """Fit the free tandem parameters to a measured terminal (V, J) curve.

    Same contract as ``fitting.fit_diode``: never raises on optimizer failure,
    fixed parameters are copied verbatim, and the returned ``FitResult``
    carries the prefixed ``free_names`` so callers can report which of the 10
    parameters were fitted. ``FitResult.params`` is a ``TandemParams``.

    Args:
        voltage: measured terminal voltage points (V).
        current: measured current density (A/cm^2), model sign convention.
        temp_k: fixed measurement temperature (K) — never fitted.
        specs: ``{name: ParamSpec}`` from ``default_tandem_specs``.
        kind: "light" or "dark"; only used to resolve ``residual_space="auto"``.
        residual_space: "auto" | "linear" | "log".
        loss: ``least_squares`` loss ("linear", "soft_l1", "huber", ...).
        penalty: residual substituted for failed/non-finite evaluations.
        max_nfev: optional cap on function evaluations.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    space = resolve_residual_space(residual_space, kind)

    return _fit_generic(
        voltage, current, specs,
        space=space, loss=loss, penalty=penalty, max_nfev=max_nfev,
        param_order=TANDEM_PARAM_NAMES,
        unpack_params=lambda theta: unpack_tandem(theta, specs, temp_k),
        predict=lambda theta: solve_tandem_current(
            voltage, unpack_tandem(theta, specs, temp_k)
        ),
    )
