"""
Two-terminal tandem solar cell model: two single-diode sub-cells in series.

In a monolithic 2-terminal tandem the sub-cells are wired in series, so the
same current density J flows through both (the current-matching constraint)
and the terminal voltage is the sum of the sub-cell voltages:

    V_tandem(J) = V_top(J) + V_bottom(J)

Each sub-cell is the existing single-diode equivalent circuit
(``single_diode.DiodeParams``), so the tandem has 10 parameters. Units follow
the same area-normalised PV Lighthouse convention (A/cm^2, Ohm.cm^2).

Rather than solving the coupled circuit numerically, this module uses the
exact closed-form *inverse* of the single-diode equation — voltage as a
function of current — which is also a Lambert W expression (Jain & Kapoor,
the same reference as ``single_diode.solve_current``):

    A    = j_ph + j_0 - J
    V(J) = A*r_sh - J*r_s - n*Vt * W[ (j_0*r_sh)/(n*Vt) * exp(A*r_sh/(n*Vt)) ]

The tandem JV curve is then generated parametrically in J and resampled onto
a voltage grid, so downstream consumers (``key_metrics``,
``local_ideality_factor``, plotting) work unchanged.

Numerical note
--------------
The W argument's exponent A*r_sh/(n*Vt) is ~1e3-1e4 for realistic shunt
resistances, far beyond float64's exp overflow at ~709. ``solve_voltage``
therefore forms the argument only in log space: scipy's ``lambertw`` for
moderate exponents, and for large ones ``_log_lambertw_exp_large`` computes
ln(W(e^x)) directly (Newton on the fixed point ln(w) = ln(x - ln(w))), which
also sidesteps the catastrophic cancellation between A*r_sh and n*Vt*W.

Limitations: no reverse-bias breakdown model (same as the single-diode page),
and no luminescent coupling between sub-cells.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw

from src.models.single_diode import DiodeParams, thermal_voltage

# Above this, exp(x) risks losing precision / overflowing inside lambertw, so
# the asymptotic branch takes over (well inside float64's exp range of ~709).
_LOG_ARG_DIRECT_MAX = 500.0


@dataclass
class TandemParams:
    """Parameters of a 2-terminal tandem: two single-diode sub-cells in series.

    Attributes:
        top: wide-bandgap sub-cell (e.g. perovskite) facing the light.
        bottom: narrow-bandgap sub-cell (e.g. silicon).

    Both sub-cells carry their own ``temp_k``; they describe one physical
    device, so the temperatures should be kept equal.
    """

    top: DiodeParams
    bottom: DiodeParams


def _log_lambertw_exp_large(x: np.ndarray) -> np.ndarray:
    """Evaluate ln(W(exp(x))) for large x, where exp(x) would overflow.

    W(e^x) satisfies w + ln(w) = x. Writing delta = ln(w) gives the fixed
    point delta = ln(x - delta), solved here by Newton iteration from the
    asymptotic seed delta = ln(x). Returning ln(w) rather than w lets the
    caller avoid the catastrophic cancellation in a*r_sh - n*Vt*w (both terms
    are huge and nearly equal for large shunt resistances).
    """
    delta = np.log(x)
    for _ in range(4):
        w = x - delta
        delta -= (delta - np.log(w)) / (1.0 + 1.0 / w)
    return delta


def solve_voltage(current: np.ndarray, params: DiodeParams) -> np.ndarray:
    """Exact voltage at each current density for one sub-cell (inverse model).

    This is the closed-form Lambert W inverse of the single-diode equation
    (see module docstring), the counterpart of ``single_diode.solve_current``.
    Unlike the forward solve, no ``r_s == 0`` special case is needed — the
    expression never divides by ``r_s``.

    Args:
        current: array of current-density points (A/cm^2)
        params: DiodeParams describing the sub-cell

    Returns:
        array of voltages (V), same shape as ``current``
    """
    current = np.asarray(current, dtype=float)
    j_ph, j_0, n, r_s, r_sh, temp_k = (
        params.j_ph, params.j_0, params.n, params.r_s, params.r_sh, params.temp_k
    )
    nvt = n * thermal_voltage(temp_k)

    a = j_ph + j_0 - current
    ln_c = np.log(j_0 * r_sh / nvt)
    x = ln_c + a * r_sh / nvt

    voltage = np.empty_like(a)
    small = x <= _LOG_ARG_DIRECT_MAX
    if np.any(small):
        # exp(x) is representable and a*r_sh is at most ~n*Vt*(500 + |ln c|)
        # volts, so the direct form loses no meaningful precision here.
        w = lambertw(np.exp(x[small])).real
        voltage[small] = a[small] * r_sh - nvt * w
    big = ~small
    if np.any(big):
        # a*r_sh - n*Vt*w reduces algebraically to n*Vt*(ln(w) - ln(c)), which
        # avoids subtracting two huge, nearly-equal numbers.
        voltage[big] = nvt * (_log_lambertw_exp_large(x[big]) - ln_c)
    return voltage - current * r_s


def tandem_voltage(current: np.ndarray, params: TandemParams) -> np.ndarray:
    """Terminal voltage of the series stack at each (shared) current density.

    Current matching: both sub-cells carry the same J, so the terminal voltage
    is simply the sum of the sub-cell voltages at that J.
    """
    return solve_voltage(current, params.top) + solve_voltage(current, params.bottom)


def _dark_subcell(params: DiodeParams) -> DiodeParams:
    """Copy of a sub-cell with the photocurrent removed (dark condition)."""
    return DiodeParams(
        j_ph=0.0, j_0=params.j_0, n=params.n,
        r_s=params.r_s, r_sh=params.r_sh, temp_k=params.temp_k,
    )


def _parametric_table(
    params: TandemParams, v_lo: float, v_hi: float, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact model points (V, J) spanning [v_lo, v_hi], sorted ascending in V.

    V(J) is strictly decreasing (each sub-cell has dV/dJ < 0), so the current
    range is bracketed by extending J downward until V >= v_hi and upward
    until V <= v_lo. A coarse parametric sweep is then refined twice by
    inverse-interpolating a uniform voltage grid back to current values and
    re-evaluating the model exactly there, so the returned points are exact
    and near-uniform in voltage (a uniform J grid would waste almost all its
    resolution outside the photocurrent plateau).
    """
    j_scale = max(min(params.top.j_ph, params.bottom.j_ph), 1e-6)

    # Current at the high-voltage end (J <= 0; J = 0 gives exactly Voc).
    j_at_vhi = 0.0
    step = 0.01 * j_scale
    for _ in range(80):
        if float(tandem_voltage(np.array([j_at_vhi]), params)[0]) >= v_hi:
            break
        j_at_vhi -= step
        step *= 2.0

    # Current at the low-voltage end (past the limiting photocurrent the
    # shunt-dominated V(J) drops steeply, so doubling converges quickly).
    j_at_vlo = 1.05 * j_scale
    step = 0.05 * j_scale
    for _ in range(80):
        if float(tandem_voltage(np.array([j_at_vlo]), params)[0]) <= v_lo:
            break
        j_at_vlo += step
        step *= 2.0

    v_grid = np.linspace(v_lo, v_hi, n_points)
    j_samples = np.linspace(j_at_vhi, j_at_vlo, max(n_points, 200))
    for _ in range(2):
        v_samples = tandem_voltage(j_samples, params)
        order = np.argsort(v_samples)
        j_refined = np.interp(v_grid, v_samples[order], j_samples[order])
        # Keep the brackets so the table always spans the full voltage range.
        j_samples = np.union1d(j_refined, [j_at_vhi, j_at_vlo])

    v_samples = tandem_voltage(j_samples, params)
    order = np.argsort(v_samples)
    v_sorted = v_samples[order]
    j_sorted = j_samples[order]

    # Drop the far bracket points (they can overshoot the window by volts in
    # the steep shunt region) while keeping the refined near-endpoint points.
    margin = max(1e-6, 1e-4 * (v_hi - v_lo))
    keep = (v_sorted >= v_lo - margin) & (v_sorted <= v_hi + margin)
    return v_sorted[keep], j_sorted[keep]


def tandem_iv_curve(
    params: TandemParams, n_points: int = 400, dark: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the terminal JV curve of the tandem stack.

    The curve spans 0 V to the tandem Voc (which is exact: Voc = V(J=0) =
    Voc_top + Voc_bottom). Points are exact model evaluations, near-uniform in
    voltage, with the zero-current point included so ``key_metrics`` finds Voc
    without extrapolating.

    Args:
        params: TandemParams describing the stack
        n_points: approximate number of points in the returned curve
        dark: if True, sets both sub-cell photocurrents to 0 while keeping the
            voltage span of the illuminated curve (mirrors ``iv_curve``'s dark
            overlay behaviour)

    Returns:
        (voltage_array, current_density_array), voltage ascending
    """
    voc = float(tandem_voltage(np.array([0.0]), params)[0])
    v_hi = voc if voc > 0.1 else 1.0

    active = params
    if dark:
        active = TandemParams(
            top=_dark_subcell(params.top), bottom=_dark_subcell(params.bottom)
        )
    return _parametric_table(active, 0.0, v_hi, n_points)


def tandem_subcell_curves(
    params: TandemParams, n_points: int = 400
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Sub-cell JV curves on the tandem's shared current grid.

    Because the stack is series-connected, both sub-cells are evaluated at the
    same current densities as the terminal curve; their voltages sum to the
    tandem voltage point-by-point. Useful for overlaying which sub-cell limits
    the current.

    Returns:
        [(label, voltage, current), ...] for the top and bottom sub-cells.
    """
    _, current = tandem_iv_curve(params, n_points)
    return [
        ("Top cell", solve_voltage(current, params.top), current),
        ("Bottom cell", solve_voltage(current, params.bottom), current),
    ]


def solve_tandem_current(
    voltage: np.ndarray, params: TandemParams, grid_points: int = 800
) -> np.ndarray:
    """Terminal current density at arbitrary voltages (forward model for fits).

    Builds one dense exact parametric (V, J) table covering the requested
    voltage range and interpolates V -> J. V(J) is strictly monotonic, so the
    interpolation is well-posed; one vectorised table per call is far cheaper
    than per-point root finding inside ``least_squares``, and the residual
    interpolation error is negligible against measurement noise.

    Args:
        voltage: voltage points (V), any order
        params: TandemParams describing the stack
        grid_points: density of the internal parametric table

    Returns:
        array of current densities (A/cm^2), same shape as ``voltage``
    """
    voltage = np.asarray(voltage, dtype=float)
    voc = float(tandem_voltage(np.array([0.0]), params)[0])
    # Pad the window so measured extremes interpolate between exact table
    # points rather than clamping to the table edge.
    v_lo = min(0.0, float(voltage.min())) - 0.02
    v_hi = max(voc, float(voltage.max())) + 0.02
    if v_hi - v_lo < 1e-6:
        v_hi = v_lo + 0.1

    v_table, j_table = _parametric_table(params, v_lo, v_hi, grid_points)
    return np.interp(voltage, v_table, j_table)
