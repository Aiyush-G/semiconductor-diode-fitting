"""
Single-diode equivalent circuit model for a solar cell.

Units follow the PV Lighthouse equivalent-circuit convention, i.e. everything
is area-normalised (per unit cell area) so the cell area cancels out:

    J    - current density,           A/cm^2   (displayed as mA/cm^2)
    J_0  - saturation current density, A/cm^2
    R_s  - series resistance,         Ohm.cm^2
    R_sh - shunt resistance,          Ohm.cm^2

Equation (implicit form), in current density:
    J = J_ph - J_0 * (exp((V + J*Rs) / (n*Vt)) - 1) - (V + J*Rs) / Rsh

The formula is unit-agnostic: with J in A/cm^2 and Rs/Rsh in Ohm.cm^2, the
product J*Rs is in volts and every term stays dimensionally consistent.

This module solves the explicit closed-form version using the Lambert W
function (standard approach — see De Soto et al. 2006 / PVsyst docs),
rather than numerically iterating, since it's faster and avoids
convergence edge cases when scanning a full voltage sweep.

Lambert W solved following: Amit Jain, Avinashi Kapoor,
Exact analytical solutions of the parameters of real solar cells using Lambert W-function,
Solar Energy Materials and Solar Cells.

See explanations/models for written explanation of this model.

https://www.pvsyst.com/help/physical-models-used/pv-module-standard-one-diode-model/index.html has the experimental setup that is followed.

Cross-check reference (single-diode mode, 2nd diode disabled J_02 = 0):
https://www.pvlighthouse.com.au/equivalent-circuit

Example Usage:
1. Create a DiodeParams object with required circuit parameters:
```py
from src.models.single_diode import DiodeParams, iv_curve, key_metrics

params = DiodeParams(
    j_ph=40e-3,    # Photo-current density in A/cm^2 (40 mA/cm^2)
    j_0=1e-13,     # Saturation current density in A/cm^2
    n=1.0,         # Ideality factor
    r_s=0.5,       # Series resistance in Ohm.cm^2
    r_sh=1000,     # Shunt resistance in Ohm.cm^2
    temp_k=298.15  # Temperature in Kelvin (25°C default)
)
```

2. Then, generate JV (current-density vs voltage) curve
```py
voltage, current = iv_curve(params, n_points=200)
```

3. Extract performance metrics
```py
metrics = key_metrics(voltage, current)
print(f"Short-circuit current density (Jsc): {metrics['jsc'] * 1e3:.2f} mA/cm^2")
print(f"Open-circuit voltage (Voc): {metrics['voc']:.3f} V")
print(f"Max power point: {metrics['pmax'] * 1e3:.2f} mW/cm^2")
print(f"Fill factor: {metrics['fill_factor']:.1%}")
print(f"Efficiency: {metrics['efficiency']:.1%}")
```
"""

from dataclasses import dataclass # for more easy storage of parameters

import numpy as np
from scipy.special import lambertw # This is for the analytical solution of real solar cells

# Physical constants
BOLTZMANN_EV = 8.617333262e-5  # eV/K
Q_CHARGE = 1.602176634e-19     # Coulombs
K_BOLTZMANN = 1.380649e-23     # J/K

# Reference input irradiance for efficiency: 1 sun = AM1.5G = 100 mW/cm^2.
P_IN_ONE_SUN_W_PER_CM2 = 0.1   # W/cm^2

@dataclass
class DiodeParams:
    """Physical parameters of a single-diode equivalent circuit.

    All quantities are area-normalised (per unit cell area), following the
    PV Lighthouse convention, so cell area cancels out of the model.

    The current density delivered to the external circuit is
    J = j_ph - j_recombination - j_sh

    Attributes:
        j_ph: Light-generated (photo) current density, A/cm^2
        j_0: Diode saturation current density, A/cm^2 - the rate at which carriers recombine across the junction under thermal equilibrium
        n: Diode ideality factor (dimensionless, typically 1-2)
        r_s: Series resistance, Ohm.cm^2 - internal resistive losses (contacts, bulk, wiring)
        r_sh: Shunt resistance, Ohm.cm^2 - models unwanted leakage paths
        temp_k: Cell temperature, Kelvin
    """
    j_ph: float
    j_0: float
    n: float
    r_s: float
    r_sh: float
    temp_k: float = 298.15  # 25C default

def thermal_voltage(temp_k: float) -> float:
    """Thermal voltage Vt = kT/q, in volts."""
    return (K_BOLTZMANN * temp_k) / Q_CHARGE


def solve_current(voltage: np.ndarray, params: DiodeParams) -> np.ndarray:
    """Solve for current density J at each voltage point using the closed-form
    Lambert W solution to the single-diode equation.

    Args:
        voltage: array of voltage points (V)
        params: DiodeParams describing the circuit

    Returns:
        array of current-density values (A/cm^2), same shape as `voltage`
    """
    j_ph, j_0, n, r_s, r_sh, temp_k = (
        params.j_ph, params.j_0, params.n, params.r_s, params.r_sh, params.temp_k
    )
    vt = thermal_voltage(temp_k)
    nvt = n * vt

    if r_s == 0:
        # Degenerate case: no series resistance, equation becomes explicit
        current = j_ph - j_0 * (np.exp(voltage / nvt) - 1) - voltage / r_sh
        return current

    # Standard closed-form (Lambert W) solution
    # This is from Exact analytical solutions of the parameters of real solar cells using Lambert W-function, Solar Energy Materials and Solar Cells.
    a = (r_s * r_sh * j_0) / (nvt * (r_s + r_sh))
    b = (r_sh * (r_s * (j_ph + j_0) + voltage)) / (nvt * (r_s + r_sh))
    w = lambertw(a * np.exp(b)).real

    current = (r_sh * (j_ph + j_0) - voltage) / (r_s + r_sh) - (nvt / r_s) * w
    return current


def iv_curve(params: DiodeParams, v_max: float = 1.2, n_points: int = 200,
             dark: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate a JV (current-density vs voltage) curve for the given parameters.

    The voltage sweep auto-extends beyond ``v_max`` if the current has not yet
    crossed zero, so that Voc is always captured (important for wide-bandgap /
    perovskite cells whose Voc can exceed the default 1.2 V and would otherwise
    be truncated, corrupting Voc/FF/efficiency).

    Args:
        params: DiodeParams describing the circuit
        v_max: initial maximum voltage to sweep to (V); extended if needed
        n_points: number of points in the sweep
        dark: if True, sets j_ph to 0 (dark JV curve) regardless of params.j_ph

    Returns:
        (voltage_array, current_density_array)
    """
    active_params = params
    if dark: # All params are copied except for the absence of light-generated current ie. no photogeneration.
        active_params = DiodeParams(
            j_ph=0.0, j_0=params.j_0, n=params.n,
            r_s=params.r_s, r_sh=params.r_sh, temp_k=params.temp_k,
        )

    # Extend the sweep until the current crosses zero (i.e. we pass Voc), so the
    # curve always spans the full first quadrant. Capped to avoid runaway.
    upper = v_max
    for _ in range(6):
        voltage = np.linspace(0, upper, n_points)
        current = solve_current(voltage, active_params)
        if dark or current[-1] <= 0:
            break
        upper *= 1.5
    return voltage, current


def local_ideality_factor(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    j_ph: float = 0.0,
    j_floor: float = 1e-9,
) -> np.ndarray:
    """Local (voltage-dependent) diode ideality factor m(V).

    Defined from the slope of the semilog *recombination* current:

        J_rec(V) = J_ph - J(V)          (the current flowing into the diode)
        m(V)     = (1 / Vt) * dV / d(ln|J_rec|)

    Using J_rec rather than the raw terminal current is what makes m physically
    meaningful for both curves: it increases exponentially with V (J_rec ~ J_0
    exp(V / n Vt)) so m sits near the diode ideality factor n in the exponential
    region, and departs where series/shunt resistance dominate. For the dark
    curve J_ph = 0, so J_rec reduces to |J_dark|; for the light curve J_ph is the
    photocurrent (a Suns-Voc / pseudo-dark style analysis).

    Args:
        voltage: voltage points (V), assumed monotonically increasing
        current: terminal current density (A/cm^2), same shape as ``voltage``
        temp_k: cell temperature (K), used for the thermal voltage Vt
        j_ph: photocurrent density (A/cm^2) used to form J_rec = J_ph - J. Pass 0
            for a dark curve (its current already encodes zero photocurrent).
        j_floor: |J_rec| values below this (A/cm^2) are treated as invalid; where
            J_rec -> 0 (e.g. near Jsc on the light curve) ln|J_rec| is singular.

    Returns:
        array of m values, same shape as ``voltage``. Points where |J_rec| <
        j_floor or where the numeric derivative is non-finite are set to NaN so
        plots leave gaps there rather than drawing spurious spikes.
    """
    vt = thermal_voltage(temp_k)
    j = np.abs(j_ph - current)
    mask = j > j_floor

    # ln|J| is only evaluated where J is safely above the floor; elsewhere it is
    # left as NaN so np.gradient propagates the gap into the derivative.
    ln_j = np.full_like(j, np.nan, dtype=float)
    np.log(j, out=ln_j, where=mask)

    dlnj_dv = np.gradient(ln_j, voltage)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 1.0 / (vt * dlnj_dv)

    m[~np.isfinite(m)] = np.nan
    return m


def key_metrics(voltage: np.ndarray, current: np.ndarray) -> dict:
    """Extract standard solar cell metrics from a JV curve.

    Current is a density (A/cm^2), so powers are densities (W/cm^2) and the
    efficiency is referenced to a 1-sun input of 100 mW/cm^2.

    Returns dict with: jsc (short-circuit current density, A/cm^2),
    voc (open-circuit voltage, V), pmax (max power density, W/cm^2),
    vmp, jmp, fill_factor, efficiency.
    """
    power = voltage * current
    idx_pmax = int(np.argmax(power)) # finds the index of the largest value in the power array.

    jsc = float(np.interp(0, voltage, current)) # short-circuit current density
    # Voc: voltage where current crosses zero - current is zero since there is no external load
    if np.any(current <= 0):
        voc = float(np.interp(0, current[::-1], voltage[::-1]))
    else:
        voc = float(voltage[-1])

    pmax = float(power[idx_pmax])
    vmp = float(voltage[idx_pmax])
    jmp = float(current[idx_pmax])
    fill_factor = pmax / (jsc * voc) if (jsc > 0 and voc > 0) else float("nan")
    efficiency = pmax / P_IN_ONE_SUN_W_PER_CM2

    return {
        "jsc": jsc, "voc": voc, "pmax": pmax,
        "vmp": vmp, "jmp": jmp, "fill_factor": fill_factor,
        "efficiency": efficiency,
    }
