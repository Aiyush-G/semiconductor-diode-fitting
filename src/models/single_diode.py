"""
Single-diode equivalent circuit model for a solar cell.

Equation (implicit form):
    I = I_ph - I_0 * (exp((V + I*Rs) / (n*Vt)) - 1) - (V + I*Rs) / Rsh

This module solves the explicit closed-form version using the Lambert W
function (standard approach — see De Soto et al. 2006 / PVsyst docs),
rather than numerically iterating, since it's faster and avoids
convergence edge cases when scanning a full voltage sweep.

Lambert W solved following: Amit Jain, Avinashi Kapoor,
Exact analytical solutions of the parameters of real solar cells using Lambert W-function,
Solar Energy Materials and Solar Cells. 

See explanations/models for written explanation of this model. 

https://www.pvsyst.com/help/physical-models-used/pv-module-standard-one-diode-model/index.html has the experimental setup that is followed.

Example Usage: 
1. Create a DiodeParams object with required circuit parameters: 
```py
from src.models.single_diode import DiodeParams, iv_curve, key_metrics

params = DiodeParams(
    i_ph=5.0,      # Photo current in Amps
    i_0=1e-12,     # Saturation current in Amps
    n=1.3,         # Ideality factor
    r_s=0.5,       # Series resistance in Ohms
    r_sh=1000,     # Shunt resistance in Ohms
    temp_k=298.15  # Temperature in Kelvin (25°C default)
)
```

2. Then, generate IV (current-voltage curve)
```py
voltage, current = iv_curve(params, v_max=1.2, n_points=200)
```

3. Extract performance metrics
```py
metrics = key_metrics(voltage, current)
print(f"Short-circuit current (Isc): {metrics['isc']:.3f} A")
print(f"Open-circuit voltage (Voc): {metrics['voc']:.3f} V")
print(f"Max power point: {metrics['pmax']:.3f} W")
print(f"Fill factor: {metrics['fill_factor']:.1%}")
```
"""

from dataclasses import dataclass # for more easy storage of parameters

import numpy as np
from scipy.special import lambertw # This is for the analytical solution of real solar cells

# Physical constants
BOLTZMANN_EV = 8.617333262e-5  # eV/K
Q_CHARGE = 1.602176634e-19     # Coulombs
K_BOLTZMANN = 1.380649e-23     # J/K

@dataclass
class DiodeParams:
    """Physical parameters of a single-diode equivalent circuit.

    The current delivered to the external current is I = i_ph - i_recombination - i_sh

    Attributes:
        i_ph: Light-generated (photo) current, A
        i_0: Diode saturation current, A - the rate at which carriers recombine across the junction under thermal equilibrium
        n: Diode ideality factor (dimensionless, typically 1-2)
        r_s: Series resistance, Ohms - internal resistive losses (contacts, bulk, wiring)
        r_sh: Shunt resistance, Ohms - models unwanted leakage paths 
        temp_k: Cell temperature, Kelvin
    """
    i_ph: float
    i_0: float
    n: float
    r_s: float
    r_sh: float
    temp_k: float = 298.15  # 25C default

def thermal_voltage(temp_k: float) -> float:
    """Thermal voltage Vt = kT/q, in volts."""
    return (K_BOLTZMANN * temp_k) / Q_CHARGE


def solve_current(voltage: np.ndarray, params: DiodeParams) -> np.ndarray:
    """Solve for current I at each voltage point using the closed-form
    Lambert W solution to the single-diode equation.

    Args:
        voltage: array of voltage points (V)
        params: DiodeParams describing the circuit

    Returns:
        array of current values (A), same shape as `voltage`
    """
    i_ph, i_0, n, r_s, r_sh, temp_k = (
        params.i_ph, params.i_0, params.n, params.r_s, params.r_sh, params.temp_k
    )
    vt = thermal_voltage(temp_k)
    nvt = n * vt

    if r_s == 0:
        # Degenerate case: no series resistance, equation becomes explicit
        current = i_ph - i_0 * (np.exp(voltage / nvt) - 1) - voltage / r_sh
        return current

    # Standard closed-form (Lambert W) solution
    # This is from Exact analytical solutions of the parameters of real solar cells using Lambert W-function, Solar Energy Materials and Solar Cells. 
    a = (r_s * r_sh * i_0) / (nvt * (r_s + r_sh))
    b = (r_sh * (r_s * (i_ph + i_0) + voltage)) / (nvt * (r_s + r_sh))
    w = lambertw(a * np.exp(b)).real

    current = (r_sh * (i_ph + i_0) - voltage) / (r_s + r_sh) - (nvt / r_s) * w
    return current


def iv_curve(params: DiodeParams, v_max: float = 1.2, n_points: int = 200,
             dark: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate an IV curve for the given parameters.

    Args:
        params: DiodeParams describing the circuit
        v_max: maximum voltage to sweep to (V)
        n_points: number of points in the sweep
        dark: if True, sets i_ph to 0 (dark IV curve) regardless of params.i_ph

    Returns:
        (voltage_array, current_array)
    """
    voltage = np.linspace(0, v_max, n_points)
    active_params = params
    if dark: # All params are copied except for the absence of light-generated current ie. no photogeneration.
        active_params = DiodeParams(
            i_ph=0.0, i_0=params.i_0, n=params.n,
            r_s=params.r_s, r_sh=params.r_sh, temp_k=params.temp_k,
        )
    current = solve_current(voltage, active_params)
    return voltage, current


def key_metrics(voltage: np.ndarray, current: np.ndarray) -> dict:
    """Extract standard solar cell metrics from an IV curve.

    Returns dict with: isc (short-circuit current), voc (open-circuit voltage),
    pmax (max power point), vmp, imp, fill_factor.
    """
    power = voltage * current
    idx_pmax = int(np.argmax(power)) # finds the index of the largest value in the power array. 

    isc = float(np.interp(0, voltage, current)) # short-circuit
    # Voc: voltage where current crosses zero - current is zero since there is no external load
    if np.any(current <= 0):
        voc = float(np.interp(0, current[::-1], voltage[::-1]))
    else:
        voc = float(voltage[-1])

    pmax = float(power[idx_pmax])
    vmp = float(voltage[idx_pmax])
    imp = float(current[idx_pmax])
    fill_factor = pmax / (isc * voc) if (isc > 0 and voc > 0) else float("nan")

    return {
        "isc": isc, "voc": voc, "pmax": pmax,
        "vmp": vmp, "imp": imp, "fill_factor": fill_factor,
    }