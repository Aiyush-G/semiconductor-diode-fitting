"""
Single-diode equivalent circuit model for a solar cell.

Equation (implicit form):
    I = I_ph - I_0 * (exp((V + I*Rs) / (n*Vt)) - 1) - (V + I*Rs) / Rsh

This module solves the explicit closed-form version using the Lambert W
function (standard approach — see De Soto et al. 2006 / PVsyst docs),
rather than numerically iterating, since it's faster and avoids
convergence edge cases when scanning a full voltage sweep.
"""

from dataclasses import dataclass # for more easy storage of parameters

import numpy as np
from scipy.special import lambertw

# Physical constants
BOLTZMANN_EV = 8.617333262e-5  # eV/K
Q_CHARGE = 1.602176634e-19     # Coulombs
K_BOLTZMANN = 1.380649e-23     # J/K