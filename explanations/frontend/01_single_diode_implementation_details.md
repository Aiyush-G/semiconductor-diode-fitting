# Underlying Explanation to the Single Diode Model

This document provides an overview into the underlying physics and how it has been implemented into the code.

*The architecture for this application allows the UI to be de-couples from the physics-logic which can be found in the src/models folder of the GitHub repository.*

Credit: I found https://www.pveducation.org/pvcdrom/solar-cell-operation/ enormously helpful.

## The Single Diode Model

The governing equation implemented in the model (implicit form, current-density normalised):


$$
J = J_{ph} - J_{0}\left(\exp\!\left(\frac{V + J R_s}{n V_t}\right) - 1\right) - \frac{V + J R_s}{R_{sh}}
$$

*Implemented in `src/models/single_diode.py`.*

Rather than iterating numerically to solve this implicit equation, `solve_current()` uses the closed-form Lambert W solution (see "Series Resistance" below):

Source: Amit Jain, Avinashi Kapoor, *Exact analytical solutions of the parameters of real solar cells using Lambert W-function*, *Solar Energy Materials and Solar Cells*.



*This equation is implicit since J appear on both sides*.

```python
def solve_current(voltage: np.ndarray, params: DiodeParams) -> np.ndarray:
    ...
    vt = thermal_voltage(temp_k)
    nvt = n * vt

    if r_s == 0:
        # Degenerate case: no series resistance, equation becomes explicit
        current = j_ph - j_0 * (np.exp(voltage / nvt) - 1) - voltage / r_sh
        return current

    # Standard closed-form (Lambert W) solution
    a = (r_s * r_sh * j_0) / (nvt * (r_s + r_sh))
    b = (r_sh * (r_s * (j_ph + j_0) + voltage)) / (nvt * (r_s + r_sh))
    w = lambertw(a * np.exp(b)).real

    current = (r_sh * (j_ph + j_0) - voltage) / (r_s + r_sh) - (nvt / r_s) * w
    return current
```
*Implemented in `src/models/single_diode.py`*

The five physical parameters below (`j_ph`, `j_0`, `n`, `r_s`, `r_sh`) are all fields on a single `DiodeParams` dataclass:

```python
@dataclass
class DiodeParams:
    j_ph: float      # Light-generated (photo) current density, A/cm^2
    j_0: float        # Diode saturation current density, A/cm^2
    n: float          # Diode ideality factor (dimensionless, typically 1-2)
    r_s: float        # Series resistance, Ohm.cm^2
    r_sh: float       # Shunt resistance, Ohm.cm^2
    temp_k: float = 298.15  # Cell temperature, Kelvin (25C default)
```
*Implemented in `src/models/single_diode.py` (`DiodeParams`, lines 78-101).*

### Reference Parameters

#### Photo-current density
When photons strike a semiconductor, they excite electrons, creating electron-hole pairs. An electrical field separates these charges, producing a current.

Photo current density is calculated by removing the dark current from the total light current. 

```python
j_ph: float  # Light-generated (photo) current density, A/cm^2
```
*Corresponds to the `j_ph` field of `DiodeParams` (`src/models/single_diode.py`) — the `J_ph` term in the governing equation above.*

#### (Reverse) Saturation current density
For a pn junction with no applied voltage, only a tiny number of carriers have enough thermal energy to diffuse across the depletion region. These thermally generated minority carriers create the reverse saturation current. 

```python
j_0: float  # Diode saturation current density, A/cm^2 - the rate at which
            # carriers recombine across the junction under thermal equilibrium
```
*Corresponds to the `j_0` field of `DiodeParams` - the `J_0` term in the governing equation above.*

#### Ideality factor (+ local)
> Source: https://www.pveducation.org/pvcdrom/solar-cell-operation/ideality-factor

Measure of how closely the diode follows the ideal diode equation - there are effects which cause deviation from this and the ideality factor describes this. 

The ideal diode equation assumes that all recombination occurs within the bulk of the device and that no recombination occurs in the junction. 

![Photodiode cross-section](https://home.sandiego.edu/~ekim/photodiode/xsect.gif)



```python
n: float  # Diode ideality factor (dimensionless, typically 1-2)
```
*Corresponds to the `n` field of `DiodeParams` (`src/models/single_diode.py`), which scales the thermal voltage (`n*Vt`) in the governing equation.*

The local ideality factor (m) defines how closely a diode’s current-voltage characteristics match a pure theoretical diode at a specific operating point. It captures the changing influence of carrier recombination mechanisms as voltage or current shifts.

$$
m = \frac{1}{V_T}\frac{dV}{d\left(\ln J\right)} = \frac{J}{V_T}\frac{dV}{dJ}
$$

*Implemented in `src/models/single_diode.py` (`local_ideality_factor`).* Here `J_rec` is the diode recombination/injection current term, i.e. the current carried by the exponential diode branch:

$$
J_{rec} = J_0\left(\exp\!\left(\frac{V + J R_s}{n V_t}\right) - 1\right)
$$


#### Series Resistance

> Series resistance in a solar cell has three causes: firstly, the movement of current through the emitter and base of the solar cell; secondly, the contact resistance between the metal contact and the silicon; and finally the resistance of the top and rear metal contacts.

```python
r_s: float  # Series resistance, Ohm.cm^2 - internal resistive losses
            # (contacts, bulk, wiring)
```
*Corresponds to the `r_s` field of `DiodeParams` (`src/models/single_diode.py`).* 

#### Shunt Resistance
Shunt resistance leads to power loss due to manufacturing defects. Where there is a low shunt resistance there is an easier path for current to flow which reduces the current through the cell which, in turn, reduces voltage from the cell. 

```python
r_sh: float  # Shunt resistance, Ohm.cm^2 - models unwanted leakage paths
```
*Corresponds to the `r_sh` field of `DiodeParams` (`src/models/single_diode.py`) — the `(V + J*Rs) / Rsh` leakage term in the governing equation.*

### Effect of Temperature

*Implemented in `src/models/temperature.py`.*

Temperature affects a solar cell in two distinct ways: it shifts the photo-current very slightly, and it shifts the saturation current density substantially. The model treats these as two independent corrections applied to a set of *reference* parameters (valid at `REFERENCE_TEMP_K = 298.15 K`, i.e. 25°C), while holding `n`, `R_s` and `R_sh` fixed. 

#### `J_0` is  temperature-sensitive

The reverse saturation current density of a pn junction is governed by the diffusion of minority carriers, and is proportional to the square of the intrinsic carrier concentration $n_i^2$:

$$
J_0 \propto n_i^2 \propto T^3 \exp\!\left(-\frac{E_g}{k_B T}\right)
$$

$n_i^2$ carries the $\exp(-E_g/k_BT)$ Boltzmann factor because generating an electron-hole pair intrinsically requires a carrier to be thermally excited across the full bandgap $E_g$ - a much larger energy barrier than the built-in potential a photo-generated carrier has to overcome. This is why $J_0$ (and hence dark current) rises roughly exponentially with temperature, while $J_{ph}$ barely moves.

Writing this relationship as a ratio between the target temperature $T$ and the reference temperature $T_{ref}$ removes the (poorly known) constant of proportionality, leaving only measurable/tabulated quantities ($E_g$, $T_{ref}$, $J_{0,ref}$):

$$
J_0(T) = J_{0,\mathrm{ref}}\left(\frac{T}{T_{\mathrm{ref}}}\right)^{3}\exp\!\left[\frac{E_g}{k_B}\left(\frac{1}{T_{\mathrm{ref}}} - \frac1T\right)\right]
$$

Note this uses $E_g/k_B$ directly (units of kelvin), **not** divided by the ideality factor `n` — the saturation current activation energy is tied to the physical bandgap of the material, not to the empirical non-ideality of a particular junction, so `n` does not appear here. (This differs from the `n·V_t` term in the main IV equation, which scales the *voltage* dependence, not the *temperature* dependence.)

#### `J_ph` 

The photo-current density depends on how many above-bandgap photons are absorbed and collected, which is a comparatively weak function of temperature. Rather than modelling this from first principles, the code uses the empirical, linear temperature coefficient $\alpha_{isc}$ (fractional, in 1/K), applied directly to $J_{ph}$:

$$
J_{ph}(T) = J_{ph,\mathrm{ref}}\left[1 + \alpha_{isc}\,(T - T_{\mathrm{ref}})\right]
$$

For crystalline silicon, $\alpha_{isc} \approx 0.0005\ \mathrm{K^{-1}}$ (+0.05 %/K), which is why this correction is small compared to the $J_0$ correction over a normal operating range (e.g. ±25°C from STC).

#### Code implementation

The temperature coefficients and bandgap are grouped into a small config object:

```python
@dataclass(frozen=True)
class TemperatureCoefficients:
    alpha_isc: float = 0.0005   # 1/K, fractional Isc coefficient
    e_g_ev: float = 1.121       # eV, silicon bandgap at Tref
```

`adjust_params_for_temperature()` takes a set of `DiodeParams` valid at `reference_temp_k`, and returns a new `DiodeParams` valid at `target_temp_k`:

```python
delta_t = target_temp_k - reference_temp_k

j_ph_new = ref_params.j_ph * (1.0 + coeffs.alpha_isc * delta_t)

temperature_ratio = target_temp_k / reference_temp_k
saturation_current_exponent = (
    coeffs.e_g_ev / BOLTZMANN_EV
    * (1.0 / reference_temp_k - 1.0 / target_temp_k)
)
j_0_new = ref_params.j_0 * temperature_ratio**3 * math.exp(saturation_current_exponent)

return DiodeParams(
    j_ph=j_ph_new, j_0=j_0_new,
    n=ref_params.n, r_s=ref_params.r_s, r_sh=ref_params.r_sh,
    temp_k=target_temp_k,
)
```
*Implemented in `src/models/temperature.py` (`TemperatureCoefficients`, `adjust_params_for_temperature`).*

`BOLTZMANN_EV = 8.617333262e-5 eV/K` (imported from `single_diode.py`) is Boltzmann's constant expressed in eV/K, matching the eV units used for `e_g_ev` — this is what lets `e_g_ev / BOLTZMANN_EV` come out in kelvin, ready to multiply against the `1/T` terms directly.

`n`, `r_s`, and `r_sh` are copied across unchanged: this is the Phase A simplifying assumption stated in the module docstring, not a physical claim that these are temperature-independent (see limitations below).


## Single Diode Fitting Overview and Limitations

### Overview

### Limitations 

### Residual Space 


