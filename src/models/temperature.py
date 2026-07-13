"""
Temperature dependence of single-diode parameters.

Phase A implements a simplified PVsyst-style, temperature-only adjustment
of the single-diode parameters at fixed irradiance.

All current quantities are area-normalised current densities (A/cm^2),
following the PV Lighthouse convention used in single_diode.py.

The reference parameters are assumed to be valid at REFERENCE_TEMP_K.
At a new cell temperature:

    J_ph(T) = J_ph,ref * [1 + alpha_isc * (T - T_ref)]

    J_0(T) = J_0,ref * (T / T_ref)^3
             * exp[
                 E_g / k_B
                 * (1 / T_ref - 1 / T)
             ]

Note: the saturation-current activation exponent uses E_g / k_B with no
ideality factor, following the standard De Soto / Shockley form.

The diode ideality factor n, series resistance R_s, and shunt resistance
R_sh are held constant in this Phase A approximation.

Notes:
- alpha_isc is stored as a fractional coefficient in 1/K.
  For example, 0.0005 1/K means +0.05 %/K.
- PVsyst conventionally expresses the short-circuit current coefficient
  as an absolute coefficient in A/K. At fixed irradiance, the two forms
  are related approximately by:

      mu_isc = alpha_isc * I_ph,ref

- This module adjusts electrical parameters after the cell temperature
  is known. It does not calculate cell temperature from weather data.

References:
- PVsyst standard one-diode model:
  https://www.pvsyst.com/help/physical-models-used/pv-module-standard-one-diode-model/index.html

- Sandia PVPMC De Soto five-parameter model:
  https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/single-diode-equivalent-circuit-models/de-soto-five-parameter-module-model/
"""

from dataclasses import dataclass
import math

from src.models.single_diode import BOLTZMANN_EV, DiodeParams


REFERENCE_TEMP_K = 298.15  # 25 °C, STC reference temperature


@dataclass(frozen=True)
class TemperatureCoefficients:
    """Temperature coefficients referenced to 25 °C.

    Attributes:
        alpha_isc:
            Fractional temperature coefficient of short-circuit current,
            in 1/K.

            Example:
                0.0005 1/K = +0.05 %/K.

            At fixed irradiance, this coefficient is applied to J_ph
            because J_ph is approximately proportional to J_sc.

        e_g_ev:
            Semiconductor bandgap energy at the reference temperature,
            in electronvolts.

            The default value of 1.121 eV is representative of crystalline
            silicon near room temperature.
    """

    alpha_isc: float = 0.0005
    e_g_ev: float = 1.121 # Si

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_isc):
            raise ValueError("alpha_isc must be finite.")

        if not math.isfinite(self.e_g_ev) or self.e_g_ev <= 0.0:
            raise ValueError("e_g_ev must be a positive finite value.")


def adjust_params_for_temperature(
    ref_params: DiodeParams,
    target_temp_k: float,
    coeffs: TemperatureCoefficients,
    reference_temp_k: float = REFERENCE_TEMP_K,
) -> DiodeParams:
    
    """Adjust reference single-diode parameters to a new cell temperature.

    This function implements a simplified PVsyst-style temperature
    adjustment at fixed irradiance.

    The adjusted photo-current density is

        J_ph(T) = J_ph,ref *
                  [1 + alpha_isc * (T - T_ref)]

    and the adjusted reverse saturation current density is

        J_0(T) = J_0,ref * (T / T_ref)^3
                 * exp[
                     E_g / k_B
                     * (1 / T_ref - 1 / T)
                 ]

    where:
        T:
            Target cell temperature, K.

        T_ref:
            Reference cell temperature, K.

        E_g:
            Semiconductor bandgap, eV.

        k_B:
            Boltzmann constant, eV/K.

    The Phase A approximation keeps n, R_s, and R_sh constant.

    Args:
        ref_params:
            Single-diode parameters at ``reference_temp_k``.

        target_temp_k:
            Target cell temperature in kelvin.

        coeffs:
            Temperature coefficients and material bandgap.

        reference_temp_k:
            Temperature at which ``ref_params`` are valid.
            Defaults to 298.15 K.

    Returns:
        A new ``DiodeParams`` instance valid at ``target_temp_k``.

    Raises:
        ValueError:
            If temperatures or diode parameters are non-physical, or if
            the photocurrent correction produces a negative value.
    """
    _validate_inputs(
        ref_params=ref_params,
        target_temp_k=target_temp_k,
        reference_temp_k=reference_temp_k,
    )

    delta_t = target_temp_k - reference_temp_k

    # Fractional form of the Jsc temperature correction.
    #
    # PVsyst normally writes:
    #     J_ph(T) = J_ph,ref + mu_isc * (T - T_ref)
    #
    # Here:
    #     mu_isc = alpha_isc * J_ph,ref
    j_ph_new = ref_params.j_ph * (
        1.0 + coeffs.alpha_isc * delta_t
    )

    if j_ph_new < 0.0:
        raise ValueError(
            "The temperature correction produced a negative photocurrent. "
            "Check alpha_isc and the requested temperature range."
        )

    temperature_ratio = target_temp_k / reference_temp_k

    saturation_current_exponent = (
        coeffs.e_g_ev
        / BOLTZMANN_EV
        * (
            1.0 / reference_temp_k
            - 1.0 / target_temp_k
        )
    )

    try:
        j_0_new = (
            ref_params.j_0
            * temperature_ratio**3
            * math.exp(saturation_current_exponent)
        )
    except OverflowError as exc:
        raise ValueError(
            "The saturation-current temperature correction overflowed. "
            "Check the temperature and bandgap."
        ) from exc

    if not math.isfinite(j_0_new):
        raise ValueError(
            "The adjusted saturation current is not finite."
        )

    return DiodeParams(
        j_ph=j_ph_new,
        j_0=j_0_new,
        n=ref_params.n,
        r_s=ref_params.r_s,
        r_sh=ref_params.r_sh,
        temp_k=target_temp_k,
    )


def _validate_inputs(
    ref_params: DiodeParams,
    target_temp_k: float,
    reference_temp_k: float,
) -> None:
    """Validate temperatures and reference diode parameters."""
    if not math.isfinite(target_temp_k) or target_temp_k <= 0.0:
        raise ValueError(
            "target_temp_k must be a positive finite temperature in kelvin."
        )

    if not math.isfinite(reference_temp_k) or reference_temp_k <= 0.0:
        raise ValueError(
            "reference_temp_k must be a positive finite temperature "
            "in kelvin."
        )

    if not math.isfinite(ref_params.j_ph) or ref_params.j_ph < 0.0:
        raise ValueError(
            "Reference photo-current density j_ph must be finite and non-negative."
        )

    if not math.isfinite(ref_params.j_0) or ref_params.j_0 <= 0.0:
        raise ValueError(
            "Reference saturation current density j_0 must be positive and finite."
        )

    if not math.isfinite(ref_params.n) or ref_params.n <= 0.0:
        raise ValueError(
            "Diode ideality factor n must be positive and finite."
        )

    if not math.isfinite(ref_params.r_s) or ref_params.r_s < 0.0:
        raise ValueError(
            "Series resistance r_s must be finite and non-negative."
        )

    if not math.isfinite(ref_params.r_sh) or ref_params.r_sh <= 0.0:
        raise ValueError(
            "Shunt resistance r_sh must be positive and finite."
        )