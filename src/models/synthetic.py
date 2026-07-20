"""Seeded synthetic single-diode J-V data with explicit measurement noise.

The deterministic forward model defines the noise-free current density.  A
noise object then defines the stochastic measurement step.  Keeping those two
parts separate makes every synthetic dataset auditable: it carries the truth,
the exact curve, the realised perturbation, the noise specification, and the
random seed used to generate it.

Only noise *generation* lives here.  Probability densities and likelihoods are
implemented once in the fitting layer in Chapter 4 so deterministic and
Bayesian inference can share them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np

from src.models.single_diode import DiodeParams, solve_current


DatasetKind = Literal["light", "dark"]


def _validate_nonnegative_finite(value: float, name: str) -> None:
    """Require a finite, non-negative scalar noise parameter."""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative; received {value!r}.")


@dataclass(frozen=True)
class NoNoise:
    """An exact forward-model dataset with no measurement perturbation."""


@dataclass(frozen=True)
class GaussianNoise:
    """Additive normal current noise.

    Attributes:
        sigma_a_per_cm2: standard deviation in A/cm^2.
    """

    sigma_a_per_cm2: float

    def __post_init__(self) -> None:
        _validate_nonnegative_finite(self.sigma_a_per_cm2, "sigma_a_per_cm2")


@dataclass(frozen=True)
class LogNormalNoise:
    """Multiplicative log-normal noise on current magnitude.

    If ``epsilon ~ Normal(0, sigma_ln^2)``, the measurement is
    ``sign(J) * |J| * exp(epsilon)``.  Thus the exact magnitude is the median,
    the sign is preserved, and ``sigma_ln`` is the standard deviation of the
    natural-log current ratio.  This is intended for non-zero dark currents;
    an exact zero remains zero.

    Attributes:
        sigma_ln: dimensionless natural-log standard deviation.
    """

    sigma_ln: float

    def __post_init__(self) -> None:
        _validate_nonnegative_finite(self.sigma_ln, "sigma_ln")


@dataclass(frozen=True)
class StudentTNoise:
    """Additive heavy-tailed Student-t current noise.

    Attributes:
        scale_a_per_cm2: Student-t scale in A/cm^2.  This is not the standard
            deviation: for ``degrees_of_freedom > 2``, the standard deviation
            is ``scale * sqrt(df / (df - 2))``.
        degrees_of_freedom: positive tail parameter.  Smaller values give
            heavier tails; the mean is undefined at df <= 1 and the variance
            is infinite at df <= 2.
    """

    scale_a_per_cm2: float
    degrees_of_freedom: float = 4.0

    def __post_init__(self) -> None:
        _validate_nonnegative_finite(self.scale_a_per_cm2, "scale_a_per_cm2")
        if not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 0:
            raise ValueError(
                "degrees_of_freedom must be finite and positive; "
                f"received {self.degrees_of_freedom!r}."
            )


NoiseModel: TypeAlias = NoNoise | GaussianNoise | LogNormalNoise | StudentTNoise


@dataclass(frozen=True)
class SyntheticDataset:
    """A realised synthetic J-V measurement and its complete provenance.

    Attributes:
        voltage: voltage design points, V.
        current: noisy measured current density, A/cm^2.
        exact_current: deterministic forward-model current density, A/cm^2.
        noise: realised additive difference ``current - exact_current``, A/cm^2.
        params: active truth used by the forward model (``j_ph=0`` for dark).
        kind: ``"light"`` or ``"dark"``.
        noise_model: immutable noise specification used for this realisation.
        seed: seed passed to NumPy's random generator, or ``None``.
    """

    voltage: np.ndarray
    current: np.ndarray
    exact_current: np.ndarray
    noise: np.ndarray
    params: DiodeParams
    kind: DatasetKind
    noise_model: NoiseModel
    seed: int | None

    @property
    def n_points(self) -> int:
        """Number of voltage/current observations."""
        return int(self.voltage.size)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    """Copy an array and make the returned provenance immutable."""
    copied = np.array(values, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _draw_noisy_current(
    exact_current: np.ndarray,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a supported noise-generating process to an exact current array."""
    if isinstance(noise_model, NoNoise):
        measured = np.array(exact_current, copy=True)
    elif isinstance(noise_model, GaussianNoise):
        measured = exact_current + rng.normal(
            loc=0.0,
            scale=noise_model.sigma_a_per_cm2,
            size=exact_current.shape,
        )
    elif isinstance(noise_model, LogNormalNoise):
        log_error = rng.normal(
            loc=0.0,
            scale=noise_model.sigma_ln,
            size=exact_current.shape,
        )
        measured = np.sign(exact_current) * np.abs(exact_current) * np.exp(log_error)
    elif isinstance(noise_model, StudentTNoise):
        measured = exact_current + noise_model.scale_a_per_cm2 * rng.standard_t(
            df=noise_model.degrees_of_freedom,
            size=exact_current.shape,
        )
    else:
        raise TypeError(
            "noise_model must be NoNoise, GaussianNoise, LogNormalNoise, "
            f"or StudentTNoise; received {type(noise_model).__name__}."
        )

    if not np.all(np.isfinite(measured)):
        raise ValueError(
            "The requested noise realisation produced non-finite current values; "
            "reduce the noise scale."
        )
    return measured


def generate_synthetic(
    params: DiodeParams,
    voltage: np.ndarray,
    *,
    kind: DatasetKind = "light",
    noise_model: NoiseModel = NoNoise(),
    seed: int | None = None,
) -> SyntheticDataset:
    """Generate a synthetic single-diode current measurement.

    Args:
        params: physical truth for the single-diode forward model.
        voltage: one-dimensional finite voltage design points, V.  Repeated
            values are allowed so repeat measurements at one voltage can be
            simulated directly.
        kind: ``"light"`` or ``"dark"``.  Dark generation copies all truth
            parameters but sets ``j_ph`` to exactly zero by construction.
        noise_model: one of the immutable noise specifications defined here.
        seed: seed for ``numpy.random.default_rng``.  Equal inputs and equal
            integer seeds produce bitwise-identical datasets.

    Returns:
        A :class:`SyntheticDataset` containing both truth and realised data.

    Raises:
        ValueError: for an unknown kind or invalid voltage array.
        TypeError: for an unsupported noise specification.
    """
    if kind not in ("light", "dark"):
        raise ValueError(f"Unknown kind {kind!r}; expected 'light' or 'dark'.")

    voltage_array = np.asarray(voltage, dtype=float)
    if voltage_array.ndim != 1 or voltage_array.size == 0:
        raise ValueError("voltage must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(voltage_array)):
        raise ValueError("voltage must contain only finite values")

    active_params = replace(params, j_ph=0.0) if kind == "dark" else replace(params)
    exact_current = np.asarray(solve_current(voltage_array, active_params), dtype=float)
    if not np.all(np.isfinite(exact_current)):
        raise ValueError("The forward model produced non-finite current values.")

    rng = np.random.default_rng(seed)
    measured_current = _draw_noisy_current(exact_current, noise_model, rng)
    realised_noise = measured_current - exact_current

    return SyntheticDataset(
        voltage=_readonly_copy(voltage_array),
        current=_readonly_copy(measured_current),
        exact_current=_readonly_copy(exact_current),
        noise=_readonly_copy(realised_noise),
        params=active_params,
        kind=kind,
        noise_model=noise_model,
        seed=seed,
    )
