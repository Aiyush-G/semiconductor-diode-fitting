"""Joint light/dark profiling and posterior-vs-profile dual reports.

The Bayesian and frequentist arms must score the same forward model and the same
noise assumptions.  This module supplies the deterministic joint likelihood,
its maximum-likelihood fit and profile, then overlays a posterior marginal on
that profile in the parameter's fitted coordinate.  A profile likelihood is not
a probability density; its normalisation here is only a shape comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from src.fitting.noise import (
    AbsoluteGaussian,
    LogNormalLikelihood,
    negative_log_likelihood,
)
from src.fitting.profile import ProfileInterval, profile_interval
from src.models.fitting import LOG_PARAMS, PARAM_NAMES, ParamSpec, pack, unpack
from src.models.single_diode import DiodeParams


@dataclass(frozen=True)
class JointMLEResult:
    """Maximum-likelihood result for one light and one dark sweep."""

    params: DiodeParams
    negative_log_likelihood: float
    success: bool
    message: str


@dataclass(frozen=True)
class JointProfileResult:
    """One-dimensional profile of the shared light-plus-dark likelihood."""

    parameter: str
    grid: np.ndarray
    profile_nll: np.ndarray
    delta_2nll: np.ndarray
    mle_value: float
    mle_nll: float
    nuisance_names: tuple[str, ...]
    nuisance_values: np.ndarray
    success: np.ndarray
    light_noise: AbsoluteGaussian
    dark_noise: LogNormalLikelihood
    reoptimised: bool


class ComparisonFlag(str, Enum):
    """Non-exclusive interpretations exposed by a dual report."""

    AGREEMENT = "agreement"
    PROFILE_OPEN = "profile_open"
    PRIOR_SHIFT = "prior_shift"
    BOUND_CONTACT = "bound_contact"
    MULTIMODAL = "multimodal"
    OPTIMISER_FAILURE = "optimiser_failure"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class DualReport:
    """Numerical overlay of a posterior marginal and a profile likelihood."""

    parameter: str
    coordinate: Literal["linear", "log10"]
    grid: np.ndarray
    coordinate_grid: np.ndarray
    profile_shape: np.ndarray
    posterior_density: np.ndarray
    profile_interval: ProfileInterval
    posterior_interval: tuple[float, float]
    posterior_median: float
    profile_mode: float
    posterior_mode: float
    interval_overlap: float
    js_divergence: float
    posterior_mass_in_grid: float
    n_posterior_modes: int
    flags: tuple[ComparisonFlag, ...]


def joint_negative_log_likelihood(
    params: DiodeParams,
    light_voltage: np.ndarray,
    light_current: np.ndarray,
    dark_voltage: np.ndarray,
    dark_current: np.ndarray,
    light_noise: AbsoluteGaussian,
    dark_noise: LogNormalLikelihood,
) -> float:
    """Sum of light and dark negative log-likelihoods for shared parameters."""
    light = negative_log_likelihood(
        params, light_voltage, light_current, light_noise, kind="light"
    )
    dark = negative_log_likelihood(
        params, dark_voltage, dark_current, dark_noise, kind="dark"
    )
    return float(light + dark)


def joint_mle(
    light_voltage: np.ndarray,
    light_current: np.ndarray,
    dark_voltage: np.ndarray,
    dark_current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    light_noise: AbsoluteGaussian,
    dark_noise: LogNormalLikelihood,
    *,
    max_iter: int = 1000,
) -> JointMLEResult:
    """Fit shared diode parameters by the joint light-plus-dark likelihood."""
    theta0, lower, upper, _ = pack(specs)

    def objective(theta: np.ndarray) -> float:
        value = joint_negative_log_likelihood(
            unpack(theta, specs, temp_k),
            light_voltage,
            light_current,
            dark_voltage,
            dark_current,
            light_noise,
            dark_noise,
        )
        return value if np.isfinite(value) else 1e12

    if theta0.size == 0:
        params = unpack(theta0, specs, temp_k)
        return JointMLEResult(params, objective(theta0), True, "All parameters fixed.")

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    # A very narrow likelihood can make L-BFGS-B stop on a line-search warning
    # even when it is already near the floor.  Powell is a slower but independent
    # derivative-free second opinion.  Keep the lower objective, but never hide
    # the fact that neither optimiser declared convergence.
    if not result.success:
        fallback = minimize(
            objective,
            result.x,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={"maxiter": int(max_iter), "xtol": 1e-9, "ftol": 1e-12},
        )
        if fallback.fun <= result.fun or fallback.success:
            result = fallback
    params = unpack(result.x, specs, temp_k)
    return JointMLEResult(
        params=params,
        negative_log_likelihood=float(objective(result.x)),
        success=bool(result.success),
        message=str(result.message),
    )


def _fix(specs: dict[str, ParamSpec], name: str, value: float) -> dict[str, ParamSpec]:
    fixed = replace(
        specs[name],
        free=False,
        value=float(value),
        lower=float(value),
        upper=float(value),
    )
    return {**specs, name: fixed}


def _initialise(
    specs: dict[str, ParamSpec], values: dict[str, float]
) -> dict[str, ParamSpec]:
    out = dict(specs)
    for name, value in values.items():
        if name in out:
            out[name] = replace(out[name], value=float(value))
    return out


def joint_profile_parameter(
    light_voltage: np.ndarray,
    light_current: np.ndarray,
    dark_voltage: np.ndarray,
    dark_current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    light_noise: AbsoluteGaussian,
    dark_noise: LogNormalLikelihood,
    parameter: str,
    grid: np.ndarray,
    *,
    max_iter: int = 1000,
) -> JointProfileResult:
    """Profile one parameter while re-optimising all shared nuisances."""
    if parameter not in specs or not specs[parameter].free:
        raise ValueError(f"parameter {parameter!r} must be free")
    grid = np.sort(np.asarray(grid, dtype=float).reshape(-1))
    if grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must contain at least two finite values")

    joint = joint_mle(
        light_voltage,
        light_current,
        dark_voltage,
        dark_current,
        temp_k,
        specs,
        light_noise,
        dark_noise,
        max_iter=max_iter,
    )
    nuisance_names = tuple(
        name for name in PARAM_NAMES if name != parameter and specs[name].free
    )
    mle_values = {name: float(getattr(joint.params, name)) for name in nuisance_names}
    profile_nll = np.full(grid.size, np.nan)
    nuisance_values = np.full((grid.size, len(nuisance_names)), np.nan)
    success = np.zeros(grid.size, dtype=bool)

    def evaluate(index: int, warm: dict[str, float]) -> dict[str, float]:
        best: JointMLEResult | None = None
        for seed in (warm, mle_values):
            local = _fix(_initialise(specs, seed), parameter, float(grid[index]))
            candidate = joint_mle(
                light_voltage,
                light_current,
                dark_voltage,
                dark_current,
                temp_k,
                local,
                light_noise,
                dark_noise,
                max_iter=max_iter,
            )
            if best is None or candidate.negative_log_likelihood < best.negative_log_likelihood:
                best = candidate
        assert best is not None
        profile_nll[index] = best.negative_log_likelihood
        success[index] = best.success
        fitted = {name: float(getattr(best.params, name)) for name in nuisance_names}
        for column, name in enumerate(nuisance_names):
            nuisance_values[index, column] = fitted[name]
        return fitted

    mle_value = float(getattr(joint.params, parameter))
    start = int(np.argmin(np.abs(grid - mle_value)))
    centre = evaluate(start, mle_values)
    warm = centre
    for index in range(start + 1, grid.size):
        warm = evaluate(index, warm)
    warm = centre
    for index in range(start - 1, -1, -1):
        warm = evaluate(index, warm)

    baseline = float(min(joint.negative_log_likelihood, np.nanmin(profile_nll)))
    return JointProfileResult(
        parameter=parameter,
        grid=grid,
        profile_nll=profile_nll,
        delta_2nll=2.0 * (profile_nll - baseline),
        mle_value=mle_value,
        mle_nll=float(joint.negative_log_likelihood),
        nuisance_names=nuisance_names,
        nuisance_values=nuisance_values,
        success=success,
        light_noise=light_noise,
        dark_noise=dark_noise,
        reoptimised=bool(nuisance_names),
    )


def _normalise(density: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=float)
    area = float(np.trapezoid(density, coordinate))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("density has no finite positive area on the grid")
    return density / area


def _interval_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    smaller = min(first[1] - first[0], second[1] - second[0])
    return float(intersection / smaller) if smaller > 0 else 0.0


def compare_profile_posterior(
    profile: JointProfileResult,
    posterior_draws: np.ndarray,
    *,
    level: float = 0.95,
    coordinate: Literal["auto", "linear", "log10"] = "auto",
    physical_bounds: tuple[float | None, float | None] | None = None,
) -> DualReport:
    """Overlay posterior density and relative profile likelihood quantitatively."""
    draws = np.asarray(posterior_draws, dtype=float).reshape(-1)
    draws = draws[np.isfinite(draws)]
    if draws.size < 20 or np.std(draws) == 0:
        raise ValueError("posterior_draws need at least 20 non-constant finite values")
    resolved = "log10" if coordinate == "auto" and profile.parameter in LOG_PARAMS else coordinate
    if resolved == "auto":
        resolved = "linear"
    if resolved not in ("linear", "log10"):
        raise ValueError("coordinate must be auto, linear, or log10")
    if resolved == "log10" and (np.any(profile.grid <= 0) or np.any(draws <= 0)):
        raise ValueError("log10 comparison requires strictly positive grid and draws")

    transform = np.log10 if resolved == "log10" else lambda value: np.asarray(value)
    inverse = (lambda value: 10.0**value) if resolved == "log10" else lambda value: value
    x_grid = np.asarray(transform(profile.grid), dtype=float)
    x_draws = np.asarray(transform(draws), dtype=float)

    relative = np.exp(-0.5 * np.maximum(profile.delta_2nll, 0.0))
    relative[~np.isfinite(relative)] = 0.0
    profile_shape = _normalise(relative, x_grid)
    posterior_density = _normalise(gaussian_kde(x_draws)(x_grid), x_grid)

    midpoint = 0.5 * (profile_shape + posterior_density)
    eps = np.finfo(float).tiny
    js = 0.5 * np.trapezoid(
        profile_shape * np.log((profile_shape + eps) / (midpoint + eps)), x_grid
    ) + 0.5 * np.trapezoid(
        posterior_density * np.log((posterior_density + eps) / (midpoint + eps)), x_grid
    )

    alpha = 100.0 * (1.0 - level) / 2.0
    posterior_interval = tuple(float(x) for x in np.percentile(draws, [alpha, 100.0 - alpha]))
    prof_interval = profile_interval(profile, level)  # structural duck type
    posterior_mode_x = float(x_grid[int(np.argmax(posterior_density))])
    profile_mode_x = float(x_grid[int(np.argmax(profile_shape))])
    prominence = 0.1 * float(np.max(posterior_density))
    peaks, _ = find_peaks(posterior_density, prominence=prominence)
    n_modes = max(1, int(peaks.size))

    in_grid = float(
        np.mean((draws >= profile.grid[0]) & (draws <= profile.grid[-1]))
    )
    flags: list[ComparisonFlag] = []
    is_open = prof_interval.lower_capped or prof_interval.upper_capped
    if is_open:
        flags.append(ComparisonFlag.PROFILE_OPEN)
    if physical_bounds is not None:
        lower_bound, upper_bound = physical_bounds
        lower_touch = (
            lower_bound is not None
            and prof_interval.lower_capped
            and np.isclose(profile.grid[0], lower_bound, rtol=1e-6, atol=0.0)
        )
        upper_touch = (
            upper_bound is not None
            and prof_interval.upper_capped
            and np.isclose(profile.grid[-1], upper_bound, rtol=1e-6, atol=0.0)
        )
        if lower_touch or upper_touch:
            flags.append(ComparisonFlag.BOUND_CONTACT)
    if n_modes > 1:
        flags.append(ComparisonFlag.MULTIMODAL)
    if not np.all(profile.success):
        flags.append(ComparisonFlag.OPTIMISER_FAILURE)

    profile_interval_x = tuple(float(x) for x in transform([prof_interval.lower, prof_interval.upper]))
    posterior_interval_x = tuple(float(x) for x in transform(posterior_interval))
    overlap = _interval_overlap(profile_interval_x, posterior_interval_x)
    median = float(np.median(draws))
    prior_shift = (
        not is_open
        and (
            median < prof_interval.lower
            or median > prof_interval.upper
            or float(js) > 0.10
        )
    )
    if prior_shift:
        flags.append(ComparisonFlag.PRIOR_SHIFT)
    if not flags:
        if float(js) <= 0.10 and overlap >= 0.5 and in_grid >= level:
            flags.append(ComparisonFlag.AGREEMENT)
        else:
            flags.append(ComparisonFlag.CONFLICT)

    return DualReport(
        parameter=profile.parameter,
        coordinate=resolved,
        grid=profile.grid,
        coordinate_grid=x_grid,
        profile_shape=profile_shape,
        posterior_density=posterior_density,
        profile_interval=prof_interval,
        posterior_interval=posterior_interval,
        posterior_median=median,
        profile_mode=float(inverse(profile_mode_x)),
        posterior_mode=float(inverse(posterior_mode_x)),
        interval_overlap=overlap,
        js_divergence=float(js),
        posterior_mass_in_grid=in_grid,
        n_posterior_modes=n_modes,
        flags=tuple(flags),
    )
