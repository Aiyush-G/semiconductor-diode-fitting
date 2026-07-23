"""Local uncertainty estimates for nonlinear least-squares diode fits.

The optimiser works in a mixed coordinate system: ``j_0`` and ``r_sh`` are
represented by their base-10 logarithms, while the other parameters remain in
natural units.  The Jacobian and its covariance therefore live in *fit space*.
This module reports that covariance and transforms it to natural parameter
units with the first-order delta method.

These estimates describe only the local quadratic approximation at the fitted
point.  A large condition number, a rank-deficient Jacobian, an active bound,
or a curved ridge is a reason to use profile likelihood rather than to trust a
small-looking symmetric standard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.fitting import (
    LOG_PARAMS,
    FitResult,
    ParamSpec,
    _forward,
    _make_residual,
    pack,
    resolve_residual_space,
)


@dataclass(frozen=True)
class JacobianUncertainty:
    """Local covariance diagnostics derived from a residual Jacobian.

    ``covariance_fit`` and ``standard_errors_fit`` use the coordinates seen by
    the optimiser (for example, decades for ``log10(j_0)``).  ``covariance``
    and ``standard_errors`` use natural parameter units.  The correlation
    matrix is the same in both because the delta-method transformation is a
    positive diagonal rescaling.
    """

    free_names: tuple[str, ...]
    estimates: np.ndarray
    covariance_fit: np.ndarray
    standard_errors_fit: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    correlation: np.ndarray
    residual_variance: float
    degrees_of_freedom: int
    rank: int
    condition_number: float
    singular_values: np.ndarray
    residual_space: str

    @property
    def full_rank(self) -> bool:
        """Whether every free-parameter direction has non-zero sensitivity."""
        return self.rank == len(self.free_names)


def _empty_uncertainty(n_points: int, residuals: np.ndarray, space: str) -> JacobianUncertainty:
    """Return the well-defined zero-parameter result for an all-fixed model."""
    shape = (0, 0)
    variance = float(np.dot(residuals, residuals) / n_points) if n_points else float("nan")
    return JacobianUncertainty(
        free_names=(),
        estimates=np.empty(0),
        covariance_fit=np.empty(shape),
        standard_errors_fit=np.empty(0),
        covariance=np.empty(shape),
        standard_errors=np.empty(0),
        correlation=np.empty(shape),
        residual_variance=variance,
        degrees_of_freedom=n_points,
        rank=0,
        condition_number=1.0,
        singular_values=np.empty(0),
        residual_space=space,
    )


def covariance_from_jacobian(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    estimates: np.ndarray,
    free_names: tuple[str, ...],
    *,
    log_names: frozenset[str] = LOG_PARAMS,
    residual_space: str = "linear",
    rcond: float | None = None,
) -> JacobianUncertainty:
    """Compute local fit- and natural-space covariance from a Jacobian.

    Args:
        jacobian: matrix ``dr_i / dtheta_k`` in optimiser coordinates.
        residuals: residual vector in the same residual space as ``jacobian``.
        estimates: fitted parameter values in natural units, in ``free_names``
            order.
        free_names: names of the free parameters, matching Jacobian columns.
        log_names: parameters represented as ``log10(value)`` in fit space.
        residual_space: label carried into the result (``linear`` or ``log``).
        rcond: relative singular-value threshold.  The default is the standard
            matrix-size-scaled float64 threshold.

    Returns:
        A :class:`JacobianUncertainty` with rank and conditioning diagnostics.

    Raises:
        ValueError: if array dimensions disagree or ``N <= p``.

    Notes:
        For a rank-deficient Jacobian the inverse information matrix does not
        exist.  Returning a Moore-Penrose pseudoinverse would make invisible
        directions look falsely precise, so both covariance matrices and their
        derived standard errors/correlations are returned as ``NaN`` instead.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    estimates = np.asarray(estimates, dtype=float).reshape(-1)
    free_names = tuple(free_names)

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    n_points, n_params = jacobian.shape
    if residuals.shape != (n_points,):
        raise ValueError("residuals must have one entry per Jacobian row")
    if estimates.shape != (n_params,) or len(free_names) != n_params:
        raise ValueError("estimates and free_names must match the Jacobian columns")
    if n_params == 0:
        return _empty_uncertainty(n_points, residuals, residual_space)
    if n_points <= n_params:
        raise ValueError(
            f"Jacobian covariance requires N > p; received N={n_points}, p={n_params}."
        )
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(residuals)):
        raise ValueError("jacobian and residuals must contain only finite values")

    _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    threshold = (
        float(rcond) * singular_values[0]
        if rcond is not None
        else np.finfo(float).eps * max(jacobian.shape) * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    dof = n_points - n_params
    residual_variance = float(np.dot(residuals, residuals) / dof)

    if rank < n_params:
        nan_matrix = np.full((n_params, n_params), np.nan)
        covariance_fit = nan_matrix.copy()
        covariance = nan_matrix.copy()
        standard_errors_fit = np.full(n_params, np.nan)
        standard_errors = np.full(n_params, np.nan)
        correlation = nan_matrix.copy()
        condition_number = float("inf")
    else:
        # SVD is more stable than explicitly forming and inverting J.T @ J.
        inverse_information = (vt.T / singular_values**2) @ vt
        covariance_fit = residual_variance * inverse_information
        standard_errors_fit = np.sqrt(np.maximum(np.diag(covariance_fit), 0.0))

        derivatives = np.ones(n_params)
        for index, name in enumerate(free_names):
            if name in log_names:
                derivatives[index] = np.log(10.0) * estimates[index]
        transform = np.diag(derivatives)
        covariance = transform @ covariance_fit @ transform
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

        denominator = np.outer(standard_errors_fit, standard_errors_fit)
        correlation = np.divide(
            covariance_fit,
            denominator,
            out=np.full_like(covariance_fit, np.nan),
            where=denominator > 0,
        )
        finite_diagonal = standard_errors_fit > 0
        correlation[finite_diagonal, finite_diagonal] = 1.0
        condition_number = float(singular_values[0] / singular_values[-1])

    return JacobianUncertainty(
        free_names=free_names,
        estimates=estimates,
        covariance_fit=covariance_fit,
        standard_errors_fit=standard_errors_fit,
        covariance=covariance,
        standard_errors=standard_errors,
        correlation=correlation,
        residual_variance=residual_variance,
        degrees_of_freedom=dof,
        rank=rank,
        condition_number=condition_number,
        singular_values=singular_values,
        residual_space=residual_space,
    )


def _central_jacobian(
    function,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    relative_step: float | None,
) -> np.ndarray:
    """Bound-aware central finite-difference Jacobian in fit coordinates."""
    theta = np.asarray(theta, dtype=float)
    base = np.asarray(function(theta), dtype=float)
    jacobian = np.empty((base.size, theta.size), dtype=float)
    rel = np.cbrt(np.finfo(float).eps) if relative_step is None else float(relative_step)
    if not np.isfinite(rel) or rel <= 0:
        raise ValueError("relative_step must be a finite positive number")

    for column in range(theta.size):
        step = rel * max(1.0, abs(theta[column]))
        room_lower = theta[column] - lower[column]
        room_upper = upper[column] - theta[column]
        central_step = min(step, room_lower, room_upper)

        if central_step > 0:
            plus = theta.copy()
            minus = theta.copy()
            plus[column] += central_step
            minus[column] -= central_step
            jacobian[:, column] = (
                np.asarray(function(plus)) - np.asarray(function(minus))
            ) / (2.0 * central_step)
            continue

        # At an active bound, use the feasible one-sided difference.
        if room_upper > 0:
            one_sided_step = min(step, room_upper)
            shifted = theta.copy()
            shifted[column] += one_sided_step
            jacobian[:, column] = (np.asarray(function(shifted)) - base) / one_sided_step
        elif room_lower > 0:
            one_sided_step = min(step, room_lower)
            shifted = theta.copy()
            shifted[column] -= one_sided_step
            jacobian[:, column] = (base - np.asarray(function(shifted))) / one_sided_step
        else:
            # A zero-width bound cannot be a meaningful free direction.
            jacobian[:, column] = 0.0
    return jacobian


def estimate_fit_uncertainty(
    fit: FitResult,
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    *,
    kind: str = "light",
    residual_space: str = "auto",
    relative_step: float | None = None,
) -> JacobianUncertainty:
    """Re-evaluate a completed diode fit and estimate its local uncertainty.

    This adapter leaves the backwards-compatible fitter untouched.  It rebuilds
    the residual Jacobian at ``fit.params`` in the exact mixed fit coordinates
    described by ``specs``, then calls :func:`covariance_from_jacobian`.

    The calculation assumes ordinary least squares.  A covariance obtained
    after a robust loss is only a local curvature heuristic and should not be
    interpreted as a calibrated sampling uncertainty.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.ndim != 1 or current.shape != voltage.shape:
        raise ValueError("voltage and current must be one-dimensional arrays of equal length")

    space = resolve_residual_space(residual_space, kind)
    _, lower, upper, free_names = pack(specs)
    fitted_values = {
        name: float(getattr(fit.params, name))
        for name in free_names
    }
    estimates = np.asarray([fitted_values[name] for name in free_names])
    theta = np.asarray([
        _forward(fitted_values[name], specs[name].log)
        for name in free_names
    ])
    residual_function = _make_residual(
        voltage, current, specs, temp_k, space, penalty=np.nan
    )
    residuals = np.asarray(residual_function(theta), dtype=float)

    if not free_names:
        return _empty_uncertainty(voltage.size, residuals, space)

    jacobian = _central_jacobian(
        residual_function, theta, lower, upper, relative_step
    )
    return covariance_from_jacobian(
        jacobian,
        residuals,
        estimates,
        free_names,
        residual_space=space,
    )
