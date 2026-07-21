"""Profile likelihood: measuring identifiability without a prior.

``j_0``--``n``--``r_s`` ridge as a large *local* correlation at
one fitted point. Two optimisers stop at different places on a
flat valley and called it a tolerance difference.  This module stops calling it
that.  It builds the **profile likelihood**: for a parameter of interest, it
walks a grid of fixed values and, at *each* value, re-optimises every other
(nuisance) parameter, recording the best log-likelihood attainable there.  The
resulting curve is the prior-free instrument that turns "the data cannot tell
these apart" into a measured statement -- a confidence interval that may span
orders of magnitude, and a shape (flat within the noise, but curving as the
noise shrinks) that distinguishes *practical* from *structural* non-identifiability.

The one non-negotiable design point, and the classic mistake this module exists
to prevent: the nuisances are **re-optimised** at every grid point, not held at
their joint-MLE values.  A profile that fixes the nuisances measures a slice
through the likelihood -- a stiff, narrow curve that badly *understates* the
uncertainty.  The true profile follows the floor of the valley, and along the
``j_0``--``n`` ridge that floor is nearly level.  :func:`profile_parameter`
exposes both (``reoptimise=True`` / ``False``) precisely so the difference can be
measured; the honest one is the default.

Everything is expressed through the  likelihood
(:mod:`src.fitting.noise`) and the/`fitting.py` parameterisation, so a
profile, a maximum-likelihood fit and a Jacobian covariance all speak the same
mixed (log10 for ``j_0``/``r_sh``) coordinates and are comparable point for
point.  Two theorems make the numbers interpretable:

* **Wilks (1938).**  Under regularity, ``2[ell_hat - ell_profile(psi)]`` is
  asymptotically chi-squared with one degree of freedom, so a level-``alpha``
  confidence interval for a scalar ``psi`` is the set where that statistic falls
  below ``chi2.ppf(alpha, 1)`` (3.84 at 95%).  :func:`profile_interval`.
* **Sloppiness (Gutenkunst/Transtrum).**  The eigenvectors of ``J^T J`` at the
  optimum are the stiff and sloppy directions of the fit; the softest one is the
  ridge, and its eigenvalue relative to the stiffest is the squared condition
  number  already measured (kappa(J) = 12381 on the example light
  curve).  :func:`sloppy_spectrum`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import chi2

from src.fitting.noise import (
    NoiseLikelihood,
    mle_fit,
    negative_log_likelihood,
)
from src.fitting.uncertainty import _central_jacobian
from src.models.fitting import (
    PARAM_NAMES,
    ParamSpec,
    _make_residual,
    pack,
    resolve_residual_space,
)

# ---------------------------------------------------------------------------
# Spec surgery: fix one parameter, warm-start the rest
# ---------------------------------------------------------------------------


def _fix_parameter(
    specs: dict[str, ParamSpec], name: str, value: float
) -> dict[str, ParamSpec]:
    """Return a copy of ``specs`` with ``name`` held fixed at ``value``.

    A fixed spec is never packed, so its ``log`` flag and (now degenerate)
    bounds are irrelevant; ``unpack`` copies ``value`` verbatim.  Every other
    spec -- the nuisances -- is left exactly as given.
    """
    if name not in specs:
        raise KeyError(f"{name!r} is not among the fit specs {tuple(specs)!r}.")
    fixed = replace(
        specs[name], free=False, value=float(value),
        lower=float(value), upper=float(value),
    )
    return {**specs, name: fixed}


def _set_initial(
    specs: dict[str, ParamSpec], values: dict[str, float]
) -> dict[str, ParamSpec]:
    """Return a copy of ``specs`` with the initial guesses in ``values`` updated.

    Used to warm-start each profile step from its neighbour's solution, so the
    optimiser tracks one continuous branch of the valley floor instead of
    hopping between local minima and drawing a jagged profile.
    """
    out = dict(specs)
    for name, v in values.items():
        if name in out:
            out[name] = replace(out[name], value=float(v))
    return out


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileResult:
    """A one-dimensional profile of the negative log-likelihood.

    Attributes:
        parameter: the profiled parameter's name.
        grid: the fixed values of ``parameter`` (natural units), ascending.
        profile_nll: the minimised negative log-likelihood at each grid value
            (over the nuisances if ``reoptimised``, else at their MLE values).
        delta_2nll: ``2 * (profile_nll - baseline)``, the Wilks statistic, with
            ``baseline`` the smallest NLL seen (the joint MLE).  Compare directly
            to ``chi2.ppf(level, 1)``.
        mle_value: the joint-MLE value of ``parameter`` (the profile minimiser).
        mle_nll: the joint-MLE negative log-likelihood (the ``baseline``).
        nuisance_names: the re-optimised parameters, in ``PARAM_NAMES`` order.
        nuisance_values: ``(len(grid), len(nuisance_names))`` natural-unit values
            the nuisances took at each grid point -- the trajectory *along* the
            valley floor, which is where the ridge shows itself.
        success: per-grid-point optimiser convergence flag.
        noise_model: the likelihood the profile assumed.
        reoptimised: whether the nuisances were re-fitted (True) or frozen (False).
    """

    parameter: str
    grid: np.ndarray
    profile_nll: np.ndarray
    delta_2nll: np.ndarray
    mle_value: float
    mle_nll: float
    nuisance_names: tuple[str, ...]
    nuisance_values: np.ndarray
    success: np.ndarray
    noise_model: NoiseLikelihood
    reoptimised: bool


def profile_parameter(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    noise_model: NoiseLikelihood,
    parameter: str,
    grid: np.ndarray,
    *,
    kind: str = "light",
    reoptimise: bool = True,
    max_iter: int = 500,
) -> ProfileResult:
    """Profile ``parameter`` over ``grid``, re-optimising the nuisances.

    The nuisances are every *other* free parameter in ``specs``.  At each grid
    value the profiled parameter is pinned and the nuisances are re-fitted by
    maximum likelihood (:func:`~src.fitting.noise.mle_fit`), warm-started from the
    neighbouring solution so the profile is one continuous branch.  The joint MLE
    (all free parameters, including ``parameter``) is computed once to fix the
    baseline and, if ``reoptimise=False``, to supply the frozen nuisance values.

    Args:
        voltage, current: the sweep (V, A/cm^2).
        temp_k: fixed measurement temperature (K).
        specs: ``{name: ParamSpec}``; ``parameter`` and the nuisances are free.
        noise_model: alikelihood.
        parameter: the parameter to profile (must be a free spec).
        grid: fixed values (natural units) to profile over; length >= 2.
        kind: "light" or "dark".
        reoptimise: True (honest profile) re-fits the nuisances at every point;
            False freezes them at the joint MLE (the instructive wrong answer).
        max_iter: per-point optimiser cap.

    Returns:
        A :class:`ProfileResult`.

    Raises:
        ValueError: if ``parameter`` is not a free spec or ``grid`` is too short.
    """
    if parameter not in specs or not specs[parameter].free:
        raise ValueError(
            f"parameter {parameter!r} must be a free spec to profile it."
        )
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("grid must contain at least two values.")
    grid = np.sort(grid)

    nuisance_names = tuple(
        n for n in PARAM_NAMES
        if n != parameter and specs.get(n) is not None and specs[n].free
    )

    # 1. Joint MLE: baseline NLL, profile minimiser, frozen-nuisance reference.
    joint = mle_fit(voltage, current, temp_k, specs, noise_model, kind=kind,
                    max_iter=max_iter)
    mle_value = float(getattr(joint.params, parameter))
    mle_nll = float(joint.negative_log_likelihood)
    mle_nuisances = {n: float(getattr(joint.params, n)) for n in nuisance_names}

    profile_nll = np.full(grid.size, np.nan)
    success = np.zeros(grid.size, dtype=bool)
    nuisance_values = np.full((grid.size, len(nuisance_names)), np.nan)

    def evaluate(index: int, warm: dict[str, float]) -> dict[str, float]:
        """Fill row ``index`` and return the nuisance values to warm-start next."""
        value = grid[index]
        if not reoptimise or not nuisance_names:
            # Freeze the nuisances at the joint MLE and only move the profiled
            # parameter: a *slice* through the likelihood, not a profile.
            params = replace(joint.params, **{parameter: float(value)})
            nll = negative_log_likelihood(params, voltage, current, noise_model,
                                          kind=kind)
            fitted = dict(mle_nuisances)
            ok = bool(np.isfinite(nll))
        else:
            # Restart from both the warm neighbour and the joint MLE nuisances,
            # and keep whichever reaches the lower NLL.  A single warm start can
            # trail into a non-global local minimum far from the optimum and put
            # a spurious spike in the profile; the second seed anchors it.
            best = None
            for seed in (warm, mle_nuisances):
                local = _fix_parameter(_set_initial(specs, seed), parameter, value)
                res = mle_fit(voltage, current, temp_k, local, noise_model,
                              kind=kind, max_iter=max_iter)
                if best is None or res.negative_log_likelihood < best.negative_log_likelihood:
                    best = res
            nll = float(best.negative_log_likelihood)
            fitted = {n: float(getattr(best.params, n)) for n in nuisance_names}
            ok = bool(best.success)
        profile_nll[index] = nll
        success[index] = ok
        for j, n in enumerate(nuisance_names):
            nuisance_values[index, j] = fitted[n]
        return fitted

    # 2. Walk outward from the grid point nearest the MLE, warm-starting both ways.
    start = int(np.argmin(np.abs(grid - mle_value)))
    warm = evaluate(start, mle_nuisances)
    up = warm
    for i in range(start + 1, grid.size):
        up = evaluate(i, up)
    down = warm
    for i in range(start - 1, -1, -1):
        down = evaluate(i, down)

    baseline = float(min(mle_nll, np.nanmin(profile_nll)))
    delta_2nll = 2.0 * (profile_nll - baseline)

    return ProfileResult(
        parameter=parameter,
        grid=grid,
        profile_nll=profile_nll,
        delta_2nll=delta_2nll,
        mle_value=mle_value,
        mle_nll=mle_nll,
        nuisance_names=nuisance_names,
        nuisance_values=nuisance_values,
        success=success,
        noise_model=noise_model,
        reoptimised=bool(reoptimise) and bool(nuisance_names),
    )


# ---------------------------------------------------------------------------
# Confidence interval from the profile (Wilks' theorem)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileInterval:
    """A profile-likelihood confidence interval for one parameter.

    Attributes:
        parameter: the profiled parameter.
        level: nominal confidence level (e.g. 0.95).
        threshold: the Wilks cut ``chi2.ppf(level, 1)`` on ``delta_2nll``.
        lower, upper: interval endpoints (natural units), found by linear
            interpolation of ``delta_2nll`` across ``threshold`` on each side of
            the minimum.
        lower_capped, upper_capped: True when the profile never rose above the
            threshold on that side within the grid -- the data place *no* bound
            there and the endpoint is the grid edge (an *open* interval, the
            signature of one-sided or absent identifiability).
        factor: ``upper / lower`` when both are positive (a dimensionless width
            for a multi-decade parameter), else NaN.
    """

    parameter: str
    level: float
    threshold: float
    lower: float
    upper: float
    lower_capped: bool
    upper_capped: bool
    factor: float


def _cross(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    """Linear interpolation for the x where a segment crosses y == level."""
    if y1 == y0:
        return x1
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def profile_interval(result: ProfileResult, level: float = 0.95) -> ProfileInterval:
    """Confidence interval from a profile via Wilks' chi-squared threshold.

    Scans outward from the profile minimum to the first grid segment whose
    ``delta_2nll`` crosses ``chi2.ppf(level, 1)`` and interpolates the crossing.
    If no crossing is found on a side, that endpoint is *capped* at the grid edge
    and flagged: the data do not bound the parameter there at this level.

    Args:
        result: a :class:`ProfileResult`.
        level: confidence level in (0, 1).

    Returns:
        A :class:`ProfileInterval`.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1.")
    threshold = float(chi2.ppf(level, df=1))
    grid = result.grid
    delta = result.delta_2nll
    imin = int(np.nanargmin(delta))

    # The interpolation is only valid when the minimum grid sample sits *below*
    # the threshold.  If even the best-sampled point is above it, the interval is
    # narrower than the grid spacing (a stiff frozen slice on a coarse grid);
    # collapse that endpoint onto the minimum rather than extrapolate nonsense.
    resolved = np.isfinite(delta[imin]) and delta[imin] < threshold

    # Upper side: the first up-crossing of the threshold above imin.
    upper, upper_capped = float(grid[-1]), True
    if resolved:
        for i in range(imin, grid.size - 1):
            if np.isfinite(delta[i + 1]) and delta[i] < threshold <= delta[i + 1]:
                upper = _cross(grid[i], delta[i], grid[i + 1], delta[i + 1], threshold)
                upper_capped = False
                break
    else:
        upper, upper_capped = float(grid[imin]), False

    # Lower side: the first up-crossing of the threshold below imin.
    lower, lower_capped = float(grid[0]), True
    if resolved:
        for i in range(imin, 0, -1):
            if np.isfinite(delta[i - 1]) and delta[i] < threshold <= delta[i - 1]:
                lower = _cross(grid[i], delta[i], grid[i - 1], delta[i - 1], threshold)
                lower_capped = False
                break
    else:
        lower, lower_capped = float(grid[imin]), False

    factor = float(upper / lower) if (lower > 0 and upper > 0) else float("nan")
    return ProfileInterval(
        parameter=result.parameter,
        level=float(level),
        threshold=threshold,
        lower=float(lower),
        upper=float(upper),
        lower_capped=lower_capped,
        upper_capped=upper_capped,
        factor=factor,
    )


# ---------------------------------------------------------------------------
# The sloppy eigendirection of J^T J
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sloppiness:
    """Stiff/sloppy eigenstructure of ``J^T J`` at a fitted point.

    ``J`` is the residual Jacobian in the mixed fit coordinates (log10 for
    ``j_0``/``r_sh``).  The Gauss--Newton approximation to the likelihood
    curvature is ``J^T J / sigma^2``; its eigenvectors are the fit's principal
    axes, ordered here from *softest* (smallest eigenvalue, the sloppy ridge) to
    *stiffest*.  A softest eigenvector dominated by ``log10(j_0)`` with an ``n``
    component of the *same* sign (the two rise together, ``rho = +0.9997``) *is*
    the ridge, seen as a direction rather than a correlation.

    Attributes:
        free_names: parameter order of the eigenvector components.
        eigenvalues: ascending eigenvalues of ``J^T J`` (softest first).
        eigenvectors: columns aligned with ``eigenvalues`` (softest = column 0).
        condition_number: ``kappa(J) = sqrt(lambda_max / lambda_min)`` -- the
            Cconditioning, recovered from the eigenvalues.
        softest: the sloppy eigenvector (column 0), the flat direction.
        stiffest: the stiff eigenvector (last column), the best-determined one.
    """

    free_names: tuple[str, ...]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    softest: np.ndarray
    stiffest: np.ndarray

    def describe(self, vector: np.ndarray) -> str:
        """A compact ``+0.71 log10(j_0)  -0.70 n`` style description of a direction."""
        parts = [f"{c:+.2f} {name}" for name, c in zip(self.free_names, vector)]
        return "  ".join(parts)


def sloppy_spectrum(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    *,
    kind: str = "light",
    residual_space: str = "auto",
    relative_step: float | None = None,
) -> Sloppiness:
    """Eigen-decompose ``J^T J`` at the point encoded by ``specs``' values.

    The Jacobian is rebuilt at the fit-space point ``pack(specs)`` gives -- so
    set each free spec's ``value`` to the fitted (MLE) parameter first, exactly
    as  covariance adapter does.  Reuses the bound-aware central-
    difference Jacobian from :mod:`src.fitting.uncertainty` so the conditioning
    reported here is identical to the covariance conditioning there.

    Returns:
        A :class:`Sloppiness`.

    Raises:
        ValueError: for fewer than two free parameters (an eigendirection needs
            a plane), or a non-finite Jacobian.
    """
    space = resolve_residual_space(residual_space, kind)
    theta, lower, upper, free_names = pack(specs)
    if len(free_names) < 2:
        raise ValueError("sloppy_spectrum needs at least two free parameters.")

    residual_fn = _make_residual(voltage, current, specs, temp_k, space, penalty=np.nan)
    jacobian = _central_jacobian(residual_fn, theta, lower, upper, relative_step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian contains non-finite entries at this point.")

    gram = jacobian.T @ jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(gram)  # ascending, orthonormal
    eigenvalues = np.maximum(eigenvalues, 0.0)
    smallest = eigenvalues[0]
    largest = eigenvalues[-1]
    condition = float(np.sqrt(largest / smallest)) if smallest > 0 else float("inf")

    return Sloppiness(
        free_names=tuple(free_names),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition,
        softest=eigenvectors[:, 0],
        stiffest=eigenvectors[:, -1],
    )
