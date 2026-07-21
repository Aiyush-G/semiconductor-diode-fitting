"""Measurement-noise models, likelihoods, and maximum-likelihood fitting.

Chapter 3's :mod:`src.models.synthetic` *generates* noisy J-V data from a
declared truth.  This module owns the matching half: the probability densities
that say how plausible a measurement is, the log-likelihood of a full sweep
given a parameter vector, and the maximum-likelihood fit that follows from it.
Both halves are written once and shared, so the deterministic fits here and the
NumPyro models in Part II never disagree about what "the noise" means.

The central identity taught in Chapter 4 is proved in code below
(:func:`gaussian_negloglike`): with a *constant* Gaussian scale, the negative
log-likelihood is an affine, strictly increasing function of the sum of squared
residuals.  Minimising it is therefore identical to ordinary least squares --
every least-squares fit is a Gaussian maximum-likelihood fit with a silent,
constant, additive noise assumption.

Four noise models cover the instruments we meet:

* :class:`AbsoluteGaussian` -- one standard deviation for the whole sweep
  (a current-mode source-meter near full scale); reproduces linear-residual
  least squares.
* :class:`RelativeGaussian` -- standard deviation proportional to the reading
  (a percent-of-value meter, or a multi-decade dark sweep); the small-noise
  limit of Chapter 3's :class:`~src.models.synthetic.LogNormalNoise`.
* :class:`FloorRelativeGaussian` -- a noise floor added in quadrature to a
  percent-of-value term, the realistic source-meter specification.
* :class:`StudentTLikelihood` -- heavy tails that survive occasional outliers
  (arcing, cosmic hits, a bumped probe); reproduces Chapter 3's
  :class:`~src.models.synthetic.StudentTNoise`.

:class:`LogNormalLikelihood` completes the correspondence with Chapter 3's
log-normal generator and makes precise the claim, first met in Chapter 2, that
fitting dark data in log-residual space *is* assuming multiplicative noise.

Densities are plain NumPy and match, term for term, the ``log_prob`` methods of
``numpyro.distributions.Normal`` and ``numpyro.distributions.StudentT`` so that
Chapters 7 and 14 reuse the scale construction verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.models.fitting import ParamSpec, pack, unpack
from src.models.single_diode import DiodeParams, solve_current

# |J| floor (A/cm^2) shared with ``fitting.py``: relative scales and log
# magnitudes must never see an exact zero (e.g. the light curve at V_oc).
CURRENT_FLOOR = 1e-9

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Elementwise log densities (match numpyro.distributions .log_prob term-for-term)
# ---------------------------------------------------------------------------


def normal_logpdf(x: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Elementwise log density of ``Normal(loc, scale)`` evaluated at ``x``.

    Identical to ``numpyro.distributions.Normal(loc, scale).log_prob(x)``.
    """
    x = np.asarray(x, dtype=float)
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    z = (x - loc) / scale
    return -0.5 * z * z - np.log(scale) - 0.5 * _LOG_2PI


def student_t_logpdf(
    x: np.ndarray, loc: np.ndarray, scale: np.ndarray, df: float
) -> np.ndarray:
    """Elementwise log density of a location-scale Student-t.

    Matches ``numpyro.distributions.StudentT(df, loc, scale).log_prob(x)``.  For
    ``df -> inf`` this tends to :func:`normal_logpdf`; the heavy tails at small
    ``df`` are what make the likelihood robust to outliers.
    """
    from scipy.special import gammaln  # local import keeps the module light

    x = np.asarray(x, dtype=float)
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    if not np.isfinite(df) or df <= 0:
        raise ValueError("df must be finite and positive")
    z = (x - loc) / scale
    normaliser = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    return normaliser - 0.5 * (df + 1.0) * np.log1p(z * z / df)


# ---------------------------------------------------------------------------
# Noise models: each maps a model current mu to a per-point log density
# ---------------------------------------------------------------------------


def _safe_magnitude(mu: np.ndarray) -> np.ndarray:
    """|mu| clamped to ``CURRENT_FLOOR`` so relative scales stay positive."""
    return np.maximum(np.abs(np.asarray(mu, dtype=float)), CURRENT_FLOOR)


def _validate_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive; received {value!r}.")


@dataclass(frozen=True)
class AbsoluteGaussian:
    """Homoscedastic additive normal noise: one ``sigma`` for the whole sweep.

    Its negative log-likelihood is an affine function of the sum of squared
    residuals, so maximum likelihood here is ordinary least squares.

    Attributes:
        sigma: standard deviation in A/cm^2.
    """

    sigma: float

    def __post_init__(self) -> None:
        _validate_positive(self.sigma, "sigma")

    def scale(self, mu: np.ndarray) -> np.ndarray:
        """Per-point standard deviation (constant, shape of ``mu``)."""
        return np.full(np.shape(mu), float(self.sigma))

    def log_prob(self, observed: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return normal_logpdf(observed, mu, self.scale(mu))


@dataclass(frozen=True)
class RelativeGaussian:
    """Additive normal noise whose scale tracks the reading: ``sigma_rel * |mu|``.

    A percent-of-value meter, and the small-``sigma`` limit of Chapter 3's
    multiplicative :class:`~src.models.synthetic.LogNormalNoise`.  Because the
    scale grows with the prediction, every decade of a dark sweep is weighted
    roughly equally -- the likelihood counterpart of a log-residual fit.

    Attributes:
        sigma_rel: dimensionless relative standard deviation.
    """

    sigma_rel: float

    def __post_init__(self) -> None:
        _validate_positive(self.sigma_rel, "sigma_rel")

    def scale(self, mu: np.ndarray) -> np.ndarray:
        return float(self.sigma_rel) * _safe_magnitude(mu)

    def log_prob(self, observed: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return normal_logpdf(observed, mu, self.scale(mu))


@dataclass(frozen=True)
class FloorRelativeGaussian:
    """A noise floor added in quadrature to a percent-of-value term.

    ``sigma(mu) = sqrt(sigma_floor**2 + (sigma_rel * |mu|)**2)`` -- the honest
    source-meter model.  Near J_sc the floor dominates; near the large dark
    currents the relative term does.  Reduces to :class:`AbsoluteGaussian` as
    ``sigma_rel -> 0`` and to :class:`RelativeGaussian` as ``sigma_floor -> 0``.

    Attributes:
        sigma_floor: additive floor in A/cm^2.
        sigma_rel: dimensionless relative term.
    """

    sigma_floor: float
    sigma_rel: float

    def __post_init__(self) -> None:
        _validate_positive(self.sigma_floor, "sigma_floor")
        _validate_positive(self.sigma_rel, "sigma_rel")

    def scale(self, mu: np.ndarray) -> np.ndarray:
        floor = float(self.sigma_floor)
        rel = float(self.sigma_rel) * _safe_magnitude(mu)
        return np.sqrt(floor * floor + rel * rel)

    def log_prob(self, observed: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return normal_logpdf(observed, mu, self.scale(mu))


@dataclass(frozen=True)
class StudentTLikelihood:
    """Heavy-tailed additive noise: the robust counterpart of a Gaussian.

    A handful of gross outliers (arcing, a cosmic hit, a bumped probe) that
    would drag a Gaussian fit contribute only ``~log|residual|`` here, so the
    fit ignores them.  The generative twin is Chapter 3's
    :class:`~src.models.synthetic.StudentTNoise`.

    Attributes:
        scale_a_per_cm2: Student-t scale in A/cm^2 (not the standard deviation:
            for ``df > 2`` the SD is ``scale * sqrt(df / (df - 2))``).
        degrees_of_freedom: tail parameter; small = heavy tails.
    """

    scale_a_per_cm2: float
    degrees_of_freedom: float = 4.0

    def __post_init__(self) -> None:
        _validate_positive(self.scale_a_per_cm2, "scale_a_per_cm2")
        _validate_positive(self.degrees_of_freedom, "degrees_of_freedom")

    def scale(self, mu: np.ndarray) -> np.ndarray:
        return np.full(np.shape(mu), float(self.scale_a_per_cm2))

    def log_prob(self, observed: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return student_t_logpdf(
            observed, mu, self.scale(mu), float(self.degrees_of_freedom)
        )


@dataclass(frozen=True)
class LogNormalLikelihood:
    """Multiplicative log-normal noise on current magnitude.

    The exact magnitude is the median and ``sigma_ln`` is the standard deviation
    of ``ln(|J_measured| / |J_model|)``.  The density of the observed magnitude
    ``y`` is ``Normal(ln|y|; ln|mu|, sigma_ln)`` with a ``-ln|y|`` change-of-
    variables term.  That Jacobian is independent of the parameters, so the
    maximum-likelihood fit minimises the sum of squared *log* residuals -- this
    is exactly what fitting dark data in ``residual_space="log"`` assumes.
    The generative twin is Chapter 3's
    :class:`~src.models.synthetic.LogNormalNoise`.

    Attributes:
        sigma_ln: dimensionless natural-log standard deviation.
    """

    sigma_ln: float

    def __post_init__(self) -> None:
        _validate_positive(self.sigma_ln, "sigma_ln")

    def log_prob(self, observed: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = _safe_magnitude(observed)
        median = _safe_magnitude(mu)
        gaussian = normal_logpdf(np.log(y), np.log(median), float(self.sigma_ln))
        return gaussian - np.log(y)  # change of variables ln|y| -> |y|


# A noise model is any object exposing ``log_prob(observed, mu) -> array``.
NoiseLikelihood = (
    AbsoluteGaussian
    | RelativeGaussian
    | FloorRelativeGaussian
    | StudentTLikelihood
    | LogNormalLikelihood
)


# ---------------------------------------------------------------------------
# Likelihood of a full sweep
# ---------------------------------------------------------------------------


def _active_params(params: DiodeParams, kind: str) -> DiodeParams:
    """Return the truth actually driving the forward model (dark forces j_ph=0)."""
    if kind == "dark":
        from dataclasses import replace

        return replace(params, j_ph=0.0)
    if kind != "light":
        raise ValueError(f"Unknown kind {kind!r}; expected 'light' or 'dark'.")
    return params


def log_likelihood(
    params: DiodeParams,
    voltage: np.ndarray,
    current: np.ndarray,
    noise_model: NoiseLikelihood,
    *,
    kind: str = "light",
) -> float:
    """Total log-likelihood ``log p(current | params, noise_model)``.

    The forward model supplies the mean current at each voltage; the noise model
    supplies the per-point density.  Points are assumed conditionally
    independent given the parameters, so the sweep log-likelihood is the sum.

    Unlike the least-squares residual in ``fitting.py`` this has no ``PENALTY``
    branch: a genuinely non-finite forward evaluation yields ``-inf`` (an
    impossible measurement) rather than a finite cliff, keeping the surface
    smooth wherever it is finite -- the book's no-penalty-in-likelihoods rule.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1:
        raise ValueError("voltage and current must be 1-D arrays of equal length")

    mu = solve_current(voltage, _active_params(params, kind))
    if not np.all(np.isfinite(mu)):
        return float("-inf")
    logp = noise_model.log_prob(current, mu)
    total = float(np.sum(logp))
    return total if np.isfinite(total) else float("-inf")


def negative_log_likelihood(
    params: DiodeParams,
    voltage: np.ndarray,
    current: np.ndarray,
    noise_model: NoiseLikelihood,
    *,
    kind: str = "light",
) -> float:
    """``-log_likelihood`` -- the scalar objective a minimiser drives down."""
    return -log_likelihood(params, voltage, current, noise_model, kind=kind)


def gaussian_negloglike(residual: np.ndarray, sigma: float) -> float:
    """Gaussian NLL written directly in the sum-of-squares that proves the point.

    ``NLL = N/2 * log(2*pi*sigma**2) + (1 / (2*sigma**2)) * sum(residual**2)``.

    The only parameter-dependent term is ``sum(residual**2)`` scaled by the
    positive constant ``1/(2*sigma**2)``.  Its minimiser is therefore the
    least-squares minimiser, for any fixed ``sigma`` -- the identity Chapter 4
    is built around.  Provided as an explicit, testable statement of that fact.
    """
    _validate_positive(sigma, "sigma")
    residual = np.asarray(residual, dtype=float)
    n = residual.size
    sse = float(np.dot(residual, residual))
    return 0.5 * n * np.log(2.0 * np.pi * sigma * sigma) + sse / (2.0 * sigma * sigma)


# ---------------------------------------------------------------------------
# Maximum-likelihood fit (reuses fitting.py's parameterisation machinery)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLEResult:
    """Outcome of a maximum-likelihood diode fit.

    Attributes:
        params: fitted ``DiodeParams`` (fixed entries unchanged).
        free_names: names of the fitted parameters, in ``PARAM_NAMES`` order.
        negative_log_likelihood: value of the objective at the optimum.
        log_likelihood: ``-negative_log_likelihood`` for convenience.
        success: optimiser convergence flag.
        message: optimiser status message.
        noise_model: the noise model the fit assumed.
        n_points: number of data points.
    """

    params: DiodeParams
    free_names: tuple[str, ...]
    negative_log_likelihood: float
    log_likelihood: float
    success: bool
    message: str
    noise_model: NoiseLikelihood
    n_points: int


def mle_fit(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    noise_model: NoiseLikelihood,
    *,
    kind: str = "light",
    max_iter: int = 500,
) -> MLEResult:
    """Fit the free parameters in ``specs`` by maximising the likelihood.

    The parameterisation (log10 for ``j_0``/``r_sh``, bounds, fixed parameters)
    is exactly ``fitting.py``'s, via :func:`~src.models.fitting.pack` /
    :func:`~src.models.fitting.unpack`, so an MLE fit and a least-squares fit
    are comparable coordinate-for-coordinate.  With ``AbsoluteGaussian`` the two
    return the same parameters to optimiser tolerance -- the chapter's headline
    equivalence.

    Never raises on optimiser failure: a failed fit is reported through
    ``success=False`` and the optimiser message.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    theta0, lower, upper, free_names = pack(specs)

    if theta0.size == 0:
        params = unpack(np.empty(0), specs, temp_k)
        nll = negative_log_likelihood(params, voltage, current, noise_model, kind=kind)
        return MLEResult(
            params=params,
            free_names=free_names,
            negative_log_likelihood=nll,
            log_likelihood=-nll,
            success=True,
            message="No free parameters; evaluated fixed model.",
            noise_model=noise_model,
            n_points=int(voltage.size),
        )

    def objective(theta: np.ndarray) -> float:
        params = unpack(theta, specs, temp_k)
        nll = negative_log_likelihood(params, voltage, current, noise_model, kind=kind)
        # A finite guard keeps a gradient-based minimiser from stalling on -inf
        # at an overflowed boundary; the interior of the feasible box is smooth.
        return nll if np.isfinite(nll) else 1e12

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": max_iter},
    )

    params = unpack(result.x, specs, temp_k)
    nll = negative_log_likelihood(params, voltage, current, noise_model, kind=kind)
    return MLEResult(
        params=params,
        free_names=free_names,
        negative_log_likelihood=float(nll),
        log_likelihood=float(-nll),
        success=bool(result.success),
        message=str(result.message),
        noise_model=noise_model,
        n_points=int(voltage.size),
    )


# ---------------------------------------------------------------------------
# Estimating the noise from repeat sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseEstimate:
    """Noise estimated from replicate measurements at shared voltages.

    Attributes:
        sigma_absolute: pooled additive standard deviation, A/cm^2 -- the
            ``AbsoluteGaussian`` scale supported by the data.
        sigma_relative: pooled relative standard deviation (dimensionless) --
            the ``RelativeGaussian`` scale, from within-group SD over |mean|.
        degrees_of_freedom: total replicate degrees of freedom used, sum over
            groups of ``(replicates - 1)``.
        n_groups: number of voltage groups with at least two replicates.
        group_voltages: the representative voltage of each used group.
        group_sigma: per-group additive standard deviation, A/cm^2.
    """

    sigma_absolute: float
    sigma_relative: float
    degrees_of_freedom: int
    n_groups: int
    group_voltages: np.ndarray
    group_sigma: np.ndarray


def estimate_noise_from_repeats(
    voltage: np.ndarray,
    current: np.ndarray,
    *,
    tolerance: float = 0.0,
    min_replicates: int = 2,
) -> NoiseEstimate:
    """Estimate the measurement noise from replicate points, assuming nothing.

    Groups points whose voltages coincide (to within ``tolerance``), computes an
    unbiased sample variance within each group, and pools them by replicate
    degrees of freedom:

        ``sigma_absolute**2 = sum_g (n_g - 1) s_g**2 / sum_g (n_g - 1)``.

    Pooling assumes the additive noise variance is common across groups; the
    per-group values are returned so that assumption can be checked (a variance
    that climbs with the reading is the signature of relative noise, and is why
    ``sigma_relative`` is reported too).  This is the honest alternative to
    *assuming* a scale before fitting.

    Args:
        voltage: measured voltages (V); repeats appear as (near-)equal entries.
        current: measured current density (A/cm^2), aligned with ``voltage``.
        tolerance: absolute voltage window within which points are one group.
            ``0.0`` groups exactly-equal voltages (the synthetic-repeat case).
        min_replicates: minimum group size to contribute (>= 2).

    Returns:
        A :class:`NoiseEstimate`.

    Raises:
        ValueError: for mismatched shapes, ``min_replicates < 2``, or when no
            group reaches ``min_replicates`` replicates.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1:
        raise ValueError("voltage and current must be 1-D arrays of equal length")
    if min_replicates < 2:
        raise ValueError("min_replicates must be at least 2")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    order = np.argsort(voltage, kind="mergesort")
    v_sorted = voltage[order]
    i_sorted = current[order]

    # Greedy left-to-right grouping: a new group opens whenever a voltage falls
    # more than ``tolerance`` beyond the current group's opening voltage.
    group_v: list[float] = []
    group_s: list[float] = []
    weighted_var_sum = 0.0
    weighted_relvar_sum = 0.0
    dof_total = 0

    start = 0
    n = v_sorted.size
    while start < n:
        end = start + 1
        while end < n and (v_sorted[end] - v_sorted[start]) <= tolerance:
            end += 1
        block = i_sorted[start:end]
        if block.size >= min_replicates:
            var = float(np.var(block, ddof=1))
            dof = block.size - 1
            group_v.append(float(np.mean(v_sorted[start:end])))
            group_s.append(np.sqrt(var))
            weighted_var_sum += dof * var
            dof_total += dof
            mean_mag = max(abs(float(np.mean(block))), CURRENT_FLOOR)
            weighted_relvar_sum += dof * var / (mean_mag * mean_mag)
        start = end

    if dof_total == 0:
        raise ValueError(
            f"No voltage group reached {min_replicates} replicates; "
            "supply repeated measurements or widen the tolerance."
        )

    sigma_absolute = float(np.sqrt(weighted_var_sum / dof_total))
    sigma_relative = float(np.sqrt(weighted_relvar_sum / dof_total))
    return NoiseEstimate(
        sigma_absolute=sigma_absolute,
        sigma_relative=sigma_relative,
        degrees_of_freedom=int(dof_total),
        n_groups=len(group_v),
        group_voltages=np.asarray(group_v, dtype=float),
        group_sigma=np.asarray(group_s, dtype=float),
    )
