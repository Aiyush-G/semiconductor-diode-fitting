"""Tests for measurement-noise models, likelihoods and MLE fitting.



1. the log densities are the standard ones (checked against SciPy);
2. a constant-scale Gaussian negative log-likelihood is an affine, increasing
   function of the sum of squared residuals, so its minimiser is least squares;
3. the *choice of noise model* reproduces the *choice of residual space* -- an
   absolute-Gaussian MLE matches a linear-residual fit and a log-normal MLE
   matches a log-residual fit, moving the fitted dark J_0 by an order of
   magnitude.
4. heavy-tailed likelihoods resist outliers, and noise can be measured from
   repeats instead of assumed.
"""

import numpy as np
import pytest
from scipy import stats

from src.fitting import noise as N
from src.models.examples import EXAMPLE_DATASETS
from src.models.fitting import default_specs, fit_diode
from src.models.single_diode import DiodeParams, solve_current
from src.models.synthetic import GaussianNoise, generate_synthetic


# ---------------------------------------------------------------------------
# Elementwise densities
# ---------------------------------------------------------------------------


def test_normal_logpdf_matches_scipy():
    x = np.array([-1.3, -0.2, 0.0, 0.7, 4.1])
    loc = np.array([0.1, 0.0, -0.5, 0.7, 2.0])
    scale = np.array([0.5, 0.2, 1.0, 0.3, 2.5])
    assert np.allclose(N.normal_logpdf(x, loc, scale), stats.norm.logpdf(x, loc, scale))


@pytest.mark.parametrize("df", [1.0, 2.5, 4.0, 30.0])
def test_student_t_logpdf_matches_scipy(df):
    x = np.array([-2.0, -0.4, 0.0, 0.9, 3.3])
    loc = np.array([0.0, 0.1, -0.2, 1.0, 2.0])
    scale = np.array([1.0, 0.4, 0.7, 0.3, 1.5])
    expected = stats.t.logpdf(x, df=df, loc=loc, scale=scale)
    assert np.allclose(N.student_t_logpdf(x, loc, scale, df), expected)


def test_student_t_tends_to_gaussian_for_large_df():
    x = np.linspace(-3, 3, 25)
    loc = np.zeros_like(x)
    scale = np.ones_like(x)
    gaussian = N.normal_logpdf(x, loc, scale)
    heavy = N.student_t_logpdf(x, loc, scale, df=5.0)
    nearly = N.student_t_logpdf(x, loc, scale, df=5000.0)
    # A large df is close to Gaussian; a small df is not.
    assert np.max(np.abs(nearly - gaussian)) < 1e-2
    assert np.max(np.abs(heavy - gaussian)) > 1e-2


def test_densities_reject_nonpositive_scale():
    with pytest.raises(ValueError):
        N.normal_logpdf(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        N.student_t_logpdf(np.array([0.0]), np.array([0.0]), np.array([1.0]), df=0.0)


# ---------------------------------------------------------------------------
# Noise-model scale rules
# ---------------------------------------------------------------------------


def test_relative_scale_tracks_the_reading():
    mu = np.array([1e-4, 1e-3, 1e-2])
    model = N.RelativeGaussian(0.05)
    assert np.allclose(model.scale(mu), 0.05 * np.abs(mu))


def test_floor_relative_reduces_to_limits():
    mu = np.array([1e-5, 1e-3, 1e-1])
    floor_only = N.FloorRelativeGaussian(2e-5, 1e-9)
    rel_only = N.FloorRelativeGaussian(1e-12, 0.05)
    assert np.allclose(floor_only.scale(mu), 2e-5, rtol=1e-3)
    assert np.allclose(rel_only.scale(mu), 0.05 * np.abs(mu), rtol=1e-3)


def test_relative_scale_never_hits_zero_at_voc():
    # At an exact current zero the relative scale must stay positive (floored).
    model = N.RelativeGaussian(0.05)
    assert np.all(model.scale(np.array([0.0])) > 0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: N.AbsoluteGaussian(0.0),
        lambda: N.RelativeGaussian(-1.0),
        lambda: N.FloorRelativeGaussian(0.0, 0.1),
        lambda: N.StudentTLikelihood(1e-5, degrees_of_freedom=0.0),
        lambda: N.LogNormalLikelihood(-0.1),
    ],
)
def test_noise_models_validate_parameters(factory):
    with pytest.raises(ValueError):
        factory()


def test_noise_models_are_immutable():
    model = N.AbsoluteGaussian(2e-5)
    with pytest.raises(Exception):
        model.sigma = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The least-squares == Gaussian-MLE identity
# ---------------------------------------------------------------------------


def test_gaussian_negloglike_matches_full_likelihood():
    truth = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.5, r_sh=800.0)
    v = np.linspace(0, 0.65, 40)
    ds = generate_synthetic(truth, v, kind="light", noise_model=GaussianNoise(2e-5), seed=1)
    mu = solve_current(ds.voltage, ds.params)
    residual = np.asarray(ds.current) - mu
    direct = N.gaussian_negloglike(residual, 2e-5)
    full = N.negative_log_likelihood(
        ds.params, ds.voltage, ds.current, N.AbsoluteGaussian(2e-5), kind="light"
    )
    assert direct == pytest.approx(full, rel=1e-12)


def test_gaussian_negloglike_is_affine_increasing_in_sse():
    rng = np.random.default_rng(0)
    r1 = rng.normal(size=50) * 1e-4
    r2 = r1 * 1.7  # strictly larger sum of squares
    sigma = 3e-4
    nll1 = N.gaussian_negloglike(r1, sigma)
    nll2 = N.gaussian_negloglike(r2, sigma)
    sse1 = float(np.dot(r1, r1))
    sse2 = float(np.dot(r2, r2))
    assert nll2 > nll1  # more residual, worse likelihood
    # Affine: (NLL2 - NLL1) == (SSE2 - SSE1) / (2 sigma^2), exactly.
    assert (nll2 - nll1) == pytest.approx((sse2 - sse1) / (2 * sigma**2), rel=1e-12)


def test_least_squares_equals_gaussian_mle_when_well_conditioned():
    # Free only (j_ph, n): a well-identified pair, so the minimum is unique and
    # both optimisers must land on it. (Freeing the sloppy j_0-n ridge instead
    # would let them stop at different points of a flat valley )
    truth = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.5, r_sh=800.0)
    v = np.linspace(0, 0.66, 80)
    ds = generate_synthetic(truth, v, kind="light", noise_model=GaussianNoise(2e-5), seed=7)
    fixed = {"j_0": (1e-12, 1e-12), "r_s": (0.5, 0.5), "r_sh": (800.0, 800.0)}
    initial = {"j_ph": 0.036, "j_0": 1e-12, "n": 1.2, "r_s": 0.5, "r_sh": 800.0}
    specs = default_specs("light", {"j_ph", "n"}, initial=initial, bounds=fixed)

    ls = fit_diode(ds.voltage, ds.current, truth.temp_k, specs, kind="light")
    ml = N.mle_fit(
        ds.voltage, ds.current, truth.temp_k, specs, N.AbsoluteGaussian(2e-5), kind="light"
    )
    assert ls.success and ml.success
    assert ml.params.j_ph == pytest.approx(ls.params.j_ph, rel=1e-4)
    assert ml.params.n == pytest.approx(ls.params.n, rel=1e-4)


# ---------------------------------------------------------------------------
# The noise model IS the residual space 
# ---------------------------------------------------------------------------


def test_absolute_gaussian_mle_matches_linear_residual_fit():
    dark = EXAMPLE_DATASETS["Example: Dark JV"]
    v, j = np.asarray(dark.voltage), np.asarray(dark.current)
    specs = default_specs("dark", {"j_0", "n", "r_s", "r_sh"})
    ls = fit_diode(v, j, dark.temp_k, specs, kind="dark", residual_space="linear")
    ml = N.mle_fit(v, j, dark.temp_k, specs, N.AbsoluteGaussian(1e-3), kind="dark")
    assert ls.params.j_0 == pytest.approx(ml.params.j_0, rel=5e-3)
    assert ls.params.n == pytest.approx(ml.params.n, rel=5e-3)
    # measured linear-residual value.
    assert ml.params.j_0 == pytest.approx(2.68e-10, rel=5e-2)


def test_lognormal_mle_matches_log_residual_fit():
    dark = EXAMPLE_DATASETS["Example: Dark JV"]
    v, j = np.asarray(dark.voltage), np.asarray(dark.current)
    specs = default_specs("dark", {"j_0", "n", "r_s", "r_sh"})
    ls = fit_diode(v, j, dark.temp_k, specs, kind="dark", residual_space="log")
    ml = N.mle_fit(v, j, dark.temp_k, specs, N.LogNormalLikelihood(0.1), kind="dark")
    assert ls.params.j_0 == pytest.approx(ml.params.j_0, rel=5e-3)
    assert ls.params.n == pytest.approx(ml.params.n, rel=5e-3)
    #  measured log-residual value -- an order of magnitude below.
    assert ml.params.j_0 == pytest.approx(3.72e-11, rel=5e-2)
    assert ml.params.j_0 < 0.2 * 2.68e-10  # decisively different from the linear fit


# ---------------------------------------------------------------------------
# Robustness and noise estimation
# ---------------------------------------------------------------------------


def test_student_t_resists_outliers_better_than_gaussian():
    truth = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.6, r_sh=800.0)
    v = np.linspace(0, 0.66, 80)
    ds = generate_synthetic(truth, v, kind="light", noise_model=GaussianNoise(2e-5), seed=3)
    contaminated = np.array(ds.current, copy=True)
    # Four one-sided gross outliers (a stuck-high meter): they do not cancel, so
    # a Gaussian fit is dragged upward in both J_L and n.
    contaminated[[16, 32, 48, 64]] += np.array([4e-3, 5e-3, 4.5e-3, 6e-3])

    fixed = {"r_s": (0.6, 0.6), "r_sh": (800.0, 800.0)}
    initial = {"j_ph": 0.036, "j_0": 1e-12, "n": 1.2, "r_s": 0.6, "r_sh": 800.0}
    specs = default_specs("light", {"j_ph", "j_0", "n"}, initial=initial, bounds=fixed)

    gaussian = N.mle_fit(
        v, contaminated, truth.temp_k, specs, N.AbsoluteGaussian(2e-5), kind="light"
    )
    robust = N.mle_fit(
        v, contaminated, truth.temp_k, specs,
        N.StudentTLikelihood(2e-5, degrees_of_freedom=3.0), kind="light",
    )
    err_gaussian = abs(gaussian.params.j_ph - 0.036)
    err_robust = abs(robust.params.j_ph - 0.036)
    # The robust fit is dramatically closer to the truth on both parameters.
    assert err_robust < 0.05 * err_gaussian
    assert robust.params.j_ph == pytest.approx(0.036, abs=5e-5)
    assert robust.params.n == pytest.approx(1.2, abs=5e-3)
    assert gaussian.params.n > 1.24  # the Gaussian fit is visibly biased


def test_estimate_noise_from_repeats_recovers_known_sigma():
    truth = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.6, r_sh=800.0)
    voltages = np.repeat(np.linspace(0.0, 0.66, 40), 60)
    ds = generate_synthetic(
        truth, voltages, kind="light", noise_model=GaussianNoise(2.5e-5), seed=11
    )
    est = N.estimate_noise_from_repeats(ds.voltage, ds.current)
    assert est.n_groups == 40
    assert est.degrees_of_freedom == 40 * 59
    assert est.sigma_absolute == pytest.approx(2.5e-5, rel=0.05)


def test_estimate_noise_relative_recovers_lognormal_scale():
    from src.models.synthetic import LogNormalNoise

    truth = DiodeParams(j_ph=0.0, j_0=1e-11, n=1.3, r_s=0.5, r_sh=1e4)
    voltages = np.repeat(np.linspace(0.3, 0.7, 30), 80)
    ds = generate_synthetic(
        truth, voltages, kind="dark", noise_model=LogNormalNoise(0.04), seed=5
    )
    est = N.estimate_noise_from_repeats(ds.voltage, ds.current)
    # For small sigma_ln, the relative SD approaches sigma_ln.
    assert est.sigma_relative == pytest.approx(0.04, rel=0.1)


def test_estimate_noise_requires_replicates():
    v = np.linspace(0, 0.6, 10)  # all voltages distinct
    j = np.linspace(0.03, -0.01, 10)
    with pytest.raises(ValueError):
        N.estimate_noise_from_repeats(v, j)
    with pytest.raises(ValueError):
        N.estimate_noise_from_repeats(v, j, min_replicates=1)


def test_estimate_noise_groups_within_tolerance():
    # Two clusters of three near-equal voltages; exact grouping would see six
    # singletons, so a tolerance is required to recover the two groups.
    v = np.array([0.10, 0.1001, 0.0999, 0.50, 0.5002, 0.4998])
    j = np.array([1.0, 1.2, 0.8, 2.0, 2.1, 1.9]) * 1e-3
    est = N.estimate_noise_from_repeats(v, j, tolerance=1e-3)
    assert est.n_groups == 2
    assert est.degrees_of_freedom == 4


# ---------------------------------------------------------------------------
# Sweep likelihood plumbing
# ---------------------------------------------------------------------------


def test_log_likelihood_forces_dark_photocurrent_to_zero():
    params = DiodeParams(j_ph=0.036, j_0=1e-11, n=1.3, r_s=0.5, r_sh=1e4)
    v = np.linspace(0.1, 0.7, 30)
    dark = generate_synthetic(params, v, kind="dark", noise_model=GaussianNoise(1e-6), seed=2)
    # Passing the light truth with kind="dark" must give the same likelihood as
    # passing an explicitly zeroed photocurrent: the dark branch ignores j_ph.
    from dataclasses import replace

    zeroed = replace(params, j_ph=0.0)
    model = N.AbsoluteGaussian(1e-6)
    with_light_jph = N.log_likelihood(params, dark.voltage, dark.current, model, kind="dark")
    with_zero_jph = N.log_likelihood(zeroed, dark.voltage, dark.current, model, kind="dark")
    assert with_light_jph == pytest.approx(with_zero_jph, rel=1e-12)


def test_log_likelihood_validates_shapes_and_kind():
    params = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.5, r_sh=800.0)
    model = N.AbsoluteGaussian(2e-5)
    with pytest.raises(ValueError):
        N.log_likelihood(params, np.zeros(5), np.zeros(4), model)
    with pytest.raises(ValueError):
        N.log_likelihood(params, np.zeros(5), np.zeros(5), model, kind="bogus")


def test_mle_fit_all_fixed_evaluates_without_optimising():
    dark = EXAMPLE_DATASETS["Example: Dark JV"]
    v, j = np.asarray(dark.voltage), np.asarray(dark.current)
    specs = default_specs("dark", set())  # nothing free
    result = N.mle_fit(v, j, dark.temp_k, specs, N.AbsoluteGaussian(1e-3), kind="dark")
    assert result.success
    assert result.free_names == ()
    assert np.isfinite(result.log_likelihood)
    assert isinstance(result.noise_model, N.AbsoluteGaussian)
