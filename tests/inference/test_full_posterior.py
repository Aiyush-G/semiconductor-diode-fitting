"""Regression tests for the five-parameter light-plus-dark posterior."""

from __future__ import annotations

import numpy as np
import numpyro
import pytest

from src.fitting.noise import AbsoluteGaussian, LogNormalLikelihood
from src.fitting.profile import profile_interval
from src.inference.comparison import (
    ComparisonFlag,
    JointProfileResult,
    compare_profile_posterior,
    joint_mle,
    joint_negative_log_likelihood,
    joint_profile_parameter,
)
from src.inference.models_numpyro import full_joint_model, joint_log_likelihood_jax
from src.inference.priors import example_physical_bounds
from src.inference.run import posterior_summary, run_nuts
from src.models.fitting import PARAM_NAMES, default_specs
from src.models.single_diode import DiodeParams
from src.models.synthetic import (
    GaussianNoise,
    LogNormalNoise,
    generate_synthetic,
)


TRUTH = DiodeParams(
    j_ph=0.036,
    j_0=1e-12,
    n=1.2,
    r_s=0.5,
    r_sh=800.0,
    temp_k=298.15,
)
SIGMA_LIGHT = 2e-5
SIGMA_DARK_LN = 0.02
V_LIGHT = np.linspace(0.0, 0.66, 80)
V_DARK = np.linspace(0.001, 0.68, 100)


def _problem():
    light = generate_synthetic(
        TRUTH,
        V_LIGHT,
        kind="light",
        noise_model=GaussianNoise(SIGMA_LIGHT),
        seed=7,
    )
    dark = generate_synthetic(
        TRUTH,
        V_DARK,
        kind="dark",
        noise_model=LogNormalNoise(SIGMA_DARK_LN),
        seed=8,
    )
    bounds = example_physical_bounds(j_ph_center=TRUTH.j_ph, r_sh_center=TRUTH.r_sh)
    box = {name: bound.to_paramspec_bounds() for name, bound in bounds.items()}
    initial = {"j_ph": 0.035, "j_0": 2e-12, "n": 1.3, "r_s": 0.3, "r_sh": 500.0}
    specs = default_specs("light", set(PARAM_NAMES), initial=initial, bounds=box)
    return light, dark, bounds, specs


def _profile(
    parameter: str,
    grid: np.ndarray,
    delta_2nll: np.ndarray,
) -> JointProfileResult:
    """Small synthetic profile for classification tests."""
    delta = np.asarray(delta_2nll, dtype=float)
    return JointProfileResult(
        parameter=parameter,
        grid=np.asarray(grid, dtype=float),
        profile_nll=0.5 * delta,
        delta_2nll=delta,
        mle_value=float(grid[int(np.argmin(delta))]),
        mle_nll=0.0,
        nuisance_names=(),
        nuisance_values=np.empty((len(grid), 0)),
        success=np.ones(len(grid), dtype=bool),
        light_noise=AbsoluteGaussian(SIGMA_LIGHT),
        dark_noise=LogNormalLikelihood(SIGMA_DARK_LN),
        reoptimised=False,
    )


def test_joint_jax_likelihood_matches_numpy_term_for_term():
    numpyro.enable_x64()
    light, dark, _, _ = _problem()
    numpy_nll = joint_negative_log_likelihood(
        TRUTH,
        V_LIGHT,
        light.current,
        V_DARK,
        dark.current,
        AbsoluteGaussian(SIGMA_LIGHT),
        LogNormalLikelihood(SIGMA_DARK_LN),
    )
    jax_nll = -float(
        joint_log_likelihood_jax(
            V_LIGHT,
            light.current,
            V_DARK,
            dark.current,
            j_ph=TRUTH.j_ph,
            j_0=TRUTH.j_0,
            n=TRUTH.n,
            r_s=TRUTH.r_s,
            r_sh=TRUTH.r_sh,
            sigma_light=SIGMA_LIGHT,
            sigma_dark_ln=SIGMA_DARK_LN,
        )
    )
    assert jax_nll == pytest.approx(numpy_nll, abs=5e-10)


def test_joint_mle_recovers_all_five_parameters():
    light, dark, _, specs = _problem()
    fit = joint_mle(
        V_LIGHT,
        light.current,
        V_DARK,
        dark.current,
        TRUTH.temp_k,
        specs,
        AbsoluteGaussian(SIGMA_LIGHT),
        LogNormalLikelihood(SIGMA_DARK_LN),
        max_iter=2000,
    )
    assert fit.success
    assert fit.params.j_ph == pytest.approx(TRUTH.j_ph, rel=5e-4)
    assert fit.params.j_0 == pytest.approx(TRUTH.j_0, rel=0.2)
    assert fit.params.n == pytest.approx(TRUTH.n, rel=0.02)
    assert fit.params.r_s == pytest.approx(TRUTH.r_s, rel=0.04)
    assert fit.params.r_sh == pytest.approx(TRUTH.r_sh, rel=0.01)


def test_joint_profile_reoptimises_every_nuisance_and_contains_mle():
    light, dark, _, specs = _problem()
    result = joint_profile_parameter(
        V_LIGHT,
        light.current,
        V_DARK,
        dark.current,
        TRUTH.temp_k,
        specs,
        AbsoluteGaussian(SIGMA_LIGHT),
        LogNormalLikelihood(SIGMA_DARK_LN),
        "n",
        np.linspace(1.15, 1.26, 13),
        max_iter=1000,
    )
    interval = profile_interval(result)
    assert result.reoptimised
    assert set(result.nuisance_names) == {"j_ph", "j_0", "r_s", "r_sh"}
    assert result.nuisance_values.shape == (13, 4)
    assert np.all(np.isfinite(result.profile_nll))
    assert np.all(result.success)
    assert interval.lower < result.mle_value < interval.upper


def test_full_joint_nuts_recovers_truth_with_physical_draws():
    light, dark, bounds, _ = _problem()
    _, idata = run_nuts(
        full_joint_model,
        model_kwargs={
            "light_voltage": V_LIGHT,
            "light_current": light.current,
            "dark_voltage": V_DARK,
            "dark_current": dark.current,
            "bounds": bounds,
            "sigma_light": SIGMA_LIGHT,
            "sigma_dark_ln": SIGMA_DARK_LN,
            "temp_k": TRUTH.temp_k,
        },
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=9,
        target_accept=0.92,
    )
    summary = posterior_summary(idata, PARAM_NAMES)
    for name in PARAM_NAMES:
        assert summary[name]["lo95"] < getattr(TRUTH, name) < summary[name]["hi95"]
        assert np.all(np.asarray(idata.posterior[name]) > 0)
        assert summary[name]["r_hat"] < 1.02
    assert np.min(np.asarray(idata.posterior["j_0"])) >= bounds["j_0"].lower
    assert np.min(np.asarray(idata.posterior["r_s"])) >= bounds["r_s"].lower


def test_dual_report_recognises_data_driven_agreement():
    rng = np.random.default_rng(12)
    grid = np.linspace(0.7, 1.3, 301)
    sigma = 0.08
    profile = _profile("n", grid, ((grid - 1.0) / sigma) ** 2)
    report = compare_profile_posterior(
        profile, rng.normal(1.0, sigma, 6000), coordinate="linear"
    )
    assert report.flags == (ComparisonFlag.AGREEMENT,)
    assert report.interval_overlap > 0.9
    assert report.js_divergence < 0.02
    assert report.posterior_mass_in_grid > 0.99


def test_dual_report_flags_open_bound_and_multimodal_cases():
    rng = np.random.default_rng(13)
    log_grid = np.logspace(-16, -10, 301)
    open_profile = _profile("j_0", log_grid, np.zeros(log_grid.size))
    open_report = compare_profile_posterior(
        open_profile,
        10.0 ** rng.normal(-12.0, 0.3, 4000),
        physical_bounds=(1e-16, None),
    )
    assert ComparisonFlag.PROFILE_OPEN in open_report.flags
    assert ComparisonFlag.BOUND_CONTACT in open_report.flags
    assert open_report.coordinate == "log10"

    grid = np.linspace(-2.0, 2.0, 401)
    curved = _profile("n", grid, grid**2)
    draws = np.concatenate(
        [rng.normal(-0.85, 0.08, 2500), rng.normal(0.85, 0.08, 2500)]
    )
    multi_report = compare_profile_posterior(curved, draws, coordinate="linear")
    assert ComparisonFlag.MULTIMODAL in multi_report.flags
    assert multi_report.n_posterior_modes == 2


def test_dual_report_rejects_invalid_draws_and_log_coordinates():
    grid = np.linspace(0.5, 1.5, 101)
    profile = _profile("n", grid, (grid - 1.0) ** 2)
    with pytest.raises(ValueError, match="at least 20"):
        compare_profile_posterior(profile, np.ones(30))
    with pytest.raises(ValueError, match="strictly positive"):
        compare_profile_posterior(profile, np.linspace(-1.0, 1.0, 40), coordinate="log10")
