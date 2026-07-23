"""Tests for the profile likelihood and sloppiness tools (Chapter 5).

Five claims the chapter rests on:

1. the profile is a genuine minimisation -- its floor equals the joint MLE and it
   never dips below it, and re-optimising the nuisances is strictly gentler than
   freezing them (the "classic mistake");
2. a profile confidence interval separates a *well-identified* parameter (tight,
   truth-bracketing) from a *practically non-identifiable* one (flat over orders
   of magnitude), using the Wilks chi-squared threshold;
3. the single-diode ``j_0`` degeneracy is *practical*, not structural: its
   profile tightens as the noise falls, and the escape route runs through an
   ideality factor ``n`` that reaches unphysical values;
4. the sloppy eigendirection of ``J^T J`` reproduces Chapter 2's conditioning
   (kappa = 12381 on the example light curve) and is the ``j_0``--``n`` ridge;
5. the tools validate their inputs.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from src.fitting import profile as P
from src.fitting.noise import AbsoluteGaussian
from src.models.examples import EXAMPLE_DATASETS
from src.models.fitting import default_specs, fit_diode
from src.models.single_diode import DiodeParams
from src.models.synthetic import GaussianNoise, generate_synthetic


# A shared, well-behaved synthetic light curve (truth known).
TRUTH = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.2, r_s=0.5, r_sh=800.0)
V = np.linspace(0.0, 0.66, 80)


def _light(sigma, seed=7):
    ds = generate_synthetic(TRUTH, V, kind="light", noise_model=GaussianNoise(sigma), seed=seed)
    return np.asarray(ds.voltage), np.asarray(ds.current)


def _specs():
    # Widen n so the ridge can run into unphysical territory rather than clip.
    return default_specs("light", {"j_ph", "j_0", "n", "r_s", "r_sh"}, bounds={"n": (0.5, 4.0)})


# ---------------------------------------------------------------------------
# 1. The profile is a real minimisation
# ---------------------------------------------------------------------------


def test_profile_floor_equals_joint_mle_and_never_dips_below():
    v, j = _light(2e-5)
    grid = np.logspace(-14, -9, 61)
    res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(2e-5), "j_0", grid)
    # Profiling off the optimum can only raise the NLL: delta >= 0 (tiny slack).
    assert np.nanmin(res.delta_2nll) >= -1e-6
    # The floor of the profile is the joint MLE, to a fraction of a chi-square unit.
    assert res.profile_nll.min() == pytest.approx(res.mle_nll, abs=0.2)
    assert res.reoptimised is True


def test_reoptimising_is_strictly_gentler_than_freezing_the_nuisances():
    v, j = _light(2e-5)
    grid = np.logspace(-14, -9, 121)
    kw = dict(temp_k=TRUTH.temp_k, specs=_specs(), noise_model=AbsoluteGaussian(2e-5),
              parameter="j_0", grid=grid)
    prof = P.profile_parameter(v, j, reoptimise=True, **kw)
    slice_ = P.profile_parameter(v, j, reoptimise=False, **kw)
    # A re-optimised profile is <= a frozen slice at every value (same baseline).
    good = np.isfinite(prof.delta_2nll) & np.isfinite(slice_.delta_2nll)
    assert np.all(prof.delta_2nll[good] <= slice_.delta_2nll[good] + 1e-6)
    # And dramatically so half a decade out: freezing n overstates the cost >100x.
    k = int(np.argmin(np.abs(grid - 3.0 * prof.mle_value)))
    assert slice_.delta_2nll[k] > 100.0 * prof.delta_2nll[k]


# ---------------------------------------------------------------------------
# 2. Confidence intervals: identified vs not
# ---------------------------------------------------------------------------


def test_well_identified_parameter_has_a_tight_interval_bracketing_truth():
    v, j = _light(2e-5)
    # j_ph is pinned by Jsc; its profile is a narrow parabola around the truth.
    grid = np.linspace(0.0350, 0.0370, 81)
    res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(2e-5), "j_ph", grid)
    ci = P.profile_interval(res, 0.95)
    assert not ci.lower_capped and not ci.upper_capped
    assert ci.lower < TRUTH.j_ph < ci.upper
    assert (ci.upper - ci.lower) < 5e-4  # sub-percent on a 36 mA/cm^2 photocurrent


def test_wilks_threshold_is_chi2_one_dof():
    v, j = _light(2e-5)
    grid = np.logspace(-14, -9, 61)
    res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(2e-5), "j_0", grid)
    ci = P.profile_interval(res, 0.95)
    assert ci.threshold == pytest.approx(chi2.ppf(0.95, 1))
    assert ci.threshold == pytest.approx(3.841458, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. The j_0 degeneracy is practical, and escapes through unphysical n
# ---------------------------------------------------------------------------


def test_j0_practical_non_identifiability_tightens_with_noise():
    grid = np.logspace(-15, -8, 71)
    factors = {}
    capped = {}
    for sigma in (2e-6, 1e-4):
        v, j = _light(sigma)
        res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(sigma), "j_0", grid)
        ci = P.profile_interval(res, 0.95)
        factors[sigma] = ci.factor
        capped[sigma] = ci.lower_capped or ci.upper_capped
    # Low noise: j_0 is pinned to within a small factor (structural would not be).
    assert factors[2e-6] < 3.0 and not capped[2e-6]
    # Realistic-to-poor noise: the profile is flat over orders of magnitude --
    # the data stop bounding j_0 on at least one side.
    assert capped[1e-4] or factors[1e-4] > 100.0
    assert factors[1e-4] > 30.0 * factors[2e-6]


def test_escape_route_runs_through_unphysical_ideality():
    v, j = _light(1e-4)
    grid = np.logspace(-15, -8, 71)
    res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(1e-4), "j_0", grid)
    n_traj = res.nuisance_values[:, res.nuisance_names.index("n")]
    log_j0 = np.log10(res.grid)
    # n rises monotonically with log10(j_0) along the valley floor: the ridge.
    within = res.delta_2nll < 10.0
    assert np.corrcoef(log_j0[within], n_traj[within])[0, 1] > 0.9
    # To hold the fit while j_0 sweeps, n must leave the physical single-mechanism
    # band [1, 2] -- reaching >= 2 is exactly why a mechanism ladder is needed.
    assert n_traj.max() >= 2.0
    assert n_traj.min() < 1.2


# ---------------------------------------------------------------------------
# 4. Sloppiness reproduces Chapter 2
# ---------------------------------------------------------------------------


def test_sloppy_spectrum_reproduces_chapter2_conditioning():
    light = EXAMPLE_DATASETS["Example: Light JV"]
    v, j = np.asarray(light.voltage), np.asarray(light.current)
    free = {"j_ph", "j_0", "n", "r_s", "r_sh"}
    ls = fit_diode(v, j, light.temp_k, default_specs("light", free), kind="light")
    at_mle = default_specs("light", free, initial={k: getattr(ls.params, k) for k in free})
    sp = P.sloppy_spectrum(v, j, light.temp_k, at_mle, kind="light")
    # Chapter 2 canon: kappa(J) = 12381.2 on the five-parameter light fit.
    assert sp.condition_number == pytest.approx(12381.2, rel=1e-3)
    # Eigenvalues are ascending (softest first) and strictly positive.
    assert np.all(np.diff(sp.eigenvalues) >= 0)
    assert sp.eigenvalues[0] > 0


def test_softest_direction_is_the_j0_n_ridge():
    light = EXAMPLE_DATASETS["Example: Light JV"]
    v, j = np.asarray(light.voltage), np.asarray(light.current)
    free = {"j_ph", "j_0", "n", "r_s", "r_sh"}
    ls = fit_diode(v, j, light.temp_k, default_specs("light", free), kind="light")
    at_mle = default_specs("light", free, initial={k: getattr(ls.params, k) for k in free})
    sp = P.sloppy_spectrum(v, j, light.temp_k, at_mle, kind="light")
    weights = dict(zip(sp.free_names, sp.softest))
    # The flat direction is dominated by j_0 with a same-sign n admixture, and is
    # essentially blind to the well-determined photocurrent.
    assert abs(weights["j_0"]) == max(abs(w) for w in sp.softest)
    assert np.sign(weights["j_0"]) == np.sign(weights["n"])
    assert abs(weights["j_ph"]) < 0.05


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------


def test_profile_rejects_a_fixed_parameter():
    v, j = _light(2e-5)
    specs = default_specs("light", {"j_ph", "n"})  # j_0 is fixed here
    with pytest.raises(ValueError):
        P.profile_parameter(v, j, TRUTH.temp_k, specs, AbsoluteGaussian(2e-5),
                            "j_0", np.logspace(-13, -11, 10))


def test_profile_rejects_a_short_grid():
    v, j = _light(2e-5)
    with pytest.raises(ValueError):
        P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(2e-5),
                            "j_0", np.array([1e-12]))


def test_profile_interval_rejects_bad_level():
    v, j = _light(2e-5)
    res = P.profile_parameter(v, j, TRUTH.temp_k, _specs(), AbsoluteGaussian(2e-5),
                             "j_0", np.logspace(-14, -10, 21))
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            P.profile_interval(res, bad)


def test_sloppy_spectrum_needs_two_free_parameters():
    v, j = _light(2e-5)
    specs = default_specs("light", {"j_0"})  # only one free direction
    with pytest.raises(ValueError):
        P.sloppy_spectrum(v, j, TRUTH.temp_k, specs, kind="light")
