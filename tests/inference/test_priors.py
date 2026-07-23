"""Tests for priors as declared physics with provenance 

"""

from __future__ import annotations

import numpy as np
import pytest

from src.fitting.profile import ProfileResult
from src.fitting.noise import AbsoluteGaussian
from src.inference import priors as P
from src.inference.priors import PhysicalBound, Provenance
from src.models.fitting import DEFAULT_BOUNDS, default_specs, pack

FLOOR_SI = 6.25e-17
N_DRAWS = 4000


def _draw(bound: PhysicalBound, n: int = N_DRAWS, seed: int = 0) -> np.ndarray:
    """Draw ``n`` samples of one parameter from its prior via the NumPyro path.

    Fills the other four slots with the example bounds so the joint prior model
    runs, then returns the column of interest.
    """
    filler = P.example_physical_bounds()
    filler[bound.name] = bound
    return P.sample_prior_parameters(filler, n, seed)[bound.name]


# ---------------------------------------------------------------------------
# 1. Provenance vocabulary and the two-consumer contract
# ---------------------------------------------------------------------------


def test_provenance_is_a_fixed_six_class_vocabulary():
    assert {p.value for p in Provenance} == {
        "measured", "reciprocity", "geometry", "mechanism", "literature", "weak"
    }


def test_to_paramspec_bounds_fills_open_sides_from_default_and_feeds_the_fitter():
    # A one-sided reciprocity floor: lower set, upper open -> falls back to DEFAULT.
    b = PhysicalBound("j_0", Provenance.RECIPROCITY, lower=FLOOR_SI, center=FLOOR_SI * 100,
                      scale=1.5, log=True)
    lo, hi = b.to_paramspec_bounds()
    assert lo == FLOOR_SI
    assert hi == DEFAULT_BOUNDS["j_0"][1]  # open upper side fell back

    # The resolved box is exactly what the existing deterministic fitter accepts.
    bounds = {name: pb.to_paramspec_bounds() for name, pb in P.example_physical_bounds().items()}
    specs = default_specs("light", free=set(P.example_physical_bounds()), bounds=bounds)
    assert specs["j_0"].lower == FLOOR_SI
    assert specs["n"].lower == pytest.approx(2.0 / 3.0)
    assert specs["n"].upper == 2.0
    assert specs["r_s"].lower == 0.10
    # pack() transforms the j_0 lower bound into log space without error.
    _, lower, _, free = pack(specs)
    assert np.isfinite(lower).all()


# ---------------------------------------------------------------------------
# 2. Hard inequalities are structural (imposed by construction)
# ---------------------------------------------------------------------------


def test_reciprocity_draws_never_fall_below_the_radiative_floor():
    b = P.example_physical_bounds()["j_0"]
    s = _draw(b)
    assert s.min() >= FLOOR_SI          # ERE in (0, 1] enforced by construction
    assert np.median(s) > FLOOR_SI      # centred above the floor, not on it


def test_geometry_draws_never_fall_below_the_geometric_floor():
    b = P.example_physical_bounds()["r_s"]
    s = _draw(b)
    assert s.min() >= 0.10


def test_mechanism_draws_stay_inside_the_ideality_envelope():
    b = P.example_physical_bounds()["n"]
    s = _draw(b)
    assert s.min() >= 2.0 / 3.0 - 1e-9
    assert s.max() <= 2.0 + 1e-9


def test_measured_prior_is_a_tight_two_sided_band_on_the_photocurrent():
    b = P.example_physical_bounds()["j_ph"]
    s = _draw(b)
    assert np.mean(s) == pytest.approx(0.036064, rel=0.02)
    assert s.std() < 0.002          # tight: a measurement, not a search range
    lo, hi = b.to_paramspec_bounds()
    assert s.min() >= lo and s.max() <= hi


def test_ere_and_delta_v_nr_are_self_consistent():
    j0r = FLOOR_SI
    for j0 in (j0r, 1e-14, 4.58e-11):
        ere = P.ere_from_j0(j0, j0r)
        assert 0.0 < ere <= 1.0
        assert P.j0_from_ere(ere, j0r) == pytest.approx(j0, rel=1e-12)
        # dV_nr = Vt ln(j0/j0_rad) >= 0, and exactly 0 at the floor.
        assert P.delta_v_nr(j0, j0r) >= -1e-12
    assert P.delta_v_nr(j0r, j0r) == pytest.approx(0.0, abs=1e-12)
    # A decade of j_0 is one ln10*Vt of deficit (~59 mV at 298 K).
    from src.models.single_diode import thermal_voltage
    one_decade = P.delta_v_nr(10 * j0r, j0r)
    assert one_decade == pytest.approx(np.log(10) * thermal_voltage(298.15), rel=1e-9)


# ---------------------------------------------------------------------------
# 3. The prior predictive check
# ---------------------------------------------------------------------------


def test_physics_prior_predictive_beats_the_flat_search_range():
    v = np.linspace(0.0, 1.5, 120)
    phys = P.prior_predictive_jv(P.example_physical_bounds(), v, n_draws=600, seed=5)
    flat = P.prior_predictive_jv(P.uniform_reference_bounds(), v, n_draws=600, seed=5)

    # The physics prior knows the photocurrent; the flat one does not.
    phys_jsc = phys.j_sc[phys.physical]
    flat_jsc = flat.j_sc[flat.physical]
    assert phys_jsc.std() < 0.1 * flat_jsc.std()
    assert np.mean(np.abs(phys_jsc - 0.036) < 0.0036) > 0.9   # >90% within 10%
    assert np.mean(np.abs(flat_jsc - 0.036) < 0.0036) < 0.3   # flat: scattered

    # And it generates first-quadrant cells far more often.
    assert phys.physical.mean() > flat.physical.mean() + 0.2


def test_uniform_reference_bounds_are_log_uniform_for_log_parameters():
    b = P.uniform_reference_bounds()["j_0"]
    s = _draw(b)  # note: example filler replaces j_0 with the flat bound
    # log-uniform over 17 decades -> draws span many decades and stay in the box.
    lo, hi = DEFAULT_BOUNDS["j_0"]
    assert s.min() >= lo and s.max() <= hi
    assert np.log10(s.max()) - np.log10(s.min()) > 6.0


# ---------------------------------------------------------------------------
# 4. The prior bounds the ridge the data cannot
# ---------------------------------------------------------------------------


def _flat_open_below_profile() -> ProfileResult:
    """A synthetic j_0 profile that is flat (open) below its MLE and rises above.

    Mirrors poor-noise result without paying for the nuisance re-fit:
    delta_2nll = 0 for j_0 <= 1e-11, then climbs steeply, so the data set an
    upper bound but no lower one.
    """
    grid = np.logspace(-18, -8, 81)
    delta = np.where(grid <= 1e-11, 0.0, (np.log10(grid) - np.log10(1e-11)) * 20.0)
    return ProfileResult(
        parameter="j_0", grid=grid, profile_nll=-0.5 * (-delta),
        delta_2nll=delta, mle_value=1e-11, mle_nll=0.0,
        nuisance_names=("j_ph", "n", "r_s", "r_sh"),
        nuisance_values={k: np.zeros_like(grid) for k in ("j_ph", "n", "r_s", "r_sh")},
        success=np.ones_like(grid, dtype=bool), noise_model=AbsoluteGaussian(1e-4),
        reoptimised=True,
    )


def test_radiative_floor_closes_the_open_j0_interval():
    res = _flat_open_below_profile()
    bound = P.example_physical_bounds()["j_0"]
    pw = P.prior_weighted_profile(res, bound, level=0.95)

    assert pw.likelihood_open_below is True             # data alone: no lower bound
    lo, hi = pw.interval
    assert np.isfinite(lo) and np.isfinite(hi)          # posterior: closed
    assert lo >= FLOOR_SI * (1 - 1e-9)                  # bounded at/above the floor
    assert hi < res.grid.max()                          # upper bound from the data
    # The prior kills all posterior mass below the floor.
    assert pw.posterior[res.grid < FLOOR_SI].max() == 0.0
    # Normalised densities (coord-space, unit area over the grid).
    coord = np.log10(res.grid)
    assert np.trapezoid(pw.posterior, coord) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------


def test_invalid_constructions_and_arguments_raise():
    with pytest.raises(ValueError):
        PhysicalBound("not_a_param", Provenance.WEAK)
    with pytest.raises(ValueError):
        PhysicalBound("j_0", "measured")  # provenance must be the enum
    with pytest.raises(ValueError):
        PhysicalBound("r_s", Provenance.GEOMETRY, lower=1.0, upper=0.5)  # lower > upper
    with pytest.raises(ValueError):
        P.j0_from_ere(1.5, FLOOR_SI)      # ERE must be in (0, 1]
    with pytest.raises(ValueError):
        # a reciprocity prior needs a floor
        PhysicalBound("j_0", Provenance.RECIPROCITY, log=True).to_prior()
    res = _flat_open_below_profile()
    with pytest.raises(ValueError):
        P.prior_weighted_profile(res, P.example_physical_bounds()["j_0"], level=1.5)
