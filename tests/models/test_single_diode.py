"""Physics regression tests for the single-diode forward model.

Every test here checks a *limiting case or exact identity* of the single-diode
equation, so failures indicate physics errors rather than style regressions:

    J = J_ph - J_0 * (exp((V + J*Rs) / (n*Vt)) - 1) - (V + J*Rs) / Rsh

The battery covers: the Lambert-W closed form satisfying the implicit equation
to machine precision; continuity of the ``r_s == 0`` special-case branch;
reduction to the ideal Shockley diode as R_sh -> inf; the exact
series-resistance invariance of Voc; light/dark superposition (exact at
R_s = 0); the squaring of the curve as J_0 -> 0 with the n*Vt*ln(10) Voc
ladder; hand-computed ``key_metrics``; the m(V) plateau of
``local_ideality_factor``; the auto-extending sweep; and the documented
overflow boundary of the Lambert-W argument at extreme forward bias.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from src.models.single_diode import (
    DiodeParams,
    iv_curve,
    key_metrics,
    local_ideality_factor,
    solve_current,
    thermal_voltage,
)

# Reference devices used across the battery (area-normalised units: A/cm^2,
# Ohm.cm^2, matching the module convention).
SILICON = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.8, r_sh=2000.0)
PEROVSKITE = DiodeParams(j_ph=20e-3, j_0=1e-18, n=1.6, r_s=2.0, r_sh=5000.0)
LOSSY = DiodeParams(j_ph=30e-3, j_0=1e-9, n=2.0, r_s=5.0, r_sh=200.0)


def implicit_residual(voltage: np.ndarray, current: np.ndarray,
                      params: DiodeParams) -> np.ndarray:
    """Residual of the implicit single-diode equation; zero iff J solves it."""
    nvt = params.n * thermal_voltage(params.temp_k)
    v_junction = voltage + current * params.r_s
    return (params.j_ph
            - params.j_0 * (np.exp(v_junction / nvt) - 1.0)
            - v_junction / params.r_sh
            - current)


# ---------------------------------------------------------------------------
# The closed form is the solution of the implicit equation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [SILICON, PEROVSKITE, LOSSY],
                         ids=["silicon", "perovskite", "lossy"])
def test_lambert_w_satisfies_implicit_equation(params):
    """|residual| stays at the machine-precision floor across the full sweep."""
    voltage, current = iv_curve(params, n_points=400)
    residual = implicit_residual(voltage, current, params)
    # Sub-pA/cm^2 floor: ~1e-16 relative on currents of at most tens of mA/cm^2,
    # with a small allowance for exp() rounding amplified by arguments ~ 25.
    assert np.abs(residual).max() < 1e-12


@pytest.mark.parametrize("params", [SILICON, PEROVSKITE, LOSSY],
                         ids=["silicon", "perovskite", "lossy"])
def test_dark_curve_also_satisfies_implicit_equation(params):
    dark_params = DiodeParams(j_ph=0.0, j_0=params.j_0, n=params.n,
                              r_s=params.r_s, r_sh=params.r_sh,
                              temp_k=params.temp_k)
    voltage, current = iv_curve(params, n_points=300, dark=True)
    residual = implicit_residual(voltage, current, dark_params)
    # Dark sweeps run to v_max unconditionally, so currents reach ~1 A/cm^2
    # through R_s at 1.2 V; the machine-precision floor scales accordingly.
    assert np.abs(residual).max() < 1e-12


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------

def test_rs_zero_branch_is_continuous():
    """The explicit r_s == 0 branch is the limit of the Lambert-W branch."""
    voltage = np.linspace(0.0, 0.7, 200)
    base = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.0, r_sh=2000.0)
    j_exact = solve_current(voltage, base)
    for r_s in (1e-6, 1e-8, 1e-10):
        tiny = DiodeParams(j_ph=base.j_ph, j_0=base.j_0, n=base.n,
                           r_s=r_s, r_sh=base.r_sh)
        j_near = solve_current(voltage, tiny)
        # dJ/dRs is bounded by ~J^2/(n Vt) ~ 0.05 A/cm^2 per Ohm.cm^2 here,
        # so the difference must shrink linearly with r_s.
        assert np.abs(j_near - j_exact).max() < 0.1 * r_s + 1e-15


def test_rsh_infinity_recovers_ideal_shockley_diode():
    """R_s = 0 and R_sh -> inf leave the ideal Shockley curve."""
    voltage = np.linspace(0.0, 0.65, 100)
    params = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.0, r_sh=1e12)
    nvt = params.n * thermal_voltage(params.temp_k)
    ideal = params.j_ph - params.j_0 * (np.exp(voltage / nvt) - 1.0)
    j = solve_current(voltage, params)
    # Residual leakage V/Rsh <= 0.65 / 1e12 < 1e-12 A/cm^2.
    assert np.abs(j - ideal).max() < 1e-12


def test_voc_is_exactly_independent_of_rs():
    """At J = 0 there is no series drop, so Voc cannot depend on R_s."""
    def voc_of(params):
        voltage, current = iv_curve(params, n_points=2000)
        # Refine by root-finding on the solver itself, not the sampled grid.
        return brentq(lambda v: solve_current(np.array([v]), params)[0],
                      voltage[0], voltage[-1], xtol=1e-14)

    base = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.0, r_sh=2000.0)
    voc0 = voc_of(base)
    for r_s in (0.5, 2.0, 10.0):
        shifted = DiodeParams(j_ph=base.j_ph, j_0=base.j_0, n=base.n,
                              r_s=r_s, r_sh=base.r_sh)
        assert abs(voc_of(shifted) - voc0) < 1e-12


def test_voc_matches_analytic_formula_for_large_rsh():
    """Voc -> n Vt ln(J_ph/J_0 + 1) as shunt leakage vanishes."""
    params = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.8, r_sh=1e9)
    nvt = params.n * thermal_voltage(params.temp_k)
    analytic = nvt * np.log(params.j_ph / params.j_0 + 1.0)
    voc = brentq(lambda v: solve_current(np.array([v]), params)[0],
                 0.1, 1.5, xtol=1e-14)
    assert abs(voc - analytic) < 1e-9


def test_superposition_is_exact_at_rs_zero_and_broken_otherwise():
    """J_light(V) - J_dark(V) = J_ph exactly iff R_s = 0."""
    voltage = np.linspace(0.0, 0.6, 100)
    no_rs = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.0, r_sh=2000.0)
    dark_no_rs = DiodeParams(j_ph=0.0, j_0=1e-12, n=1.1, r_s=0.0, r_sh=2000.0)
    gap = solve_current(voltage, no_rs) - solve_current(voltage, dark_no_rs)
    # Exact up to one rounding of the subtraction: eps * J_ph ~ 8e-18.
    np.testing.assert_allclose(gap, no_rs.j_ph, rtol=0, atol=1e-16)

    with_rs = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=1.0, r_sh=2000.0)
    dark_rs = DiodeParams(j_ph=0.0, j_0=1e-12, n=1.1, r_s=1.0, r_sh=2000.0)
    gap_rs = solve_current(voltage, with_rs) - solve_current(voltage, dark_rs)
    # Near Voc the junction voltages differ by ~J*Rs = 36 mV, so superposition
    # must fail detectably (the standard caveat on dark-curve analysis).
    assert np.abs(gap_rs - with_rs.j_ph).max() > 1e-4


def test_j0_to_zero_squares_the_curve_and_climbs_the_voc_ladder():
    """Each decade of J_0 buys n Vt ln(10) of Voc; the FF rises towards 1."""
    nvt = 1.1 * thermal_voltage(298.15)
    voc_values, ff_values = [], []
    for j_0 in (1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        params = DiodeParams(j_ph=36e-3, j_0=j_0, n=1.1, r_s=0.0, r_sh=1e9)
        voltage, current = iv_curve(params, n_points=3000)
        metrics = key_metrics(voltage, current)
        voc_values.append(metrics["voc"])
        ff_values.append(metrics["fill_factor"])
    steps = np.diff(voc_values)
    np.testing.assert_allclose(steps, nvt * np.log(10.0), rtol=1e-3)
    assert np.all(np.diff(ff_values) > 0)


def test_current_is_strictly_decreasing_in_voltage():
    """dJ/dV < 0 everywhere: the forward model is monotone (no S-shapes)."""
    for params in (SILICON, PEROVSKITE, LOSSY):
        voltage, current = iv_curve(params, n_points=500)
        assert np.all(np.diff(current) < 0)


# ---------------------------------------------------------------------------
# iv_curve sweep behaviour
# ---------------------------------------------------------------------------

def test_iv_curve_auto_extends_to_capture_high_voc():
    """A wide-gap cell with Voc > 1.2 V must not be truncated at v_max."""
    params = DiodeParams(j_ph=20e-3, j_0=1e-23, n=1.0, r_s=1.0, r_sh=5000.0)
    voltage, current = iv_curve(params, v_max=1.2, n_points=200)
    assert voltage[-1] > 1.2          # the sweep extended itself
    assert current[-1] <= 0           # ... far enough to cross Voc
    metrics = key_metrics(voltage, current)
    nvt = params.n * thermal_voltage(params.temp_k)
    analytic_voc = nvt * np.log(params.j_ph / params.j_0 + 1.0)
    assert abs(metrics["voc"] - analytic_voc) < 5e-3


def test_iv_curve_dark_flag_zeroes_photocurrent_only():
    voltage, current = iv_curve(SILICON, n_points=100, dark=True)
    explicit_dark = DiodeParams(j_ph=0.0, j_0=SILICON.j_0, n=SILICON.n,
                                r_s=SILICON.r_s, r_sh=SILICON.r_sh)
    np.testing.assert_array_equal(current, solve_current(voltage, explicit_dark))
    assert np.all(current <= 0)


# ---------------------------------------------------------------------------
# key_metrics and local_ideality_factor
# ---------------------------------------------------------------------------

def test_key_metrics_hand_computed_triangle():
    """A 3-point curve whose metrics are computable by hand."""
    voltage = np.array([0.0, 0.5, 1.0])
    current = np.array([2.0, 1.0, 0.0])
    m = key_metrics(voltage, current)
    assert m["jsc"] == pytest.approx(2.0)
    assert m["voc"] == pytest.approx(1.0)
    assert m["pmax"] == pytest.approx(0.5)       # power = (0, 0.5, 0)
    assert m["vmp"] == pytest.approx(0.5)
    assert m["jmp"] == pytest.approx(1.0)
    assert m["fill_factor"] == pytest.approx(0.25)
    assert m["efficiency"] == pytest.approx(5.0)  # 0.5 W/cm^2 vs 0.1 W/cm^2 in


def test_jsc_sits_just_below_jph_for_real_devices():
    """At V = 0 the series/shunt drop steals a little current: 0 < J_ph - Jsc << J_ph."""
    for params in (SILICON, LOSSY):
        jsc = solve_current(np.array([0.0]), params)[0]
        assert jsc < params.j_ph
        assert params.j_ph - jsc < 0.05 * params.j_ph


def test_local_ideality_factor_plateaus_at_n():
    """On an ideal-diode dark curve m(V) must sit at n in the exponential region."""
    n_true = 1.37
    params = DiodeParams(j_ph=0.0, j_0=1e-12, n=n_true, r_s=0.0, r_sh=1e12)
    voltage = np.linspace(0.0, 0.6, 600)
    current = solve_current(voltage, params)
    m = local_ideality_factor(voltage, current, params.temp_k, j_ph=0.0)
    window = (voltage > 0.3) & (voltage < 0.55)
    assert np.nanmax(np.abs(m[window] - n_true)) < 0.01


def test_local_ideality_factor_reads_shunt_as_v_over_vt():
    """In the ohmic (shunt-dominated) region m(V) = V / Vt exactly.

    For pure leakage J = V/R_sh, d(ln J)/dV = 1/V, so m = V/Vt — a straight
    line through the origin with slope 1/Vt, not a plateau. This is the
    signature by which the m(V) plot flags shunt-dominated data, and it is the
    dark low-voltage region whose slope reads the shunt (the tandem chapters
    lean on exactly this).
    """
    params = DiodeParams(j_ph=0.0, j_0=1e-12, n=1.0, r_s=0.0, r_sh=500.0)
    voltage = np.linspace(0.0, 0.6, 600)
    vt = thermal_voltage(params.temp_k)
    current = solve_current(voltage, params)
    m = local_ideality_factor(voltage, current, params.temp_k, j_ph=0.0)
    low = (voltage > 0.05) & (voltage < 0.2)     # diode current ~1e5x smaller
    np.testing.assert_allclose(m[low], voltage[low] / vt, rtol=5e-3)


def test_local_ideality_factor_masks_below_floor():
    """Points with |J_rec| below j_floor come back NaN, not spikes."""
    voltage = np.linspace(0.0, 0.4, 100)
    params = DiodeParams(j_ph=0.0, j_0=1e-12, n=1.0, r_s=0.0, r_sh=1e12)
    current = solve_current(voltage, params)
    m = local_ideality_factor(voltage, current, params.temp_k,
                              j_ph=0.0, j_floor=1e-6)
    assert np.isnan(m[np.abs(current) < 1e-6]).all()


# ---------------------------------------------------------------------------
# Documented limitation: Lambert-W argument overflow at extreme forward bias
# ---------------------------------------------------------------------------

def test_lambert_w_argument_overflows_at_extreme_bias():
    """exp(b) overflows for V far beyond Voc: the closed form returns non-finite.

    This is a *documented boundary*, not desired behaviour: for silicon-like
    parameters the overflow sits near b ~ 709 (float64 exp limit), i.e. tens of
    volts — far outside any measurement — but the tandem chapters must not feed
    such voltages to the per-subcell solver. If this test ever fails because the
    result became finite, an overflow guard was added: update this test and the
    book text together.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        j = solve_current(np.array([25.0]), SILICON)
    assert not np.isfinite(j[0])

    # ... while the whole measurable range stays finite.
    j_ok = solve_current(np.linspace(0.0, 2.0, 50), SILICON)
    assert np.all(np.isfinite(j_ok))