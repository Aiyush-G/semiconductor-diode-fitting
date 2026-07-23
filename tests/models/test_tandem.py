"""Tests for the 2-terminal tandem model (series stack, current matching)."""

import numpy as np
import pytest
from scipy.special import lambertw

from src.models.single_diode import DiodeParams, iv_curve, key_metrics, solve_current
from src.models.tandem import (
    TandemParams,
    _log_lambertw_exp_large,
    solve_tandem_current,
    solve_voltage,
    tandem_iv_curve,
    tandem_subcell_curves,
    tandem_voltage,
)

TOP = DiodeParams(j_ph=0.020, j_0=1e-16, n=1.5, r_s=1.0, r_sh=2000.0)
BOTTOM = DiodeParams(j_ph=0.0195, j_0=1e-13, n=1.0, r_s=0.5, r_sh=5000.0)
STACK = TandemParams(top=TOP, bottom=BOTTOM)


# --- Lambert W in log space --------------------------------------------------

@pytest.mark.parametrize("x", [600.0, 1e3, 1e4, 1e6])
def test_log_lambertw_satisfies_defining_equation_for_large_args(x):
    # W(e^x) satisfies w + ln(w) = x; the direct exp would overflow here.
    ln_w = float(_log_lambertw_exp_large(np.array([x]))[0])
    w = np.exp(ln_w)
    assert w + ln_w == pytest.approx(x, rel=1e-12)


@pytest.mark.parametrize("x", [300.0, 450.0, 500.0])
def test_log_lambertw_matches_scipy_where_both_are_valid(x):
    # Below the overflow threshold both evaluations work; they must agree, so
    # the branch switch in solve_voltage is seamless.
    ln_w = float(_log_lambertw_exp_large(np.array([x]))[0])
    assert np.exp(ln_w) == pytest.approx(float(lambertw(np.exp(x)).real), rel=1e-12)


# --- Inverse solver round trip ----------------------------------------------

@pytest.mark.parametrize("r_sh", [1e2, 1e4, 1e6, 1e8])
@pytest.mark.parametrize("r_s", [0.0, 0.5, 5.0])
@pytest.mark.parametrize("n", [1.0, 2.0])
def test_solve_voltage_round_trip(r_sh, r_s, n):
    """solve_current(solve_voltage(J)) must recover J across the parameter box.

    Large r_sh drives the W exponent far past float64 exp overflow, so this
    exercises both branches of solve_voltage.
    """
    params = DiodeParams(j_ph=0.02, j_0=1e-13, n=n, r_s=r_s, r_sh=r_sh)
    current = np.linspace(-0.002, 0.0198, 40)
    voltage = solve_voltage(current, params)
    assert np.all(np.isfinite(voltage))
    np.testing.assert_allclose(solve_current(voltage, params), current, atol=1e-12)


def test_solve_voltage_at_zero_current_matches_single_cell_voc():
    # V(J=0) is the exact Voc; cross-check against the forward-model sweep.
    for params in (TOP, BOTTOM):
        voc_exact = float(solve_voltage(np.array([0.0]), params)[0])
        v, j = iv_curve(params, n_points=2000)
        voc_swept = key_metrics(v, j)["voc"]
        assert voc_exact == pytest.approx(voc_swept, abs=2e-3)


# --- Tandem physics ----------------------------------------------------------

def test_tandem_voc_is_sum_of_subcell_vocs():
    voc_top = float(solve_voltage(np.array([0.0]), TOP)[0])
    voc_bot = float(solve_voltage(np.array([0.0]), BOTTOM)[0])
    v, j = tandem_iv_curve(STACK)
    voc_tandem = key_metrics(v, j)["voc"]
    assert voc_tandem == pytest.approx(voc_top + voc_bot, abs=1e-3)


def test_identical_subcells_split_voltage_equally():
    stack = TandemParams(top=BOTTOM, bottom=BOTTOM)
    current = np.linspace(-0.001, 0.019, 30)
    np.testing.assert_allclose(
        tandem_voltage(current, stack), 2.0 * solve_voltage(current, BOTTOM)
    )


def test_tandem_jsc_near_smaller_photocurrent():
    """Jsc sits slightly *above* min(j_ph): with no breakdown model, the
    current-limited sub-cell is pushed into reverse bias at V=0 and its shunt
    resistance passes the extra current."""
    v, j = tandem_iv_curve(STACK)
    jsc = key_metrics(v, j)["jsc"]
    j_limit = min(TOP.j_ph, BOTTOM.j_ph)
    assert j_limit < jsc < max(TOP.j_ph, BOTTOM.j_ph)
    assert jsc == pytest.approx(j_limit, rel=0.05)


def test_tandem_jsc_follows_the_limiting_subcell():
    starved_bottom = DiodeParams(
        j_ph=0.015, j_0=BOTTOM.j_0, n=BOTTOM.n, r_s=BOTTOM.r_s, r_sh=BOTTOM.r_sh
    )
    v, j = tandem_iv_curve(TandemParams(top=TOP, bottom=starved_bottom))
    jsc = key_metrics(v, j)["jsc"]
    assert 0.015 < jsc < TOP.j_ph
    assert jsc == pytest.approx(0.015, rel=0.05)


def test_tandem_iv_curve_shape():
    v, j = tandem_iv_curve(STACK)
    assert np.all(np.diff(v) >= 0)
    assert v[0] == pytest.approx(0.0, abs=2e-4)
    assert j[0] > 0  # Jsc end
    assert np.min(j) <= 0  # curve reaches/crosses zero current at Voc
    # Current is monotone non-increasing with voltage (series diode stack).
    assert np.all(np.diff(j) <= 1e-12)


def test_tandem_dark_curve_is_forward_injection():
    v, j = tandem_iv_curve(STACK, dark=True)
    assert np.all(np.isfinite(j))
    assert np.all(j <= 1e-12)  # no photocurrent: terminal current <= 0
    # Spans the same voltage window as the light curve.
    v_light, _ = tandem_iv_curve(STACK)
    assert v[-1] == pytest.approx(v_light[-1], abs=1e-3)


def test_subcell_curves_sum_to_tandem_voltage():
    curves = tandem_subcell_curves(STACK)
    (label_top, v_top, j_top), (label_bot, v_bot, j_bot) = curves
    assert label_top == "Top cell" and label_bot == "Bottom cell"
    np.testing.assert_allclose(j_top, j_bot)  # shared current grid
    np.testing.assert_allclose(v_top + v_bot, tandem_voltage(j_top, STACK))


# --- Forward model for fitting ----------------------------------------------

def test_solve_tandem_current_inverts_the_curve():
    v, j = tandem_iv_curve(STACK)
    j_pred = solve_tandem_current(v, STACK)
    np.testing.assert_allclose(j_pred, j, atol=1e-5)


def test_solve_tandem_current_handles_voltages_beyond_the_curve():
    voltage = np.array([-0.1, 0.0, 1.0, 2.5])  # below 0 and above Voc
    j = solve_tandem_current(voltage, STACK)
    assert np.all(np.isfinite(j))
    assert np.all(np.diff(j) <= 0)  # still monotone in V
