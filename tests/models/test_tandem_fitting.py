"""Tests for the tandem (10-parameter) least-squares fitting module."""

import numpy as np
import pytest

from src.models.fitting import pack
from src.models.tandem import TandemParams, solve_tandem_current, tandem_iv_curve
from src.models.tandem_fitting import (
    TANDEM_DEFAULT_INITIAL,
    TANDEM_PARAM_NAMES,
    default_tandem_specs,
    fit_tandem,
    unpack_tandem,
)


def _true_stack(temp_k: float = 298.15) -> TandemParams:
    specs = default_tandem_specs("light", free=set())
    return unpack_tandem(np.empty(0), specs, temp_k)


def test_param_names_cover_both_subcells():
    assert TANDEM_PARAM_NAMES == (
        "top_j_ph", "top_j_0", "top_n", "top_r_s", "top_r_sh",
        "bot_j_ph", "bot_j_0", "bot_n", "bot_r_s", "bot_r_sh",
    )


def test_dark_specs_force_both_photocurrents_fixed_zero():
    specs = default_tandem_specs("dark", free={"top_j_ph", "bot_j_ph", "top_j_0"})
    for name in ("top_j_ph", "bot_j_ph"):
        assert specs[name].free is False
        assert specs[name].value == 0.0
    assert specs["top_j_0"].free is True


def test_pack_orders_free_names_canonically():
    specs = default_tandem_specs(
        "light", free={"bot_n", "top_j_0", "bot_j_0", "top_n"}
    )
    _, _, _, free_names = pack(specs, TANDEM_PARAM_NAMES)
    assert free_names == ("top_j_0", "top_n", "bot_j_0", "bot_n")


def test_unpack_tandem_round_trip():
    specs = default_tandem_specs("light", free={"top_j_0", "bot_r_sh"})
    theta0, lower, upper, _ = pack(specs, TANDEM_PARAM_NAMES)
    stack = unpack_tandem(theta0, specs, temp_k=298.15)
    assert stack.top.j_0 == pytest.approx(TANDEM_DEFAULT_INITIAL["top_j_0"])
    assert stack.bottom.r_sh == pytest.approx(TANDEM_DEFAULT_INITIAL["bot_r_sh"])
    assert stack.top.temp_k == 298.15
    # Log-space parameters have log10-transformed bounds.
    assert lower[0] == pytest.approx(np.log10(specs["top_j_0"].lower))
    assert upper[1] == pytest.approx(np.log10(specs["bot_r_sh"].upper))


def test_fixed_parameters_stay_exactly_unchanged():
    true = _true_stack()
    v, j = tandem_iv_curve(true, n_points=80)

    fixed_r_s = 0.77
    specs = default_tandem_specs(
        "light",
        free={"top_j_0", "top_n"},
        initial={"top_r_s": fixed_r_s},
    )
    result = fit_tandem(v, j, 298.15, specs, kind="light")
    assert result.params.top.r_s == fixed_r_s  # exact, not approx
    assert result.params.bottom.j_0 == TANDEM_DEFAULT_INITIAL["bot_j_0"]


def test_all_fixed_skips_optimizer():
    true = _true_stack()
    v, j = tandem_iv_curve(true, n_points=60)
    specs = default_tandem_specs("light", free=set())
    result = fit_tandem(v, j, 298.15, specs, kind="light")
    assert result.success
    assert result.free_names == ()
    assert isinstance(result.params, TandemParams)
    # Exact model, exact params: only the forward-interpolation error remains.
    assert result.rmse < 1e-5


def test_synthetic_recovery_light():
    true = _true_stack()
    v = np.linspace(0.0, 1.85, 90)
    j = solve_tandem_current(v, true)

    # Free a small, identifiable subset from a perturbed start; the rest stay
    # fixed at truth (10 free parameters on one curve is genuinely degenerate).
    specs = default_tandem_specs(
        "light",
        free={"top_j_0", "top_n", "bot_j_0"},
        initial={"top_j_0": 1e-15, "top_n": 1.8, "bot_j_0": 5e-13},
    )
    result = fit_tandem(v, j, 298.15, specs, kind="light")
    assert result.success
    assert result.residual_space == "linear"
    assert result.free_names == ("top_j_0", "top_n", "bot_j_0")
    assert result.params.top.j_0 == pytest.approx(true.top.j_0, rel=5e-2)
    assert result.params.top.n == pytest.approx(true.top.n, rel=1e-2)
    assert result.params.bottom.j_0 == pytest.approx(true.bottom.j_0, rel=5e-2)
    assert result.r_squared > 0.999


def test_synthetic_recovery_dark():
    specs_true = default_tandem_specs("dark", free=set())
    true = unpack_tandem(np.empty(0), specs_true, 298.15)
    v = np.linspace(0.05, 1.6, 90)
    j = solve_tandem_current(v, true)  # negative: forward injection

    specs = default_tandem_specs(
        "dark",
        free={"top_j_0", "bot_j_0"},
        initial={"top_j_0": 1e-15, "bot_j_0": 1e-12},
    )
    result = fit_tandem(v, j, 298.15, specs, kind="dark")
    assert result.success
    assert result.residual_space == "log"  # auto -> log for dark
    assert result.params.top.j_ph == 0.0
    assert result.params.bottom.j_ph == 0.0
    assert result.params.top.j_0 == pytest.approx(true.top.j_0, rel=5e-2)
    assert result.params.bottom.j_0 == pytest.approx(true.bottom.j_0, rel=5e-2)
