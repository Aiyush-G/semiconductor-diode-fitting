"""Tests for the least-squares fitting module."""

import numpy as np
import pytest

from src.models.fitting import (
    default_specs,
    fit_diode,
    pack,
    resolve_residual_space,
    unpack,
)
from src.models.single_diode import DiodeParams, solve_current


def _synthetic(params, v):
    return solve_current(v, params)


def test_default_specs_dark_forces_jph_fixed_zero():
    # Even if the caller asks to fit j_ph on dark data, it must be fixed at 0.
    specs = default_specs("dark", free={"j_ph", "j_0", "n"})
    assert specs["j_ph"].free is False
    assert specs["j_ph"].value == 0.0


def test_light_exposes_all_params():
    specs = default_specs("light", free={"j_ph", "j_0", "n", "r_s", "r_sh"})
    assert specs["j_ph"].free is True
    assert set(n for n, s in specs.items() if s.free) == {
        "j_ph", "j_0", "n", "r_s", "r_sh"
    }


def test_pack_unpack_round_trip():
    specs = default_specs("light", free={"j_0", "n", "r_s", "r_sh"})
    theta0, lower, upper, free_names = pack(specs)
    # Free names in canonical order, log params transformed.
    assert free_names == ("j_0", "n", "r_s", "r_sh")
    params = unpack(theta0, specs, temp_k=298.15)
    assert params.j_0 == pytest.approx(specs["j_0"].value)
    assert params.n == pytest.approx(specs["n"].value)
    assert params.r_sh == pytest.approx(specs["r_sh"].value)
    # log bounds are transformed with log10
    assert lower[0] == pytest.approx(np.log10(specs["j_0"].lower))
    assert upper[0] == pytest.approx(np.log10(specs["j_0"].upper))


def test_fixed_parameter_stays_exactly_unchanged():
    true = DiodeParams(j_ph=0.036, j_0=1e-13, n=1.1, r_s=0.6, r_sh=2000.0)
    v = np.linspace(0, 0.62, 50)
    j = _synthetic(true, v)

    fixed_r_s = 0.42
    specs = default_specs(
        "light",
        free={"j_ph", "j_0", "n", "r_sh"},  # r_s NOT free
        initial={"r_s": fixed_r_s},
    )
    result = fit_diode(v, j, 298.15, specs, kind="light")
    assert result.params.r_s == fixed_r_s  # exact, not approx


def test_synthetic_recovery_light():
    true = DiodeParams(j_ph=0.036, j_0=1e-13, n=1.1, r_s=0.5, r_sh=2500.0)
    v = np.linspace(0, 0.62, 80)
    j = _synthetic(true, v)

    # Fix the poorly-identified resistances at truth; fit the diode terms from a
    # perturbed start. (Fitting all five at once is genuinely degenerate.)
    specs = default_specs(
        "light",
        free={"j_ph", "j_0", "n"},
        initial={"j_ph": 0.03, "j_0": 5e-13, "n": 1.3, "r_s": 0.5, "r_sh": 2500.0},
    )
    result = fit_diode(v, j, 298.15, specs, kind="light")
    assert result.success
    assert result.residual_space == "linear"
    assert result.params.j_ph == pytest.approx(true.j_ph, rel=1e-3)
    assert result.params.j_0 == pytest.approx(true.j_0, rel=5e-2)
    assert result.params.n == pytest.approx(true.n, rel=1e-2)
    assert result.rmse < 1e-5
    assert result.r_squared > 0.999


def test_synthetic_recovery_dark():
    true = DiodeParams(j_ph=0.0, j_0=5e-13, n=1.3, r_s=0.4, r_sh=5000.0)
    v = np.linspace(0.01, 0.72, 90)
    j = _synthetic(true, v)  # already negative (dark forward bias)

    specs = default_specs(
        "dark",
        free={"j_0", "n", "r_s", "r_sh"},
        initial={"j_0": 1e-12, "n": 1.0, "r_s": 0.8, "r_sh": 2000.0},
    )
    result = fit_diode(v, j, 298.15, specs, kind="dark")
    assert result.success
    assert result.residual_space == "log"  # auto -> log for dark
    assert result.params.j_ph == 0.0  # photocurrent never introduced
    assert result.params.j_0 == pytest.approx(true.j_0, rel=5e-2)
    assert result.params.n == pytest.approx(true.n, rel=1e-2)
    assert result.params.r_s == pytest.approx(true.r_s, rel=5e-2)
    assert result.params.r_sh == pytest.approx(true.r_sh, rel=5e-2)
    assert result.rmse_log is not None and result.rmse_log < 1e-3


def test_all_fixed_skips_optimizer():
    true = DiodeParams(j_ph=0.036, j_0=1e-13, n=1.1, r_s=0.5, r_sh=2500.0)
    v = np.linspace(0, 0.6, 20)
    j = _synthetic(true, v)
    specs = default_specs(
        "light",
        free=set(),
        initial={"j_ph": 0.036, "j_0": 1e-13, "n": 1.1, "r_s": 0.5, "r_sh": 2500.0},
    )
    result = fit_diode(v, j, 298.15, specs, kind="light")
    assert result.success
    assert result.free_names == ()
    assert result.rmse < 1e-9  # exact model, exact params


def test_residual_space_auto_resolution():
    assert resolve_residual_space("auto", "dark") == "log"
    assert resolve_residual_space("auto", "light") == "linear"
    assert resolve_residual_space("linear", "dark") == "linear"
    assert resolve_residual_space("log", "light") == "log"
    with pytest.raises(ValueError):
        resolve_residual_space("bogus", "light")


def test_robustness_exception_returns_penalty(monkeypatch):
    """A forward-model exception in the residual must yield a finite penalty."""
    import src.models.fitting as fitting

    v = np.linspace(0.01, 0.6, 20)
    j = np.full_like(v, -1e-4)
    specs = default_specs("dark", free={"j_0", "n"})

    def boom(voltage, params):
        raise OverflowError("simulated overflow")

    monkeypatch.setattr(fitting, "solve_current", boom)
    residuals = fitting._make_residual(v, j, specs, 298.15, "log", fitting.PENALTY)
    theta0, *_ = fitting.pack(specs)
    out = residuals(theta0)
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, fitting.PENALTY)


def test_robustness_nonfinite_output_penalised(monkeypatch):
    """Non-finite forward output is replaced element-wise with the penalty."""
    import src.models.fitting as fitting

    v = np.linspace(0.01, 0.6, 20)
    j = np.full_like(v, -1e-4)
    specs = default_specs("dark", free={"j_0", "n"})

    def partly_bad(voltage, params):
        out = np.full_like(voltage, -1e-4)
        out[0] = np.inf
        return out

    monkeypatch.setattr(fitting, "solve_current", partly_bad)
    residuals = fitting._make_residual(v, j, specs, 298.15, "linear", fitting.PENALTY)
    theta0, *_ = fitting.pack(specs)
    out = residuals(theta0)
    assert np.all(np.isfinite(out))
    assert out[0] == fitting.PENALTY
