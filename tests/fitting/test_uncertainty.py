"""Tests for Jacobian-based local uncertainty estimates."""
# pytest -q tests/fitting/test_uncertainty.py tests/models

import numpy as np
import pytest

from src.fitting.uncertainty import (
    covariance_from_jacobian,
    estimate_fit_uncertainty,
)
from src.models.fitting import default_specs, fit_diode
from src.models.single_diode import DiodeParams, solve_current


def test_well_conditioned_covariance_matches_direct_inverse():
    jacobian = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [2.0, -1.0],
    ])
    residuals = np.array([0.1, -0.1, 0.2, -0.2])
    result = covariance_from_jacobian(
        jacobian, residuals, np.array([2.0, 3.0]), ("n", "r_s")
    )
    sigma_squared = residuals @ residuals / (4 - 2)
    expected = sigma_squared * np.linalg.inv(jacobian.T @ jacobian)

    np.testing.assert_allclose(result.covariance_fit, expected)
    np.testing.assert_allclose(result.covariance, expected)
    np.testing.assert_allclose(result.standard_errors, np.sqrt(np.diag(expected)))
    assert result.full_rank
    assert result.degrees_of_freedom == 2


def test_log_parameter_covariance_is_transformed_to_natural_units():
    jacobian = np.array([[1.0], [2.0], [3.0]])
    residuals = np.array([0.1, -0.1, 0.1])
    estimate = 1e-12
    result = covariance_from_jacobian(
        jacobian, residuals, np.array([estimate]), ("j_0",)
    )

    scale = np.log(10.0) * estimate
    assert result.standard_errors[0] == pytest.approx(
        result.standard_errors_fit[0] * scale
    )
    assert result.covariance[0, 0] == pytest.approx(
        result.covariance_fit[0, 0] * scale**2
    )


def test_rank_deficiency_is_reported_not_hidden_by_pseudoinverse():
    jacobian = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])
    result = covariance_from_jacobian(
        jacobian,
        np.array([0.1, -0.1, 0.1, -0.1]),
        np.array([1.0, 1.0]),
        ("n", "r_s"),
    )

    assert result.rank == 1
    assert not result.full_rank
    assert np.isinf(result.condition_number)
    assert np.isnan(result.covariance).all()
    assert np.isnan(result.correlation).all()


def test_too_few_residuals_has_no_variance_estimate():
    with pytest.raises(ValueError, match="N > p"):
        covariance_from_jacobian(
            np.eye(2), np.ones(2), np.ones(2), ("n", "r_s")
        )


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="one entry"):
        covariance_from_jacobian(
            np.ones((3, 1)), np.ones(2), np.ones(1), ("n",)
        )


def test_all_fixed_fit_returns_empty_uncertainty():
    params = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.1, r_s=0.8, r_sh=2000.0)
    voltage = np.linspace(0.0, 0.65, 40)
    current = solve_current(voltage, params)
    specs = default_specs(
        "light",
        free=set(),
        initial={name: getattr(params, name) for name in ("j_ph", "j_0", "n", "r_s", "r_sh")},
    )
    fit = fit_diode(voltage, current, params.temp_k, specs, kind="light")
    result = estimate_fit_uncertainty(
        fit, voltage, current, params.temp_k, specs, kind="light"
    )

    assert result.free_names == ()
    assert result.full_rank
    assert result.covariance.shape == (0, 0)
    assert result.degrees_of_freedom == voltage.size


def test_synthetic_fit_uncertainty_is_finite_and_tracks_residual_space():
    rng = np.random.default_rng(20260720)
    truth = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.15, r_s=0.7, r_sh=1800.0)
    voltage = np.linspace(0.0, 0.68, 100)
    current = solve_current(voltage, truth) + rng.normal(0.0, 2e-5, voltage.size)
    specs = default_specs(
        "light",
        free={"j_ph", "j_0", "n", "r_s", "r_sh"},
        initial={"j_ph": 0.035, "j_0": 3e-12, "n": 1.3, "r_s": 1.0, "r_sh": 1000.0},
    )
    fit = fit_diode(voltage, current, truth.temp_k, specs, kind="light")
    result = estimate_fit_uncertainty(
        fit, voltage, current, truth.temp_k, specs, kind="light"
    )

    assert fit.success
    assert result.residual_space == "linear"
    assert result.free_names == fit.free_names
    assert result.covariance.shape == (5, 5)
    assert result.full_rank
    assert np.all(np.isfinite(result.standard_errors))
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    np.testing.assert_allclose(result.correlation, result.correlation.T)
    # The diode knee creates a visible local j0-n trade-off.
    j0 = result.free_names.index("j_0")
    n = result.free_names.index("n")
    assert abs(result.correlation[j0, n]) > 0.9


def test_dark_auto_space_uses_log_residual_jacobian():
    rng = np.random.default_rng(42)
    truth = DiodeParams(j_ph=0.0, j_0=5e-13, n=1.3, r_s=0.4, r_sh=5000.0)
    voltage = np.linspace(0.01, 0.72, 90)
    exact = solve_current(voltage, truth)
    current = exact * np.exp(rng.normal(0.0, 0.002, voltage.size))
    specs = default_specs(
        "dark",
        free={"j_0", "n", "r_s", "r_sh"},
        initial={"j_0": 1e-12, "n": 1.1, "r_s": 0.8, "r_sh": 2500.0},
    )
    fit = fit_diode(voltage, current, truth.temp_k, specs, kind="dark")
    result = estimate_fit_uncertainty(
        fit, voltage, current, truth.temp_k, specs, kind="dark"
    )

    assert fit.success
    assert result.residual_space == "log"
    assert result.full_rank
    assert np.all(np.isfinite(result.standard_errors_fit))


def test_relative_step_must_be_positive():
    params = DiodeParams(j_ph=0.036, j_0=1e-12, n=1.1, r_s=0.8, r_sh=2000.0)
    voltage = np.linspace(0.0, 0.65, 30)
    current = solve_current(voltage, params)
    specs = default_specs("light", free={"j_ph", "n"})
    fit = fit_diode(voltage, current, params.temp_k, specs, kind="light")
    with pytest.raises(ValueError, match="positive"):
        estimate_fit_uncertainty(
            fit, voltage, current, params.temp_k, specs,
            kind="light", relative_step=0.0,
        )
