"""Statistical and provenance tests for synthetic single-diode data."""

import numpy as np
import pytest

from src.models.fitting import default_specs, fit_diode
from src.models.single_diode import DiodeParams, solve_current
from src.models.synthetic import (
    GaussianNoise,
    LogNormalNoise,
    NoNoise,
    StudentTNoise,
    _draw_noisy_current,
    generate_synthetic,
)


TRUTH = DiodeParams(
    j_ph=0.036,
    j_0=1e-12,
    n=1.1,
    r_s=0.8,
    r_sh=2000.0,
    temp_k=298.15,
)


def test_no_noise_matches_forward_model_exactly():
    voltage = np.linspace(0.0, 0.68, 80)
    dataset = generate_synthetic(TRUTH, voltage)

    expected = solve_current(voltage, TRUTH)
    np.testing.assert_array_equal(dataset.current, expected)
    np.testing.assert_array_equal(dataset.exact_current, expected)
    np.testing.assert_array_equal(dataset.noise, 0.0)
    assert isinstance(dataset.noise_model, NoNoise)
    assert dataset.n_points == voltage.size


def test_dark_generation_forces_photocurrent_to_zero():
    voltage = np.linspace(0.01, 0.72, 90)
    dataset = generate_synthetic(TRUTH, voltage, kind="dark")
    dark_truth = DiodeParams(
        j_ph=0.0,
        j_0=TRUTH.j_0,
        n=TRUTH.n,
        r_s=TRUTH.r_s,
        r_sh=TRUTH.r_sh,
        temp_k=TRUTH.temp_k,
    )

    assert dataset.params.j_ph == 0.0
    assert dataset.kind == "dark"
    np.testing.assert_array_equal(dataset.current, solve_current(voltage, dark_truth))
    assert np.all(dataset.current < 0.0)


def test_equal_seed_is_bitwise_reproducible_and_different_seed_changes_data():
    voltage = np.linspace(0.0, 0.68, 80)
    model = GaussianNoise(sigma_a_per_cm2=2e-5)
    first = generate_synthetic(TRUTH, voltage, noise_model=model, seed=1234)
    second = generate_synthetic(TRUTH, voltage, noise_model=model, seed=1234)
    third = generate_synthetic(TRUTH, voltage, noise_model=model, seed=1235)

    np.testing.assert_array_equal(first.current, second.current)
    assert not np.array_equal(first.current, third.current)


def test_gaussian_noise_recovers_requested_mean_and_standard_deviation():
    n_repeats = 100_000
    sigma = 2e-5
    voltage = np.full(n_repeats, 0.40)
    dataset = generate_synthetic(
        TRUTH,
        voltage,
        noise_model=GaussianNoise(sigma_a_per_cm2=sigma),
        seed=20260720,
    )

    standard_error_of_mean = sigma / np.sqrt(n_repeats)
    assert abs(np.mean(dataset.noise)) < 5.0 * standard_error_of_mean
    assert np.std(dataset.noise, ddof=1) == pytest.approx(sigma, rel=0.01)


def test_lognormal_noise_has_normal_log_ratio_and_preserves_dark_sign():
    n_repeats = 100_000
    sigma_ln = 0.03
    voltage = np.full(n_repeats, 0.50)
    dataset = generate_synthetic(
        TRUTH,
        voltage,
        kind="dark",
        noise_model=LogNormalNoise(sigma_ln=sigma_ln),
        seed=7,
    )
    log_ratio = np.log(np.abs(dataset.current) / np.abs(dataset.exact_current))

    assert np.all(dataset.current < 0.0)
    assert abs(np.mean(log_ratio)) < 5.0 * sigma_ln / np.sqrt(n_repeats)
    assert np.std(log_ratio, ddof=1) == pytest.approx(sigma_ln, rel=0.01)


def test_lognormal_noise_leaves_an_exact_zero_at_zero():
    exact = np.array([-1.0, 0.0, 1.0])
    measured = _draw_noisy_current(
        exact,
        LogNormalNoise(sigma_ln=0.2),
        np.random.default_rng(11),
    )

    assert measured[0] < 0.0
    assert measured[1] == 0.0
    assert measured[2] > 0.0


def test_student_t_noise_recovers_finite_variance_when_df_exceeds_two():
    n_repeats = 200_000
    scale = 2e-5
    degrees_of_freedom = 5.0
    voltage = np.full(n_repeats, 0.40)
    dataset = generate_synthetic(
        TRUTH,
        voltage,
        noise_model=StudentTNoise(scale, degrees_of_freedom),
        seed=99,
    )
    expected_std = scale * np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))

    assert abs(np.mean(dataset.noise)) < 5.0 * expected_std / np.sqrt(n_repeats)
    assert np.std(dataset.noise, ddof=1) == pytest.approx(expected_std, rel=0.02)


def test_student_t_has_more_extreme_draws_than_same_scale_gaussian():
    n_draws = 200_000
    scale = 1.0
    exact = np.zeros(n_draws)
    t_draws = _draw_noisy_current(
        exact, StudentTNoise(scale, degrees_of_freedom=4.0), np.random.default_rng(1)
    )
    gaussian_draws = _draw_noisy_current(
        exact, GaussianNoise(scale), np.random.default_rng(1)
    )

    assert np.count_nonzero(np.abs(t_draws) > 5.0 * scale) > 100
    assert np.count_nonzero(np.abs(gaussian_draws) > 5.0 * scale) < 5


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: GaussianNoise(-1.0), "sigma_a_per_cm2"),
        (lambda: GaussianNoise(np.inf), "sigma_a_per_cm2"),
        (lambda: LogNormalNoise(-0.1), "sigma_ln"),
        (lambda: StudentTNoise(-1.0, 4.0), "scale_a_per_cm2"),
        (lambda: StudentTNoise(1.0, 0.0), "degrees_of_freedom"),
    ],
)
def test_invalid_noise_parameters_are_rejected(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()


@pytest.mark.parametrize(
    "voltage, message",
    [
        (np.array([]), "non-empty"),
        (np.zeros((2, 2)), "one-dimensional"),
        (np.array([0.0, np.nan]), "finite"),
    ],
)
def test_invalid_voltage_design_is_rejected(voltage, message):
    with pytest.raises(ValueError, match=message):
        generate_synthetic(TRUTH, voltage)


def test_unknown_kind_and_noise_type_are_rejected():
    with pytest.raises(ValueError, match="Unknown kind"):
        generate_synthetic(TRUTH, np.array([0.0]), kind="moonlight")
    with pytest.raises(TypeError, match="noise_model must be"):
        generate_synthetic(TRUTH, np.array([0.0]), noise_model=object())


def test_returned_arrays_are_copies_and_read_only():
    voltage = np.linspace(0.0, 0.6, 10)
    dataset = generate_synthetic(TRUTH, voltage)
    voltage[0] = 99.0

    assert dataset.voltage[0] == 0.0
    for values in (dataset.voltage, dataset.current, dataset.exact_current, dataset.noise):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 0.0


def test_synthetic_dataset_is_compatible_with_existing_fitter():
    voltage = np.linspace(0.0, 0.68, 100)
    dataset = generate_synthetic(
        TRUTH,
        voltage,
        noise_model=GaussianNoise(sigma_a_per_cm2=1e-5),
        seed=31415,
    )
    specs = default_specs(
        "light",
        free={"j_ph"},
        initial={
            "j_ph": 0.034,
            "j_0": TRUTH.j_0,
            "n": TRUTH.n,
            "r_s": TRUTH.r_s,
            "r_sh": TRUTH.r_sh,
        },
    )
    fit = fit_diode(
        dataset.voltage,
        dataset.current,
        TRUTH.temp_k,
        specs,
        kind="light",
    )

    assert fit.success
    assert fit.params.j_ph == pytest.approx(TRUTH.j_ph, abs=5e-6)
