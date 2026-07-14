"""Tests for the built-in example datasets."""

import numpy as np

from src.models.examples import DARK_JV_EXAMPLE, EXAMPLE_DATASETS, LIGHT_JV_EXAMPLE
from src.models.fitting import DEFAULT_BOUNDS, default_specs, fit_diode


def test_examples_registered():
    assert set(EXAMPLE_DATASETS) == {"Example: Light JV", "Example: Dark JV"}


def test_light_example_shape_and_units():
    ds = LIGHT_JV_EXAMPLE
    assert ds.kind == "light"
    assert ds.voltage_units == "V"
    assert ds.current_units == "A/cm2"
    assert ds.voltage.size == ds.current.size
    assert ds.voltage.size == ds.raw_voltage.size
    assert np.all(np.diff(ds.voltage) > 0)  # sorted, unique
    # Light current keeps its sign (starts near Jsc > 0, ends at ~0 near Voc).
    assert ds.current[0] > 0


def test_dark_example_shape_and_sign():
    ds = DARK_JV_EXAMPLE
    assert ds.kind == "dark"
    assert ds.voltage.size == ds.current.size
    assert np.all(np.diff(ds.voltage) > 0)
    # Dark current stored in model convention (negative in forward bias).
    assert np.all(ds.current <= 0)


def _within_bounds(params):
    checks = {
        "j_ph": params.j_ph, "j_0": params.j_0, "n": params.n,
        "r_s": params.r_s, "r_sh": params.r_sh,
    }
    for name, value in checks.items():
        lo, hi = DEFAULT_BOUNDS[name]
        assert lo - 1e-12 <= value <= hi + 1e-12, f"{name}={value} out of {(lo, hi)}"


def test_light_example_fits_within_bounds():
    ds = LIGHT_JV_EXAMPLE
    specs = default_specs("light", free={"j_ph", "j_0", "n", "r_s", "r_sh"})
    result = fit_diode(ds.voltage, ds.current, ds.temp_k, specs, kind="light")
    assert result.success
    _within_bounds(result.params)
    # A real single-diode curve should fit well.
    assert result.r_squared > 0.99


def test_dark_example_fits_within_bounds():
    ds = DARK_JV_EXAMPLE
    specs = default_specs("dark", free={"j_0", "n", "r_s", "r_sh"})
    result = fit_diode(ds.voltage, ds.current, ds.temp_k, specs, kind="dark")
    assert result.success
    assert result.params.j_ph == 0.0
    _within_bounds(result.params)
    # Log-space RMSE across the decades should be small for good data.
    assert result.rmse_log is not None and result.rmse_log < 0.2
