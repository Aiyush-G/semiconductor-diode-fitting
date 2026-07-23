"""Tests for the generated tandem example datasets."""

import numpy as np

from src.models.examples_tandem import (
    TANDEM_DARK_JV_EXAMPLE,
    TANDEM_EXAMPLE_DATASETS,
    TANDEM_LIGHT_JV_EXAMPLE,
    tandem_voltage,
)
from src.models.tandem_fitting import default_tandem_specs, fit_tandem


def test_selector_mapping_contains_both_examples():
    assert TANDEM_EXAMPLE_DATASETS == {
        "Example: Tandem Light JV": TANDEM_LIGHT_JV_EXAMPLE,
        "Example: Tandem Dark JV": TANDEM_DARK_JV_EXAMPLE,
    }


def test_light_example_is_valid_and_tandem_scaled():
    ds = TANDEM_LIGHT_JV_EXAMPLE
    assert ds.kind == "light"
    assert ds.voltage.size > 30
    assert np.all(np.diff(ds.voltage) > 0)
    # Tandem signature: Voc well above a single junction, Jsc ~ 20 mA/cm².
    assert ds.voltage.max() > 1.5
    assert 0.015 < ds.current[0] < 0.025


def test_dark_example_follows_the_dark_sign_convention():
    ds = TANDEM_DARK_JV_EXAMPLE
    assert ds.kind == "dark"
    assert ds.voltage.size > 30
    assert np.all(ds.current <= 0)  # stored as -abs (forward injection)
    assert np.all(np.diff(ds.voltage) > 0)


def test_generating_parameters_are_recoverable_from_the_light_example():
    # The example was generated from TANDEM_DEFAULT_INITIAL, so a fit freeing a
    # small subset from those defaults must converge with small residuals.
    ds = TANDEM_LIGHT_JV_EXAMPLE
    specs = default_tandem_specs("light", free={"top_j_0", "top_n", "bot_j_0"})
    result = fit_tandem(ds.voltage, ds.current, ds.temp_k, specs, kind="light")
    assert result.success
    assert result.r_squared > 0.995
