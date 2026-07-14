"""Tests for parsing, unit conversion, sign handling, and validation."""

import numpy as np
import pytest

from src.models.data_import import (
    DataImportError,
    build_dataset,
    normalize,
    parse_two_column,
    to_base_units,
)

# The same five (V, J) points expressed with different delimiters / decoration.
_ROWS = [(0.0, 0.001), (0.1, 0.002), (0.2, 0.004), (0.3, 0.008), (0.4, 0.016)]
_EXPECTED_V = np.array([r[0] for r in _ROWS])
_EXPECTED_J = np.array([r[1] for r in _ROWS])


@pytest.mark.parametrize(
    "text",
    [
        "0 0.001\n0.1 0.002\n0.2 0.004\n0.3 0.008\n0.4 0.016",  # whitespace
        "0\t0.001\n0.1\t0.002\n0.2\t0.004\n0.3\t0.008\n0.4\t0.016",  # tab
        "0,0.001\n0.1,0.002\n0.2,0.004\n0.3,0.008\n0.4,0.016",  # comma
        "0;0.001\n0.1;0.002\n0.2;0.004\n0.3;0.008\n0.4;0.016",  # semicolon
        "0 ,\t0.001\n0.1,  0.002\n0.2\t 0.004\n0.3 , 0.008\n0.4  0.016",  # mixed
        # header + comment + blank + trailing blanks
        "V, J\n# measured\n0,0.001\n0.1,0.002\n0.2,0.004\n0.3,0.008\n0.4,0.016\n\n",
        # scientific notation
        "0,1e-3\n0.1,2e-3\n0.2,4e-3\n0.3,8e-3\n0.4,1.6e-2",
    ],
)
def test_parse_delimiters_and_decoration(text):
    v, j = parse_two_column(text)
    np.testing.assert_allclose(v, _EXPECTED_V)
    np.testing.assert_allclose(j, _EXPECTED_J)


def test_unit_conversion_round_trip():
    v = np.array([10.0, 20.0, 30.0])  # mV
    j = np.array([1.0, 2.0, 3.0])  # mA/cm2
    v_base, j_base = to_base_units(v, j, "mV", "mA/cm2")
    np.testing.assert_allclose(v_base, v * 1e-3)
    np.testing.assert_allclose(j_base, j * 1e-3)

    # V / A pass through unchanged.
    v2, j2 = to_base_units(v, j, "V", "A/cm2")
    np.testing.assert_allclose(v2, v)
    np.testing.assert_allclose(j2, j)


def test_mv_and_ma_import_matches_v_and_a():
    text_mv_ma = "0,1\n100,2\n200,4\n300,8\n400,16"  # mV, mA/cm2
    text_v_a = "0,0.001\n0.1,0.002\n0.2,0.004\n0.3,0.008\n0.4,0.016"  # V, A/cm2
    ds_mv = build_dataset(text_mv_ma, label="a", kind="light",
                          voltage_units="mV", current_units="mA/cm2")
    ds_v = build_dataset(text_v_a, label="b", kind="light",
                         voltage_units="V", current_units="A/cm2")
    np.testing.assert_allclose(ds_mv.voltage, ds_v.voltage)
    np.testing.assert_allclose(ds_mv.current, ds_v.current)


def test_dark_sign_is_negative():
    # Positive dark magnitudes become negative (model sign convention).
    _, j = normalize(np.array([0.1, 0.2, 0.3]), np.array([1e-6, 2e-6, 4e-6]), "dark")
    assert np.all(j <= 0)
    # Light keeps its sign.
    _, jl = normalize(np.array([0.0, 0.1]), np.array([0.03, -0.01]), "light")
    np.testing.assert_allclose(jl, np.array([0.03, -0.01]))


def test_sorts_by_voltage():
    text = "0.4,0.016\n0.0,0.001\n0.2,0.004\n0.1,0.002\n0.3,0.008"
    ds = build_dataset(text, label="x", kind="light")
    assert np.all(np.diff(ds.voltage) > 0)
    # raw arrays preserve the original (unsorted) order
    assert ds.raw_voltage[0] == 0.4


def test_preserves_raw_units():
    text = "0,1\n100,2\n200,4\n300,8\n400,16"
    ds = build_dataset(text, label="x", kind="light",
                       voltage_units="mV", current_units="mA/cm2")
    # raw stays in the imported units; normalized is converted to base units
    np.testing.assert_allclose(ds.raw_voltage, [0, 100, 200, 300, 400])
    np.testing.assert_allclose(ds.voltage, [0, 0.1, 0.2, 0.3, 0.4])


def test_rejects_too_few_points():
    with pytest.raises(DataImportError, match="at least"):
        build_dataset("0,1\n0.1,2\n0.2,3", label="x", kind="light")


def test_rejects_duplicate_voltage():
    text = "0,1\n0.1,2\n0.1,3\n0.2,4\n0.3,5"
    with pytest.raises(DataImportError, match="[Dd]uplicate"):
        build_dataset(text, label="x", kind="light")


def test_rejects_non_finite():
    text = "0,1\n0.1,2\n0.2,nan\n0.3,4\n0.4,5"
    with pytest.raises(DataImportError, match="non-finite"):
        build_dataset(text, label="x", kind="light")


def test_rejects_single_column():
    with pytest.raises(DataImportError):
        parse_two_column("1\n2\n3\n4\n5")


def test_rejects_empty():
    with pytest.raises(DataImportError):
        parse_two_column("   \n  \n")


def test_rejects_non_numeric_row():
    with pytest.raises(DataImportError):
        parse_two_column("0,1\n0.1,abc\n0.2,3")


def test_unknown_units_raise():
    with pytest.raises(DataImportError):
        to_base_units(np.array([1.0]), np.array([1.0]), "kV", "A/cm2")
