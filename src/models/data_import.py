"""
Import and normalise custom two-column J-V (current-density vs voltage) data.

This module turns raw pasted/uploaded text into a validated, model-ready
``ImportedDataset``. It is deliberately UI-free (pure numpy) so the same parsing
and validation can be exercised from tests and reused by any front end, mirroring
the split used by ``single_diode.py`` / ``temperature.py``.

Unit and sign conventions follow the rest of the app (PV Lighthouse convention):

    voltage - stored in volts (V); accepted as V or mV
    current - stored as a density in A/cm^2; accepted as A/cm^2 or mA/cm^2

Sign convention
---------------
The forward model (``single_diode.solve_current``) returns the *terminal* current
density from the cell's perspective. For a dark curve (``j_ph = 0``) this is
negative in forward bias (current is injected into the diode), which is how the
app already plots the model dark curve. Measured dark data, however, is usually
reported as positive magnitudes. To keep imported points, the model dark curve,
and residuals on a single consistent convention, dark current is stored as
``-abs(current)``. Light current is stored as-is, since it legitimately crosses
zero at Voc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Minimum number of (V, J) points for a fit to be meaningful.
MIN_POINTS = 5

VOLTAGE_UNITS = ("V", "mV")
CURRENT_UNITS = ("A/cm2", "mA/cm2")

# Split a data row on any run of comma / semicolon / tab / space.
_COLUMN_SPLIT = re.compile(r"[,;\t ]+")


class DataImportError(ValueError):
    """Raised when pasted/uploaded J-V data cannot be parsed or validated.

    Subclasses ``ValueError`` so callers can catch either; the message is written
    to be shown directly to the user.
    """


@dataclass(frozen=True)
class ImportedDataset:
    """A validated, model-ready measured J-V dataset.

    The ``voltage`` / ``current`` arrays are normalised (base units, sorted by
    voltage ascending, dark sign applied) and are what the fitter consumes. The
    ``raw_*`` arrays preserve the original imported values (in the imported
    units, original order) so nothing the user supplied is lost.

    Attributes:
        voltage: voltage points in V, float64, sorted ascending.
        current: current density in A/cm^2 (model sign convention; see module docstring).
        label: user-facing name for the dataset.
        kind: "light" or "dark".
        raw_voltage: original voltage column, as imported (imported units, original order).
        raw_current: original current column, as imported.
        voltage_units: units the voltage column was imported in ("V" or "mV").
        current_units: units the current column was imported in ("A/cm2" or "mA/cm2").
        temp_k: measurement temperature in Kelvin (used as a fixed input when fitting).
    """

    voltage: np.ndarray
    current: np.ndarray
    label: str
    kind: Literal["light", "dark"]
    raw_voltage: np.ndarray
    raw_current: np.ndarray
    voltage_units: str
    current_units: str
    temp_k: float = 298.15


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse free-form two-column numeric text into (voltage, current) arrays.

    Accepts whitespace-, tab-, comma-, or semicolon-separated data. Blank lines,
    ``#`` comment lines, and non-numeric header lines (e.g. ``LightJV``, ``V``,
    ``A/cm^2``) are skipped. Every remaining line must contain exactly two numeric
    columns.

    Args:
        text: raw pasted or uploaded text.

    Returns:
        (voltage, current) as float64 arrays in the *imported* units and order.

    Raises:
        DataImportError: on empty input, a row without exactly two numeric
            columns, or no numeric rows at all.
    """
    if text is None or not text.strip():
        raise DataImportError("No data provided. Paste or upload two columns of numbers.")

    voltages: list[float] = []
    currents: list[float] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = [t for t in _COLUMN_SPLIT.split(line) if t]

        # Skip a header/label line where no token is numeric (e.g. "V  A/cm^2").
        if not any(_is_number(t) for t in tokens):
            continue

        if len(tokens) != 2 or not all(_is_number(t) for t in tokens):
            raise DataImportError(
                f"Line {lineno}: expected exactly two numeric columns, got "
                f"{raw_line.strip()!r}. Use two columns (voltage, current) "
                "separated by space, tab, comma, or semicolon."
            )

        voltages.append(float(tokens[0]))
        currents.append(float(tokens[1]))

    if not voltages:
        raise DataImportError(
            "No numeric data rows found. Provide two columns of numbers "
            "(voltage, current)."
        )

    return np.asarray(voltages, dtype=float), np.asarray(currents, dtype=float)


def to_base_units(
    voltage: np.ndarray,
    current: np.ndarray,
    voltage_units: str,
    current_units: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert voltage to V and current density to A/cm^2.

    Args:
        voltage: voltage column in ``voltage_units``.
        current: current column in ``current_units``.
        voltage_units: "V" or "mV".
        current_units: "A/cm2" or "mA/cm2".

    Returns:
        (voltage_V, current_A_per_cm2).

    Raises:
        DataImportError: on an unknown unit string.
    """
    if voltage_units not in VOLTAGE_UNITS:
        raise DataImportError(
            f"Unknown voltage units {voltage_units!r}; expected one of {VOLTAGE_UNITS}."
        )
    if current_units not in CURRENT_UNITS:
        raise DataImportError(
            f"Unknown current units {current_units!r}; expected one of {CURRENT_UNITS}."
        )

    v = voltage * (1e-3 if voltage_units == "mV" else 1.0)
    j = current * (1e-3 if current_units == "mA/cm2" else 1.0)
    return v, j


def normalize(
    voltage: np.ndarray,
    current: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by voltage ascending and apply the dark sign convention.

    Assumes ``voltage`` / ``current`` are already in base units and have passed
    validation (equal length, finite, strictly monotonic once sorted).

    Args:
        voltage: voltage points in V.
        current: current density in A/cm^2.
        kind: "light" (keep sign) or "dark" (store as ``-abs`` per module docstring).

    Returns:
        (voltage_sorted, current_sorted) as float64 arrays.
    """
    v = np.asarray(voltage, dtype=float)
    j = np.asarray(current, dtype=float)

    order = np.argsort(v)
    v = v[order]
    j = j[order]

    if kind == "dark":
        j = -np.abs(j)
    return v, j


def _validate(voltage: np.ndarray, current: np.ndarray) -> None:
    """Validate a base-unit (V, A/cm^2) two-column dataset, pre-sort.

    Raises:
        DataImportError: on mismatched lengths, non-finite values, too few
            points, or duplicate/non-monotonic voltages.
    """
    if voltage.shape[0] != current.shape[0]:
        raise DataImportError(
            f"Voltage and current columns have different lengths "
            f"({voltage.shape[0]} vs {current.shape[0]})."
        )
    if voltage.size < MIN_POINTS:
        raise DataImportError(
            f"Need at least {MIN_POINTS} data points to fit; got {voltage.size}."
        )
    if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(current)):
        raise DataImportError("Data contains non-finite values (NaN or infinity).")

    # Duplicate voltages make the curve ambiguous and break slope-based analysis.
    sorted_v = np.sort(voltage)
    if np.any(np.diff(sorted_v) == 0):
        raise DataImportError(
            "Duplicate voltage values found. Each voltage must be unique."
        )


def build_dataset(
    text: str,
    *,
    label: str,
    kind: str,
    voltage_units: str = "V",
    current_units: str = "A/cm2",
    temp_k: float = 298.15,
) -> ImportedDataset:
    """Parse, validate, and normalise raw two-column text into an ImportedDataset.

    Pipeline: ``parse_two_column`` -> unit check -> ``_validate`` (in base units)
    -> ``to_base_units`` -> ``normalize``. The original imported columns are
    preserved on the returned dataset.

    Args:
        text: raw pasted/uploaded text (two columns).
        label: user-facing dataset name.
        kind: "light" or "dark".
        voltage_units: "V" or "mV".
        current_units: "A/cm2" or "mA/cm2".
        temp_k: measurement temperature in Kelvin.

    Returns:
        A validated ``ImportedDataset``.

    Raises:
        DataImportError: on any parse/validation failure (message is user-facing).
    """
    if kind not in ("light", "dark"):
        raise DataImportError(f"Unknown dataset kind {kind!r}; expected 'light' or 'dark'.")

    raw_v, raw_j = parse_two_column(text)
    v_base, j_base = to_base_units(raw_v, raw_j, voltage_units, current_units)
    _validate(v_base, j_base)
    v_norm, j_norm = normalize(v_base, j_base, kind)

    return ImportedDataset(
        voltage=v_norm,
        current=j_norm,
        label=label,
        kind=kind,  # type: ignore[arg-type]
        raw_voltage=raw_v,
        raw_current=raw_j,
        voltage_units=voltage_units,
        current_units=current_units,
        temp_k=temp_k,
    )
