"""
Built-in example measured J-V datasets for the tandem page.

Unlike ``examples.py`` (digitised reference measurements stored as literal
constants), the tandem examples are generated at import time from the default
perovskite/silicon stack (``TANDEM_DEFAULT_INITIAL``) with a small amount of
fixed-seed noise — deterministic, so tests can assert that fitting recovers
the generating parameters. They live in their own module so ``examples.py``
stays free of any tandem dependency.

Both datasets are passed through the same validation/normalisation path as
user data (``_dataset_from_arrays`` in ``examples.py``), so the dark sign
convention and checks are applied identically.
"""

from __future__ import annotations

import numpy as np

from src.models.data_import import ImportedDataset
from src.models.examples import _dataset_from_arrays
from src.models.tandem import TandemParams, solve_tandem_current, tandem_voltage
from src.models.tandem_fitting import default_tandem_specs, unpack_tandem

# Fixed seed: the examples must be identical on every import.
_RNG = np.random.default_rng(42)


def _default_stack(model: str) -> TandemParams:
    specs = default_tandem_specs(model, free=set())
    return unpack_tandem(np.empty(0), specs, temp_k=298.15)


def _light_example() -> ImportedDataset:
    stack = _default_stack("light")
    voc = float(tandem_voltage(np.array([0.0]), stack)[0])
    voltage = np.linspace(0.0, voc, 61)
    current = solve_tandem_current(voltage, stack)
    # ~0.2% of Jsc of additive noise: visible in residuals, recoverable by a fit.
    current = current + _RNG.normal(0.0, 0.002 * current[0], size=current.shape)
    return _dataset_from_arrays(
        tuple(voltage), tuple(current), label="Example: Tandem Light JV", kind="light"
    )


def _dark_example() -> ImportedDataset:
    stack = _default_stack("dark")
    # Parametric in current: log-spaced forward-injection magnitudes give even
    # coverage of the exponential region across many decades.
    magnitude = np.logspace(-7, -1, 60)
    voltage = tandem_voltage(-magnitude, stack)
    # Multiplicative noise, as measured dark currents are log-distributed.
    magnitude = magnitude * 10.0 ** _RNG.normal(0.0, 0.01, size=magnitude.shape)
    return _dataset_from_arrays(
        tuple(voltage), tuple(magnitude), label="Example: Tandem Dark JV", kind="dark"
    )


TANDEM_LIGHT_JV_EXAMPLE = _light_example()
TANDEM_DARK_JV_EXAMPLE = _dark_example()

# Ordered mapping used to populate the tandem page's example-dataset selector.
TANDEM_EXAMPLE_DATASETS: dict[str, ImportedDataset] = {
    TANDEM_LIGHT_JV_EXAMPLE.label: TANDEM_LIGHT_JV_EXAMPLE,
    TANDEM_DARK_JV_EXAMPLE.label: TANDEM_DARK_JV_EXAMPLE,
}
