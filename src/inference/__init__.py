"""Bayesian inference layer for single-diode (and, later, tandem) fitting.

The likelihood (``src.fitting.noise``) says how
plausible a *measurement* is given the parameters; this package adds the other
factor Bayes' theorem needs — the **prior**, the statement of what we know about
the parameters *before* the data. Proved the single-diode likelihood
has a flat ridge the data cannot bound; the prior is what bounds it, and
:mod:`src.inference.priors` builds priors as *declared physics with a provenance
label* rather than as arbitrary optimiser search ranges.

Nothing in the existing ``src/`` tree is modified: this package grows beside
``src.models.fitting`` and reuses its ``ParamSpec``/``pack`` machinery and the
``src.fitting.noise`` likelihood, so the deterministic and Bayesian arms fit the
same model.
"""

from src.inference.priors import (
    PhysicalBound,
    Provenance,
    delta_v_nr,
    ere_from_j0,
    example_physical_bounds,
    j0_from_ere,
    prior_predictive_jv,
    prior_weighted_profile,
    sample_prior_parameters,
    uniform_reference_bounds,
    voc_radiative,
)

__all__ = [
    "PhysicalBound",
    "Provenance",
    "delta_v_nr",
    "ere_from_j0",
    "example_physical_bounds",
    "j0_from_ere",
    "prior_predictive_jv",
    "prior_weighted_profile",
    "sample_prior_parameters",
    "uniform_reference_bounds",
    "voc_radiative",
]
