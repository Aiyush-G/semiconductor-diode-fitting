"""Tests for convergence diagnostics and the funnel reparameterisation."""

from __future__ import annotations

import arviz as az
import numpy as np

from src.inference.diagnostics import (
    DiagnosticThresholds,
    diagnose,
)
from src.inference.funnel import noncentred_funnel
from src.inference.run import run_nuts


def _from_dict(posterior, sample_stats=None):
    """Construct sample data under both pre-1.0 and 1.x ArviZ APIs."""
    groups = {"posterior": posterior}
    if sample_stats is not None:
        groups["sample_stats"] = sample_stats
    try:
        return az.from_dict(groups)
    except TypeError:  # ArviZ <= 0.23
        return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def _healthy_idata(seed=42):
    rng = np.random.default_rng(seed)
    shape = (4, 1000)
    return _from_dict(
        {"theta": rng.normal(size=shape)},
        {
            "diverging": np.zeros(shape, dtype=bool),
            "tree_depth": np.full(shape, 5),
            "energy": rng.normal(size=shape),
        },
    )


def test_independent_chains_pass_the_default_policy():
    report = diagnose(_healthy_idata())

    assert report.passed
    assert report.total_draws == 4000
    assert report.variables["theta"].r_hat < 1.01
    assert report.variables["theta"].ess_bulk > 3000
    assert report.variables["theta"].ess_tail > 3000
    assert report.n_divergences == report.n_treedepth_hits == 0
    assert min(report.ebfmi) > 0.3


def test_rank_rhat_detects_a_chain_that_targets_the_wrong_region():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(4, 1000))
    draws[-1] += 2.0
    stats = {
        "diverging": np.zeros((4, 1000), dtype=bool),
        "tree_depth": np.ones((4, 1000), dtype=int),
        "energy": rng.normal(size=(4, 1000)),
    }

    report = diagnose(_from_dict({"theta": draws}, stats))

    assert not report.passed
    assert report.variables["theta"].r_hat > 1.1
    assert any("R-hat" in issue for issue in report.issues)


def test_autocorrelation_collapses_effective_sample_size():
    rng = np.random.default_rng(2)
    innovations = rng.normal(size=(4, 1000))
    draws = np.empty_like(innovations)
    draws[:, 0] = innovations[:, 0]
    for index in range(1, draws.shape[1]):
        draws[:, index] = 0.99 * draws[:, index - 1] + innovations[:, index]
    stats = {
        "diverging": np.zeros((4, 1000), dtype=bool),
        "tree_depth": np.ones((4, 1000), dtype=int),
        "energy": rng.normal(size=(4, 1000)),
    }

    report = diagnose(_from_dict({"theta": draws}, stats))

    assert report.variables["theta"].ess_bulk < 100
    assert any("bulk ESS" in issue for issue in report.issues)


def test_hamiltonian_pathologies_are_counted_not_hidden():
    rng = np.random.default_rng(3)
    shape = (4, 1000)
    divergent = np.zeros(shape, dtype=bool)
    divergent[1, 12] = True
    reached = np.zeros(shape, dtype=bool)
    reached[2, 34] = True
    # Slowly varying energy makes successive energy changes tiny relative to
    # the marginal energy variance: E-BFMI is deliberately pathological.
    energy = np.cumsum(rng.normal(scale=0.02, size=shape), axis=1)
    idata = _from_dict(
        {"theta": rng.normal(size=shape)},
        {
            "diverging": divergent,
            "reached_max_tree_depth": reached,
            "energy": energy,
        },
    )

    report = diagnose(idata)

    assert report.n_divergences == 1
    assert report.n_treedepth_hits == 1
    assert min(report.ebfmi) < 0.3
    assert any("divergent" in issue for issue in report.issues)
    assert any("tree depth" in issue for issue in report.issues)
    assert any("E-BFMI" in issue for issue in report.issues)


def test_vector_variables_reduce_to_the_worst_element():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(4, 1000, 2))
    draws[-1, :, 1] += 2.0
    shape = (4, 1000)
    idata = _from_dict(
        {"theta": draws},
        {
            "diverging": np.zeros(shape, dtype=bool),
            "tree_depth": np.ones(shape, dtype=int),
            "energy": rng.normal(size=shape),
        },
    )

    report = diagnose(idata)

    separate = az.rhat(idata, var_names=["theta"], method="rank")["theta"].values
    assert report.variables["theta"].r_hat == float(np.max(separate))
    assert report.variables["theta"].r_hat > 1.1


def test_noncentred_funnel_runs_cleanly():
    """A short real NUTS run verifies the geometry, driver, and diagnostics."""
    _, idata = run_nuts(
        noncentred_funnel,
        model_kwargs={"n_latent": 4},
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        seed=123,
        target_accept=0.8,
    )
    short_run_policy = DiagnosticThresholds(
        max_r_hat=1.05,
        min_ess_bulk=100,
        min_ess_tail=100,
        min_ebfmi=0.2,
    )

    report = diagnose(
        idata, var_names=("z", "x"), thresholds=short_run_policy
    )

    assert report.passed, report.issues
    assert report.n_divergences == report.n_treedepth_hits == 0
