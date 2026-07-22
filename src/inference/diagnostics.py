"""Convergence diagnostics for NumPyro/ArviZ posterior samples.

No scalar diagnostic certifies a Markov chain.  This module therefore reports
the complementary checks needed before a diode parameter is interpreted:

* rank-normalised split R-hat for between-chain agreement;
* bulk and tail effective sample sizes for Monte-Carlo precision;
* divergent transitions and maximum-tree-depth hits for failed trajectories;
* energy Bayesian fraction of missing information (E-BFMI) for momentum mixing.

The defaults follow the conservative workflow thresholds used in modern HMC:
R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, no saturated trees,
and E-BFMI >= 0.3.  Passing them is necessary, not sufficient: model checking
and identifiability analysis remain separate obligations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import arviz as az
import numpy as np


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Thresholds used to turn diagnostic measurements into explicit issues."""

    max_r_hat: float = 1.01
    min_ess_bulk: float = 400.0
    min_ess_tail: float = 400.0
    max_divergences: int = 0
    max_treedepth_hits: int = 0
    min_ebfmi: float = 0.3
    max_tree_depth: int = 10


@dataclass(frozen=True)
class VariableDiagnostics:
    """Worst scalar diagnostic across every element of one model variable."""

    r_hat: float
    ess_bulk: float
    ess_tail: float


@dataclass(frozen=True)
class DiagnosticReport:
    """Machine-readable sampler health report.

    ``passed`` means that none of the configured thresholds was breached.  It
    does not mean that the likelihood is scientifically adequate, the forward
    model is correct, or the parameter is identifiable.
    """

    variables: dict[str, VariableDiagnostics]
    n_chains: int
    draws_per_chain: int
    n_divergences: int
    n_treedepth_hits: int
    ebfmi: tuple[float, ...]
    issues: tuple[str, ...]

    @property
    def total_draws(self) -> int:
        """Total retained draws across all chains."""
        return self.n_chains * self.draws_per_chain

    @property
    def passed(self) -> bool:
        """Whether every configured diagnostic threshold passed."""
        return not self.issues


def _finite_extreme(values: np.ndarray, *, maximum: bool) -> float:
    """Finite max/min of an ArviZ result, or NaN when none is available."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) if maximum else np.min(finite))


def _has_stat(idata: az.InferenceData, name: str) -> bool:
    return hasattr(idata, "sample_stats") and name in idata.sample_stats


def diagnose(
    idata: az.InferenceData,
    *,
    var_names: Iterable[str] | None = None,
    thresholds: DiagnosticThresholds | None = None,
) -> DiagnosticReport:
    """Measure convergence and Hamiltonian pathologies in posterior draws.

    Vector variables are reduced conservatively: the largest R-hat and the
    smallest bulk/tail ESS among their elements are reported.  Missing sampler
    statistics are not silently treated as healthy; each becomes an issue.

    Args:
        idata: ArviZ ``InferenceData`` containing ``posterior`` and preferably
            NumPyro ``sample_stats``.
        var_names: variables to check; defaults to every posterior data variable.
        thresholds: optional policy; defaults to :class:`DiagnosticThresholds`.

    Returns:
        A :class:`DiagnosticReport` suitable for tests, a UI, or serialisation.
    """
    if not hasattr(idata, "posterior"):
        raise ValueError("InferenceData must contain a posterior group")
    if "chain" not in idata.posterior.sizes or "draw" not in idata.posterior.sizes:
        raise ValueError("posterior must have chain and draw dimensions")

    policy = thresholds or DiagnosticThresholds()
    names = list(var_names) if var_names is not None else list(idata.posterior.data_vars)
    if not names:
        raise ValueError("at least one posterior variable is required")
    missing = [name for name in names if name not in idata.posterior]
    if missing:
        raise KeyError(f"posterior variables not found: {missing}")

    r_hat = az.rhat(idata, var_names=names, method="rank")
    ess_bulk = az.ess(idata, var_names=names, method="bulk")
    ess_tail = az.ess(idata, var_names=names, method="tail")

    variables: dict[str, VariableDiagnostics] = {}
    issues: list[str] = []
    for name in names:
        item = VariableDiagnostics(
            r_hat=_finite_extreme(r_hat[name].values, maximum=True),
            ess_bulk=_finite_extreme(ess_bulk[name].values, maximum=False),
            ess_tail=_finite_extreme(ess_tail[name].values, maximum=False),
        )
        variables[name] = item
        if not np.isfinite(item.r_hat):
            issues.append(f"{name}: R-hat unavailable (run at least two chains)")
        elif item.r_hat > policy.max_r_hat:
            issues.append(
                f"{name}: R-hat {item.r_hat:.3f} > {policy.max_r_hat:.3f}"
            )
        if not np.isfinite(item.ess_bulk) or item.ess_bulk < policy.min_ess_bulk:
            issues.append(
                f"{name}: bulk ESS {item.ess_bulk:.1f} < {policy.min_ess_bulk:.1f}"
            )
        if not np.isfinite(item.ess_tail) or item.ess_tail < policy.min_ess_tail:
            issues.append(
                f"{name}: tail ESS {item.ess_tail:.1f} < {policy.min_ess_tail:.1f}"
            )

    if _has_stat(idata, "diverging"):
        n_divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
        if n_divergences > policy.max_divergences:
            issues.append(
                f"{n_divergences} divergent transitions > {policy.max_divergences}"
            )
    else:
        n_divergences = 0
        issues.append("divergence statistic unavailable")

    if _has_stat(idata, "reached_max_tree_depth"):
        n_treedepth_hits = int(
            np.asarray(idata.sample_stats["reached_max_tree_depth"]).sum()
        )
        if n_treedepth_hits > policy.max_treedepth_hits:
            issues.append(
                f"{n_treedepth_hits} transitions reached maximum tree depth "
                f"> {policy.max_treedepth_hits} allowed"
            )
    elif _has_stat(idata, "tree_depth"):
        depths = np.asarray(idata.sample_stats["tree_depth"])
        n_treedepth_hits = int(np.count_nonzero(depths >= policy.max_tree_depth))
        if n_treedepth_hits > policy.max_treedepth_hits:
            issues.append(
                f"{n_treedepth_hits} transitions reached tree depth "
                f"{policy.max_tree_depth} > {policy.max_treedepth_hits} allowed"
            )
    else:
        n_treedepth_hits = 0
        issues.append("tree-depth statistic unavailable")

    if _has_stat(idata, "energy"):
        bfmi_result = az.bfmi(idata)
        # ArviZ <=0.23 returns an ndarray; ArviZ >=1.0 returns a DataTree.
        if hasattr(bfmi_result, "dataset"):
            bfmi_array = np.asarray(bfmi_result["energy"].values)
        else:
            bfmi_array = np.asarray(bfmi_result)
        ebfmi_values = tuple(float(x) for x in bfmi_array.reshape(-1))
        for chain, value in enumerate(ebfmi_values):
            if not np.isfinite(value) or value < policy.min_ebfmi:
                issues.append(
                    f"chain {chain}: E-BFMI {value:.3f} < {policy.min_ebfmi:.3f}"
                )
    else:
        ebfmi_values = ()
        issues.append("energy statistic unavailable")

    return DiagnosticReport(
        variables=variables,
        n_chains=int(idata.posterior.sizes["chain"]),
        draws_per_chain=int(idata.posterior.sizes["draw"]),
        n_divergences=n_divergences,
        n_treedepth_hits=n_treedepth_hits,
        ebfmi=ebfmi_values,
        issues=tuple(issues),
    )
