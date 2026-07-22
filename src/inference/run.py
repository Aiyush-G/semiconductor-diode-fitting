"""Run the No-U-Turn Sampler on a NumPyro model and package the draws.

A thin, reusable driver: configure NUTS (the self-tuning Hamiltonian sampler
NumPyro ships), run several chains, and return both the raw ``MCMC`` object and
an ArviZ ``InferenceData`` so every downstream diagnostic reads from
one standard container.  ``numpyro.enable_x64()`` is called first because the
reciprocity ``j_0`` spans decades and float32 rounding would coarsen its tail.

This is deliberately small.  The model is the subject; the sampler is
infrastructure, and NUTS needs no per-problem tuning — it adapts its step size
and trajectory length during warm-up, which is exactly why the book reaches for
it rather than a hand-tuned random walk.
"""

from __future__ import annotations

from typing import Any, Callable

import arviz as az
import numpy as np
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_median


def run_nuts(
    model: Callable[..., None],
    *,
    model_kwargs: dict[str, Any],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
    init_strategy=init_to_median,
) -> tuple[MCMC, az.InferenceData]:
    """Sample ``model`` with NUTS and return ``(mcmc, idata)``.

    Args:
        model: a NumPyro model function.
        model_kwargs: keyword arguments passed to ``model`` on every evaluation
            (the data and its configuration).
        num_warmup: adaptation (burn-in) iterations per chain — discarded.
        num_samples: retained draws per chain.
        num_chains: independent chains (>= 2 so R-hat is meaningful).
        seed: PRNG seed; equal seeds reproduce the run bitwise.
        target_accept: NUTS target acceptance probability.  Raising it (e.g. to
            0.95) shortens the step size and is the first remedy for divergences
            on a difficult geometry.
        init_strategy: where each chain starts.  Default ``init_to_median``: on a
            sloppy ridge the default ``init_to_uniform`` can seed a chain deep in
            the flat tail where its adapted step size collapses and it never moves
            (a frozen chain, R-hat ~ 3); starting at the prior median avoids that.

    Returns:
        ``(mcmc, idata)`` — the raw sampler and an ArviZ ``InferenceData``.
    """
    numpyro.enable_x64()
    kernel = NUTS(model, target_accept_prob=target_accept, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed), **model_kwargs)
    idata = az.from_numpyro(mcmc)
    return mcmc, idata


def posterior_summary(
    idata: az.InferenceData, var_names: tuple[str, ...] = ("j_0", "n")
) -> dict[str, dict[str, float]]:
    """Median, central 95% interval, ESS and R-hat per variable.

    A compact, plain-``dict`` summary (independent of ArviZ's printed table) so a
    test or a figure caption can assert on the numbers directly.
    """
    posterior = idata.posterior
    ess = az.ess(idata, var_names=list(var_names))
    rhat = az.rhat(idata, var_names=list(var_names))
    out: dict[str, dict[str, float]] = {}
    for name in var_names:
        draws = np.asarray(posterior[name].values).reshape(-1)
        out[name] = {
            "median": float(np.median(draws)),
            "lo95": float(np.percentile(draws, 2.5)),
            "hi95": float(np.percentile(draws, 97.5)),
            "ess": float(ess[name].values),
            "r_hat": float(rhat[name].values),
        }
    return out
