"""Differentiable single-diode forward model and the minimal NumPyro model.

The deterministic forward model (:func:`src.models.single_diode.solve_current`)
uses SciPy's ``lambertw`` and forms the Lambert-W argument ``a * exp(b)``
directly.  That is correct for a forward sweep but unusable for gradient-based
sampling: it is not JAX-traceable, and ``exp(b)`` *overflows* to ``inf`` for the
large ``b`` a wide-bandgap cell or an exploring sampler produces, turning a
gradient into ``nan`` and killing a No-U-Turn trajectory.

This module supplies the JAX twin the Bayesian layer needs:

* :func:`wright_omega` — the Wright omega function ``omega(L) = W(exp(L))``,
  computed by Halley iteration **in the log of the argument** ``L = ln a + b``,
  so ``exp(b)`` is never formed and the result is finite and differentiable for
  every finite ``L`` (the log-argument overflow guard).
* :func:`solve_current_jax` — the single-diode current, term for term the
  algebra of ``solve_current`` but written in ``jax.numpy`` and routed through
  :func:`wright_omega`.  It agrees with the SciPy solver to machine precision
  (see ``tests/inference/test_models_numpyro.py``).
* :func:`two_parameter_model` — the minimal model this book samples first:
  ``j_0`` and ``n`` free, drawn from the physics priors (the
  reciprocity floor keeps ``j_0`` above ``j_0,rad`` by construction), the other
  three parameters fixed at truth, and a Gaussian measurement likelihood at a
  known scale. 

Nothing in the existing ``src/`` tree is modified; this grows beside
``priors.py`` and reuses its ``PhysicalBound.to_prior`` machinery so the
deterministic and Bayesian arms constrain the same parameters identically.

Float64 is strongly recommended for the reciprocity ``j_0`` (which spans
decades); :func:`src.inference.run.run_nuts` calls ``numpyro.enable_x64()``
before sampling.  The forward model itself is precision-agnostic.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from src.inference.priors import PhysicalBound
from src.models.single_diode import DiodeParams, K_BOLTZMANN, Q_CHARGE

# Halley iterations for wright_omega.  The iteration is cubically convergent, so
# eight steps reach float64 machine precision from the asymptotic seed across the
# whole range of L the diode produces. 
_HALLEY_STEPS = 8

FIXABLE = ("j_ph", "r_s", "r_sh")


def thermal_voltage(temp_k: float) -> float:
    """Thermal voltage ``Vt = kT/q`` in volts (JAX-safe scalar arithmetic)."""
    return K_BOLTZMANN * temp_k / Q_CHARGE


def wright_omega(log_arg: jnp.ndarray) -> jnp.ndarray:
    """Wright omega ``omega(L) = W(exp(L))``, stable in ``L`` (no ``exp`` overflow).

    ``omega`` is the unique real root ``w > 0`` of ``ln w + w = L``; substituting
    ``w = W(exp(L))`` recovers the principal Lambert-W branch of a positive
    argument.  Working with ``L = ln(argument)`` instead of the argument itself is
    the overflow guard: ``a * exp(b)`` can be ``1e+300`` or ``inf``, but
    ``L = ln a + b`` stays finite, and ``omega(L) ~ L - ln L`` for large ``L``.

    Both branches of the initial guess are always finite (the unused branch is
    clamped) so the function — and its autodiff gradient — is defined for every
    finite ``L``:

    * ``L > 1``  : ``w0 = L - ln L``      (large-argument asymptotic)
    * ``L <= 1`` : ``w0 = exp(min(L, 0))`` (small-argument limit ``W(z) ~ z``)

    Args:
        log_arg: array (or scalar) of ``L`` values.

    Returns:
        ``omega(L)``, same shape as ``log_arg``.
    """
    ell = log_arg
    w = jnp.where(
        ell > 1.0,
        ell - jnp.log(jnp.maximum(ell, 1e-12)),
        jnp.exp(jnp.minimum(ell, 0.0)),
    )
    for _ in range(_HALLEY_STEPS):
        g = jnp.log(w) + w - ell          # g(w) = ln w + w - L, root at omega(L)
        g_prime = 1.0 / w + 1.0
        g_double = -1.0 / (w * w)
        # Halley step (cubic): w <- w - g / (g' - 0.5 g g'' / g').
        w = w - g / (g_prime - 0.5 * g * g_double / g_prime)
    return w


def solve_current_jax(
    voltage: jnp.ndarray,
    j_ph: float,
    j_0: float,
    n: float,
    r_s: float,
    r_sh: float,
    temp_k: float = 298.15,
) -> jnp.ndarray:
    """JAX single-diode current density, differentiable and overflow-safe.

    Identical algebra to :func:`src.models.single_diode.solve_current` for the
    ``r_s > 0`` branch (always true in a fit), but written in ``jax.numpy`` and
    routed through :func:`wright_omega` so it is traceable, differentiable, and
    finite for every parameter set a sampler can reach.

    Args:
        voltage: voltage points (V).
        j_ph, j_0, n, r_s, r_sh: single-diode parameters (area-normalised units);
            any may be a JAX tracer.  ``r_s`` must be strictly positive.
        temp_k: cell temperature (K).

    Returns:
        current density (A/cm^2), same shape as ``voltage``.
    """
    nvt = n * thermal_voltage(temp_k)
    denom = nvt * (r_s + r_sh)
    # log of the Lambert-W argument a*exp(b): ln a + b, never exp(b) itself.
    ln_a = jnp.log(r_s) + jnp.log(r_sh) + jnp.log(j_0) - jnp.log(denom)
    b = r_sh * (r_s * (j_ph + j_0) + voltage) / denom
    w = wright_omega(ln_a + b)
    return (r_sh * (j_ph + j_0) - voltage) / (r_s + r_sh) - (nvt / r_s) * w


def fixed_parameters(
    truth: DiodeParams, free: tuple[str, ...] = ("j_0", "n")
) -> dict[str, float]:
    """The ``{name: value}`` held fixed at truth when ``free`` parameters vary.

    The minimal model frees ``j_0`` and ``n`` (the sloppy ridge) and pins the well-identified three (``j_ph``, ``r_s``, ``r_sh``) at their
    true values, isolating the degeneracy the sampler must explore.
    """
    allvals = {
        "j_ph": truth.j_ph, "j_0": truth.j_0, "n": truth.n,
        "r_s": truth.r_s, "r_sh": truth.r_sh,
    }
    return {name: float(allvals[name]) for name in FIXABLE if name not in free}


def two_parameter_model(
    voltage: jnp.ndarray,
    current: jnp.ndarray,
    *,
    bounds: dict[str, PhysicalBound],
    fixed: dict[str, float],
    sigma: float,
    temp_k: float = 298.15,
) -> None:
    """The minimal single-diode NumPyro model: ``j_0`` and ``n`` free.

    ``j_0`` is drawn from reciprocity prior — the ERE
    reparameterisation, so every draw satisfies ``j_0 >= j_0,rad`` and the
    sampler can never enter the super-radiative region the data alone 
    failed to exclude — and ``n`` from its mechanism envelope ``Uniform(2/3, 2)``.
    The remaining parameters are fixed at ``fixed``.  The likelihood is a
    Gaussian at the known measurement scale ``sigma`` 

    Args:
        voltage: measured voltage points (V).
        current: measured current density (A/cm^2).
        bounds: a ``PhysicalBound`` map (e.g. ``example_physical_bounds()``); only
            the ``j_0`` and ``n`` entries are read.
        fixed: ``{j_ph, r_s, r_sh}`` held at truth (see :func:`fixed_parameters`).
        sigma: known Gaussian current-noise scale (A/cm^2).
        temp_k: cell temperature (K).
    """
    j_0 = numpyro.sample("j_0", bounds["j_0"].to_prior())
    n = numpyro.sample("n", bounds["n"].to_prior())
    mu = solve_current_jax(
        jnp.asarray(voltage), fixed["j_ph"], j_0, n,
        fixed["r_s"], fixed["r_sh"], temp_k,
    )
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=jnp.asarray(current))
