"""Priors as declared physics with provenance — the object that closes the ridge.

The single-diode degeneracy: profiling ``j_0`` with every
other parameter re-optimised gives a 95% interval that widens as the noise grows
until, at realistic noise, it is *open* — the data place no lower bound on
``j_0`` at all, because the fit escapes down the ``j_0``--``n`` valley into
ideality factors physics forbids. Least squares cannot fix this; the flatness is
real. The resolution is to add the information the experiment lacks. In Bayes'
theorem,

    posterior(theta | data)  ∝  likelihood(data | theta)  ×  prior(theta),

a flat likelihood direction is bounded *only* by the prior. This module builds
that prior — not as an arbitrary optimiser search range, but as a physical
statement carried with a **provenance** label that records how it was obtained
and therefore how hard it is and how the resulting number may be reported.

The one load-bearing abstraction is :class:`PhysicalBound`: a provenance-tagged
constraint that computes once and feeds *both* consumers of the book's doctrine —

* ``.to_paramspec_bounds()`` → the ``(lower, upper)`` the existing deterministic
  bounded-least-squares fitter already accepts via
  ``src.models.fitting.default_specs(bounds=...)`` — usable today; and
* ``.to_prior()`` → a NumPyro distribution for the Bayesian arm of Part II.

One physics computation, two consumers, so the deterministic box and the
Bayesian prior can never drift apart — the standard way a physics-constrained
fitter rots. The three *hardnesses* of a constraint map onto three forms:

* **hard, exact inequalities** (``j_0 >= j_0_rad`` from reciprocity;
  ``r_s >= r_s_floor`` from finger geometry) are made structural by
  *reparameterisation*, so violation is impossible and no wall is ever passed to
  the optimiser (external radiative efficiency ``ERE in (0, 1]`` for ``j_0``; a
  non-negative excess ``r_s = r_s_floor + delta`` for ``r_s``);
* **soft, uncertain limits** (the radiative floor's EQE band-tail systematic; the
  mechanism ideality assignment) become a *prior with a width*;
* the **deterministic box** is the shadow of the prior — a support-level
  truncation of the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.models.fitting import DEFAULT_BOUNDS, LOG_PARAMS, PARAM_NAMES
from src.models.single_diode import DiodeParams, solve_current, thermal_voltage

# NumPyro / JAX are the Part II substrate (verified installable in the sandbox).
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import Predictive


# ---------------------------------------------------------------------------
# Provenance: how a constraint was obtained, and therefore how hard it is
# ---------------------------------------------------------------------------


class Provenance(str, Enum):
    """Where a parameter constraint comes from — the label that governs the report.

    A number may be called a *measurement* only if its constraint is
    ``MEASURED``; a number whose posterior equals its ``RECIPROCITY`` /
    ``LITERATURE`` prior is *prior-informed*, not extracted, and must be labelled
    so. The value of physical bounds evaporates the moment a prior-dominated
    number is reported as a measurement — the field's chronic failure this
    project exists to fix.
    """

    MEASURED = "measured"        # lab measurement + algebra, no fit → two-sided, tight
    RECIPROCITY = "reciprocity"  # detailed balance from EQE → one-sided floor (soft)
    GEOMETRY = "geometry"        # device dimensions + sheet R → one-sided floor (hard)
    MECHANISM = "mechanism"      # recombination physics fixes n → discrete / envelope
    LITERATURE = "literature"    # representative value for the class → broad two-sided
    WEAK = "weak"                # genuinely unknown → broad, regularising only


# Which provenance classes express a one-sided *floor* (imposed by construction).
_FLOOR_PROVENANCES = frozenset({Provenance.RECIPROCITY, Provenance.GEOMETRY})


# ---------------------------------------------------------------------------
# External radiative efficiency and the non-radiative voltage deficit
# ---------------------------------------------------------------------------


def ere_from_j0(j_0: float, j_0_rad: float) -> float:
    """External radiative efficiency ``ERE = j_0_rad / j_0 in (0, 1]``.

    ``ERE = 1`` is the radiative (Shockley--Queisser) limit; smaller ``ERE`` means
    more non-radiative recombination. Since ``j_0 >= j_0_rad`` always (every
    non-radiative channel is additive and non-negative), ``ERE`` never exceeds 1.
    """
    _require_positive(j_0, "j_0")
    _require_positive(j_0_rad, "j_0_rad")
    return float(j_0_rad / j_0)


def j0_from_ere(ere: float, j_0_rad: float) -> float:
    """Invert :func:`ere_from_j0`: ``j_0 = j_0_rad / ERE`` with ``ERE in (0, 1]``.

    Sampling ``ERE in (0, 1]`` makes ``j_0 >= j_0_rad`` *impossible to violate* —
    the reciprocity floor imposed by construction rather than by a wall.
    """
    if not (0.0 < ere <= 1.0):
        raise ValueError(f"ERE must lie in (0, 1]; received {ere!r}.")
    _require_positive(j_0_rad, "j_0_rad")
    return float(j_0_rad / ere)


def delta_v_nr(j_0: float, j_0_rad: float, temp_k: float = 298.15) -> float:
    """Non-radiative open-circuit voltage deficit ``dV_nr = (kT/q) ln(j_0/j_0_rad)``.

    An ``O(50-200 mV)``, non-negative quantity (``dV_nr = 0`` exactly at the
    radiative floor), robust to the ``j_0``--``n`` pairing problem — a ``j_0``
    value is meaningless without its paired ``n``, but a voltage deficit is not —
    and directly comparable to absolute electroluminescence measurements. It is
    Bonilla's own perovskite modelling knob, so the extracted quantity and his
    forward model become the same variable.
    """
    _require_positive(j_0, "j_0")
    _require_positive(j_0_rad, "j_0_rad")
    return float(thermal_voltage(temp_k) * np.log(j_0 / j_0_rad))


def voc_radiative(j_sc: float, j_0_rad: float, temp_k: float = 298.15) -> float:
    """Radiative open-circuit voltage ``V_oc,rad = (kT/q) ln(j_sc/j_0_rad + 1)``.

    The ceiling no cell exceeds; the measured ``V_oc`` sits ``dV_nr`` below it.
    """
    _require_positive(j_sc, "j_sc")
    _require_positive(j_0_rad, "j_0_rad")
    return float(thermal_voltage(temp_k) * np.log(j_sc / j_0_rad + 1.0))


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive; received {value!r}.")


# ---------------------------------------------------------------------------
# The provenance-tagged constraint object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalBound:
    """A constraint on one parameter, tagged with how it was obtained.

    Computes once, feeds both arms: ``to_paramspec_bounds`` for the deterministic
    bounded-LS fitter, ``to_prior`` for the NumPyro Bayesian model. The
    ``provenance`` selects the prior *shape* (a floor reparameterises; a measured
    quantity is a tight two-sided density; a weak one is broad) and, downstream,
    governs how the fitted number may be reported.

    Attributes:
        name: one of ``PARAM_NAMES`` (``j_ph``, ``j_0``, ``n``, ``r_s``, ``r_sh``).
        provenance: the :class:`Provenance` class of the constraint.
        lower, upper: hard support edges in the parameter's natural units; ``None``
            means unbounded on that side (a one-sided floor has ``upper=None``).
        center: prior location / a good initial value (natural units).
        scale: prior width. For a log-scaled parameter (``j_0``, ``r_sh``) and for
            the reciprocity floor this is measured in *decades*; for a linear
            parameter it is in natural units.
        log: True if the parameter's natural coordinate is ``log10`` (``j_0``,
            ``r_sh``) — matches ``src.models.fitting.LOG_PARAMS``.
        note: free text recording the physics ("radiative floor, Si 1.12 eV,
            EQE edge +/-20 meV -> x2").
    """

    name: str
    provenance: Provenance
    lower: float | None = None
    upper: float | None = None
    center: float | None = None
    scale: float | None = None
    log: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.name not in PARAM_NAMES:
            raise ValueError(f"Unknown parameter {self.name!r}; expected one of {PARAM_NAMES}.")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance enum member.")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) exceeds upper ({self.upper}) for {self.name!r}.")

    # -- deterministic arm ---------------------------------------------------

    def to_paramspec_bounds(
        self, fallback: dict[str, tuple[float, float]] | None = None
    ) -> tuple[float, float]:
        """Return the ``(lower, upper)`` box the existing fitter accepts.

        Sides left ``None`` (an open floor's upper edge, say) fall back to the
        generic ``DEFAULT_BOUNDS`` so the deterministic fit stays feasible while
        the *physical* side of the constraint bites. This is the box as the
        shadow of the prior: the same physics, truncated to a support.
        """
        fb = DEFAULT_BOUNDS if fallback is None else fallback
        lo_fb, hi_fb = fb[self.name]
        lower = lo_fb if self.lower is None else float(self.lower)
        upper = hi_fb if self.upper is None else float(self.upper)
        if lower > upper:
            raise ValueError(
                f"resolved lower ({lower}) exceeds upper ({upper}) for {self.name!r}."
            )
        return (lower, upper)

    # -- Bayesian arm --------------------------------------------------------

    def to_prior(self) -> dist.Distribution:
        """Return a NumPyro distribution over the parameter in **natural units**.

        The distribution's *support* enforces the hard part of the constraint
        (a floor's support starts exactly at ``lower``, so no draw can violate it),
        and its *shape* carries the soft part (a wide density for an uncertain
        limit, a tight one for a measurement). Dispatch is on provenance:

        * ``RECIPROCITY`` — ``j_0 = j_0_rad * 10**excess`` with the excess (decades
          above the floor) ``HalfNormal(scale)``; support ``[j_0_rad, inf)``. This
          is the ERE reparameterisation: ``ERE in (0, 1]`` by construction.
        * ``GEOMETRY`` — ``r_s = r_s_floor + delta`` with ``delta ~ HalfNormal(scale)``;
          support ``[r_s_floor, inf)``.
        * ``MECHANISM`` — ``Uniform(lower, upper)`` over the ideality envelope
          ``[2/3, 2]`` (no mechanism produces a value outside it).
        * ``MEASURED`` — a tight ``TruncatedNormal(center, scale)`` on the measured
          value (the two-sided calibration band).
        * ``LITERATURE`` — a broad density: ``LogNormal`` for a log parameter,
          ``TruncatedNormal`` otherwise.
        * ``WEAK`` — a broad ``LogNormal`` (log parameter) or ``TruncatedNormal``,
          the non-committal regulariser (e.g. ``r_sh`` centred on the slope).
        """
        if self.provenance is Provenance.RECIPROCITY:
            return self._reciprocity_prior()
        if self.provenance is Provenance.GEOMETRY:
            return self._geometry_prior()
        if self.provenance is Provenance.MECHANISM:
            return self._envelope_prior()
        if self.provenance is Provenance.MEASURED:
            return self._measured_prior()
        # LITERATURE and WEAK share the broad-density path (log vs linear).
        return self._broad_prior()

    # -- prior builders ------------------------------------------------------

    def _reciprocity_prior(self) -> dist.Distribution:
        floor = self._require_lower("reciprocity floor")
        scale_decades = float(self.scale) if self.scale is not None else 1.5
        _require_positive(scale_decades, "scale (decades)")
        # Reparameterise by the non-radiative excess in decades above the floor,
        #   excess = log10(j_0 / j_0_rad) = dV_nr / (ln10 * kT/q) >= 0,
        # so j_0 = floor * 10**excess = exp(ln floor + excess * ln 10) >= floor and
        # ERE = 10**(-excess) in (0, 1] can never be violated. The excess is
        # *centred* on a typical deficit (from ``center``, else ~2 decades ≈ 120 mV)
        # rather than on the floor, so the prior is weakly-informative, not
        # optimistic; TruncatedNormal at ``low=0`` keeps the floor exact.
        if self.center is not None and self.center > floor:
            excess_center = float(np.log10(self.center / floor))
        else:
            excess_center = 2.0
        base = dist.TruncatedNormal(excess_center, scale_decades, low=0.0)
        transforms = [
            dist.transforms.AffineTransform(np.log(floor), np.log(10.0)),
            dist.transforms.ExpTransform(),
        ]
        return dist.TransformedDistribution(base, transforms)

    def _geometry_prior(self) -> dist.Distribution:
        floor = self._require_lower("geometry floor")
        scale = float(self.scale) if self.scale is not None else 0.3
        _require_positive(scale, "scale")
        # r_s = floor + delta, delta ~ HalfNormal(scale) >= 0.
        base = dist.HalfNormal(scale)
        return dist.TransformedDistribution(
            base, [dist.transforms.AffineTransform(floor, 1.0)]
        )

    def _envelope_prior(self) -> dist.Distribution:
        lo, hi = self.to_paramspec_bounds()
        if not hi > lo:
            raise ValueError(f"mechanism envelope needs lower < upper; got [{lo}, {hi}].")
        return dist.Uniform(lo, hi)

    def _measured_prior(self) -> dist.Distribution:
        center = self._require_center("measured value")
        scale = float(self.scale) if self.scale is not None else abs(center) * 0.02
        _require_positive(scale, "scale")
        lo, hi = self.to_paramspec_bounds()
        return dist.TruncatedNormal(center, scale, low=lo, high=hi)

    def _broad_prior(self) -> dist.Distribution:
        center = self._require_center("prior centre")
        if self.log:
            # Broad multiplicative prior: scale is in decades of log10.
            scale_decades = float(self.scale) if self.scale is not None else 1.0
            _require_positive(scale_decades, "scale (decades)")
            return dist.LogNormal(np.log(center), scale_decades * np.log(10.0))
        scale = float(self.scale) if self.scale is not None else abs(center) * 0.5
        _require_positive(scale, "scale")
        lo, hi = self.to_paramspec_bounds()
        return dist.TruncatedNormal(center, scale, low=lo, high=hi)

    def _require_lower(self, what: str) -> float:
        if self.lower is None:
            raise ValueError(f"{self.provenance.value} prior needs a {what} (lower).")
        _require_positive(self.lower, "lower")
        return float(self.lower)

    def _require_center(self, what: str) -> float:
        if self.center is None:
            raise ValueError(f"{self.provenance.value} prior needs a {what} (center).")
        return float(self.center)


# ---------------------------------------------------------------------------
# Ready-made bound maps: the physics prior, and the flat-search-range strawman
# ---------------------------------------------------------------------------


def example_physical_bounds(
    *,
    j_0_rad: float = 6.25e-17,
    r_s_floor: float = 0.10,
    j_ph_center: float = 0.036064,
    r_sh_center: float = 437.0,
    temp_k: float = 298.15,
    reciprocity_scale_decades: float = 1.5,
) -> dict[str, PhysicalBound]:
    """The physics prior for the repository's silicon example cell.

    Every edge is an inequality from device physics, quantified with the memo's
    real numbers (silicon 1.12 eV radiative floor at 298 K, step EQE; a
    representative TCE geometric ``r_s`` floor; the mechanism ideality envelope;
    the photocurrent from ``J_sc``; a weak ``r_sh``). Defaults are keyword-only so
    a caller with real EQE / sheet-resistance / spectrum passes device-computed
    values in the same shape.

    Args:
        j_0_rad: radiative saturation-current floor (A/cm^2). Default: silicon,
            1.12 eV, step EQE, 298 K (``6.25e-17``);  
        r_s_floor: geometric series-resistance floor (Ohm.cm^2). Default ``0.10``
            = 15 Ohm/sq, 2 mm pitch, front + rear.
        j_ph_center: photocurrent (A/cm^2); default the example ``J_sc`` (the
            Chan & Phang fallback when EQE is unavailable).
        r_sh_center: shunt-resistance prior centre (Ohm.cm^2); from the near-``J_sc``
            slope (here the fitted value stands in).
        temp_k: cell temperature (K).
        reciprocity_scale_decades: width of the ``j_0`` soft floor, in decades.
    """
    return {
        "j_ph": PhysicalBound(
            "j_ph", Provenance.MEASURED,
            lower=j_ph_center * 0.94, upper=j_ph_center * 1.06,
            center=j_ph_center, scale=j_ph_center * 0.02,
            note="optical generation q∫Φ·EQE (here J_sc fallback), +/-2% calibration",
        ),
        "j_0": PhysicalBound(
            "j_0", Provenance.RECIPROCITY,
            lower=j_0_rad, center=j_0_rad * 100.0,  # excess centred ~2 decades ≈ 120 mV
            scale=reciprocity_scale_decades, log=True,
            note="radiative floor q∫EQE·φ_BB (Si 1.12 eV, 298 K); ERE in (0,1]; soft x5",
        ),
        "n": PhysicalBound(
            "n", Provenance.MECHANISM,
            lower=2.0 / 3.0, upper=2.0, center=1.2,
            note="mechanism ladder: 2/3 Auger, 1 diffusion/radiative, 2 SRH",
        ),
        "r_s": PhysicalBound(
            "r_s", Provenance.GEOMETRY,
            lower=r_s_floor, center=r_s_floor + 0.2, scale=0.3,
            note="TCE geometric floor R_sheet·l^2/12 (x2 front+rear); r_s = floor + delta",
        ),
        "r_sh": PhysicalBound(
            "r_sh", Provenance.WEAK,
            center=r_sh_center, scale=1.0, log=True,
            note="near-J_sc slope centre; weak, one decade, ready to report one-sided",
        ),
    }


def uniform_reference_bounds(
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, PhysicalBound]:
    """The field's convention: a flat search range per parameter, no physics.

    Every parameter gets a ``WEAK`` uniform prior over ``DEFAULT_BOUNDS`` —
    log-uniform for the log parameters (``j_0``, ``r_sh``), linear-uniform
    otherwise. This is the strawman the prior-predictive check exposes: seventeen
    decades of ``j_0`` and an ideality up to 5 generate physically absurd JV
    curves. Use it only as the *before* picture.
    """
    box = DEFAULT_BOUNDS if bounds is None else bounds
    out: dict[str, PhysicalBound] = {}
    for name in PARAM_NAMES:
        lo, hi = box[name]
        out[name] = PhysicalBound(
            name, Provenance.WEAK, lower=lo, upper=hi,
            log=name in LOG_PARAMS, note="flat search range (no physics)",
        )
    return out


def _uniform_prior(bound: PhysicalBound) -> dist.Distribution:
    """Uniform prior over a bound's box (log-uniform if the parameter is log)."""
    lo, hi = bound.to_paramspec_bounds()
    if bound.log:
        base = dist.Uniform(np.log10(lo), np.log10(hi))
        return dist.TransformedDistribution(
            base,
            [dist.transforms.AffineTransform(0.0, np.log(10.0)), dist.transforms.ExpTransform()],
        )
    return dist.Uniform(lo, hi)


# ---------------------------------------------------------------------------
# Prior predictive: push the prior through the forward model
# ---------------------------------------------------------------------------


def _prior_for(bound: PhysicalBound) -> dist.Distribution:
    """A flat ``WEAK`` bound with an explicit box samples log/linear-uniform;
    every other bound uses its provenance-selected prior."""
    if bound.provenance is Provenance.WEAK and bound.center is None:
        return _uniform_prior(bound)
    return bound.to_prior()


def sample_prior_parameters(
    bounds_map: dict[str, PhysicalBound], n_draws: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Draw ``n_draws`` parameter sets from a bound map's priors, via NumPyro.

    Uses ``numpyro.infer.Predictive`` on a model that samples each of the five
    parameters from its prior — the same NumPyro
    will drive with a likelihood, exercised here with the prior alone. Returns a
    ``{name: array(n_draws)}`` dict in natural units.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1.")

    def model() -> None:
        for name in PARAM_NAMES:
            numpyro.sample(name, _prior_for(bounds_map[name]))

    predictive = Predictive(model, num_samples=int(n_draws))
    draws = predictive(random.PRNGKey(int(seed)))
    return {name: np.asarray(draws[name], dtype=float).reshape(-1) for name in PARAM_NAMES}


@dataclass(frozen=True)
class PriorPredictive:
    """Result of pushing a prior through the forward model.

    Attributes:
        samples: ``{name: array(n_draws)}`` parameter draws (natural units).
        voltage: the voltage grid the curves are evaluated on (V).
        current: ``(n_draws, n_points)`` model current densities (A/cm^2); rows that
            overflowed the forward model are all-NaN.
        j_sc, v_oc: per-draw short-circuit current (A/cm^2) and open-circuit
            voltage (V); NaN where the curve is unusable or never crosses zero.
        physical: boolean mask of draws with a finite, first-quadrant JV curve
            (positive J_sc, a V_oc inside the swept range).
    """

    samples: dict[str, np.ndarray]
    voltage: np.ndarray
    current: np.ndarray
    j_sc: np.ndarray
    v_oc: np.ndarray
    physical: np.ndarray


def prior_predictive_jv(
    bounds_map: dict[str, PhysicalBound],
    voltage: np.ndarray,
    *,
    n_draws: int = 400,
    seed: int = 0,
    temp_k: float = 298.15,
) -> PriorPredictive:
    """Sample the prior and evaluate the JV curve of every draw.

    The prior predictive answers the check every prior must pass *before* any data
    are seen: does it generate physically plausible measurements? A physics prior
    produces a tight band of first-quadrant curves with sane ``V_oc``; the flat
    search-range prior produces garbage — the visual proof that "physically
    plausible bounds" is a real constraint, not decoration.
    """
    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 1 or voltage.size < 2:
        raise ValueError("voltage must be a 1-D grid of at least two points.")

    samples = sample_prior_parameters(bounds_map, n_draws, seed)
    n = n_draws
    current = np.full((n, voltage.size), np.nan)
    j_sc = np.full(n, np.nan)
    v_oc = np.full(n, np.nan)
    physical = np.zeros(n, dtype=bool)

    for i in range(n):
        params = DiodeParams(
            j_ph=float(samples["j_ph"][i]), j_0=float(samples["j_0"][i]),
            n=float(samples["n"][i]), r_s=float(samples["r_s"][i]),
            r_sh=float(samples["r_sh"][i]), temp_k=temp_k,
        )
        try:
            j = solve_current(voltage, params)
        except Exception:
            continue
        if not np.all(np.isfinite(j)):
            continue
        current[i] = j
        j_sc[i] = float(np.interp(0.0, voltage, j))
        if np.any(j <= 0.0):
            v_oc[i] = float(np.interp(0.0, j[::-1], voltage[::-1]))
        # Physical = a real first-quadrant cell: positive Jsc, a Voc in-range.
        if j_sc[i] > 0 and np.isfinite(v_oc[i]) and voltage[0] < v_oc[i] <= voltage[-1]:
            physical[i] = True

    return PriorPredictive(
        samples=samples, voltage=voltage, current=current,
        j_sc=j_sc, v_oc=v_oc, physical=physical,
    )


# ---------------------------------------------------------------------------
# The prior bounds the ridge: a prior applied to a profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorWeightedProfile:
    """A profile likelihood multiplied by a prior

    Attributes:
        grid: the parameter grid inherited from the profile (natural units).
        likelihood: normalised profile likelihood ``exp(-delta_2nll / 2)`` (area 1
            over the finite grid), the shape the *data* alone imply.
        prior_density: the prior density on the same grid (natural units), zero
            outside the prior's support (below a floor).
        posterior: normalised product ``likelihood * prior`` — the (profile)
            posterior, bounded wherever the prior is.
        interval: ``(lower, upper)`` central ``level`` credible interval of
            ``posterior``.
        likelihood_open_below: True if the pure profile never rose past the Wilks
            threshold on the low side .
    """

    grid: np.ndarray
    likelihood: np.ndarray
    prior_density: np.ndarray
    posterior: np.ndarray
    interval: tuple[float, float]
    likelihood_open_below: bool


def prior_weighted_profile(
    profile_result,
    bound: PhysicalBound,
    *,
    level: float = 0.95,
    wilks_threshold: float = 3.8414588,
) -> PriorWeightedProfile:
    """Multiply a``ProfileResult`` by ``bound``'s prior and re-read the interval.

    This is Bayes on one axis: the profile likelihood ``exp(-delta_2nll/2)`` is the
    data's contribution along the profiled parameter, and multiplying by the prior
    density gives a (profile) posterior. Where the likelihood is flat — the ridge
    found unbounded — the posterior follows the *prior*, so a floor in
    the prior becomes a lower bound on the parameter that the data could not
    supply. The full marginal posterior (integrating, not profiling, the
    nuisances) awaits the MCMC (to be implemented) this is its one-dimensional,
    profile-based preview.

    Args:
        profile_result: a ``src.fitting.profile.ProfileResult`` (has ``grid`` and
            ``delta_2nll``).
        bound: the :class:`PhysicalBound` whose ``to_prior()`` density weights it.
        level: central credible mass for the reported interval.
        wilks_threshold: chi-square(1) cut used to decide if the *likelihood* alone
            was open below (default 3.8415, the 95% Wilks level).

    Returns:
        A :class:`PriorWeightedProfile`.
    """
    grid = np.asarray(profile_result.grid, dtype=float)
    delta = np.asarray(profile_result.delta_2nll, dtype=float)
    if grid.shape != delta.shape or grid.ndim != 1 or grid.size < 3:
        raise ValueError("profile_result.grid / delta_2nll must be matching 1-D arrays (>=3).")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1).")

    # Work in the parameter's fit coordinate: log10 for a multi-decade positive
    # parameter (j_0, r_sh), linear otherwise. This is the coordinate the repo
    # fits in and the one profiled, and it keeps a heavy upper tail from
    # dominating the central interval. ``coord`` is the integration variable;
    # ``jac`` turns a natural-units density into a coord-space density.
    coord = np.log10(grid) if bound.log else grid.copy()
    jac = grid * np.log(10.0) if bound.log else np.ones_like(grid)

    finite = np.isfinite(delta)
    like = np.zeros_like(grid)
    like[finite] = np.exp(-0.5 * delta[finite])  # coordinate-free profile likelihood

    prior = bound.to_prior()
    logp = np.asarray(prior.log_prob(jnp.asarray(grid)), dtype=float)
    prior_natural = np.where(np.isfinite(logp), np.exp(logp), 0.0)
    # NumPyro log_prob does not enforce support (a HalfNormal/TruncatedNormal will
    # happily evaluate below its floor), so mask to the bound's support explicitly:
    # below a reciprocity/geometry floor the density must be exactly zero.
    if bound.lower is not None:
        prior_natural = np.where(grid >= bound.lower, prior_natural, 0.0)
    if bound.upper is not None:
        prior_natural = np.where(grid <= bound.upper, prior_natural, 0.0)
    prior_coord = prior_natural * jac

    like_n = _normalise(like, coord)
    prior_n = _normalise(prior_coord, coord)
    post_n = _normalise(like * prior_coord, coord)

    lo_c, hi_c = _central_interval(coord, post_n, level)
    interval = (float(10.0 ** lo_c), float(10.0 ** hi_c)) if bound.log else (lo_c, hi_c)

    # Was the likelihood alone open below? Its minimum sits at the joint MLE; if no
    # grid point *below* that minimum rises past the Wilks cut, the data set no
    # lower bound
    imin = int(np.nanargmin(np.where(finite, delta, np.inf)))
    below = delta[:imin]
    likelihood_open_below = bool(imin == 0 or not np.any(below[np.isfinite(below)] >= wilks_threshold))

    return PriorWeightedProfile(
        grid=grid, likelihood=like_n, prior_density=prior_n,
        posterior=post_n, interval=interval,
        likelihood_open_below=likelihood_open_below,
    )


def _normalise(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Normalise a non-negative density to unit trapezoidal area over ``grid``."""
    area = float(np.trapezoid(density, grid))
    if area <= 0 or not np.isfinite(area):
        return np.zeros_like(density)
    return density / area


def _central_interval(grid: np.ndarray, density: np.ndarray, level: float) -> tuple[float, float]:
    """Central ``level`` interval from a normalised density on ``grid``."""
    cdf = _cumulative(density, grid)
    lo_q = 0.5 * (1.0 - level)
    hi_q = 1.0 - lo_q
    lower = float(np.interp(lo_q, cdf, grid))
    upper = float(np.interp(hi_q, cdf, grid))
    return (lower, upper)


def _cumulative(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Trapezoidal CDF of a density on ``grid``, normalised to end at 1."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    total = cdf[-1]
    if total <= 0 or not np.isfinite(total):
        return np.linspace(0.0, 1.0, grid.size)
    return cdf / total
