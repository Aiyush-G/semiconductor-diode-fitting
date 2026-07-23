"""
Bounded nonlinear least-squares fitting of the single-diode model to measured
J-V data.

The forward model is the existing Lambert-W solver (``single_diode.solve_current``);
this module wraps it in a residual suitable for ``scipy.optimize.least_squares``.
It is UI-free so it can be tested and reused independently.

Design
------
- The user chooses which parameters are *free* (fitted) and which are *fixed*.
  Fixed parameters are copied verbatim into every trial ``DiodeParams`` and so
  remain exactly unchanged.
- Positive, multi-decade parameters (``j_0``, ``r_sh``) are fitted in log10 space;
  ``j_ph``, ``n`` and ``r_s`` are fitted linearly (``r_s`` can legitimately be ~0,
  where the forward model is continuous, so a log floor would only add a wall).
- Light data is fitted with linear current residuals (it crosses zero at Voc, so
  a log residual is unsafe). Dark data spans many decades of |J|, so it is fitted
  with log10 residuals by default. ``residual_space="auto"`` picks per model kind.
- Dark fits never expose a photocurrent term: ``j_ph`` is forced fixed at 0.
- A failed forward evaluation (overflow, NaN/inf) returns a large finite penalty
  vector of the correct length rather than crashing the optimizer.

Reported RMSE / R^2 / max-abs-residual are always in linear current units
(A/cm^2) so they are comparable across fits; ``rmse_log`` is additionally
reported when a log residual was used (a linear R^2 on dark data reads
deceptively close to 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from scipy.optimize import least_squares

from src.models.single_diode import DiodeParams, solve_current

if TYPE_CHECKING:
    from src.models.tandem import TandemParams

PARAM_NAMES = ("j_ph", "j_0", "n", "r_s", "r_sh")

# Which parameters are fitted in log10 space.
LOG_PARAMS = frozenset({"j_0", "r_sh"})

# Physically sensible default bounds (area-normalised: A/cm^2, Ohm.cm^2).
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "j_ph": (0.0, 0.1),
    "j_0": (1e-20, 1e-3),
    "n": (0.8, 5.0),
    "r_s": (0.0, 50.0),
    "r_sh": (1e1, 1e8),
}

# Default initial guesses used when the caller does not supply one.
DEFAULT_INITIAL: dict[str, float] = {
    "j_ph": 0.035,
    "j_0": 1e-12,
    "n": 1.0,
    "r_s": 0.5,
    "r_sh": 1000.0,
}

# Residual value returned for failed/non-finite forward evaluations.
PENALTY = 1e6
# |J| floor (A/cm^2) used when forming log residuals, so log(0) never occurs.
CURRENT_FLOOR = 1e-9

ResidualSpace = Literal["auto", "linear", "log"]


@dataclass(frozen=True)
class ParamSpec:
    """Fit configuration for a single model parameter.

    Attributes:
        name: one of ``PARAM_NAMES``.
        free: True if fitted, False if held fixed.
        value: initial guess when free; the fixed value when not free.
        lower, upper: bounds in the parameter's natural (linear) units.
        log: True if fitted in log10 space (bounds are transformed to match).
    """

    name: str
    free: bool
    value: float
    lower: float
    upper: float
    log: bool


@dataclass(frozen=True)
class FitResult:
    """Outcome of a fit.

    Headline error metrics (``rmse``, ``r_squared``, ``max_abs_residual``) are in
    linear current units (A/cm^2). ``rmse_log`` is populated only when a log
    residual space was used.

    Attributes:
        params: fitted parameter container (fixed entries unchanged) —
            ``DiodeParams`` for a single-diode fit, ``TandemParams`` for a
            tandem fit.
        free_names: names of the parameters that were fitted, in order.
        success: optimizer convergence flag (True for the all-fixed evaluation).
        message: optimizer status message.
        rmse: root-mean-square of (J_model - J_meas), A/cm^2.
        r_squared: coefficient of determination in linear current space.
        max_abs_residual: max |J_model - J_meas|, A/cm^2.
        rmse_log: RMSE of the log10|J| residuals (None unless log space was used).
        residual: linear residual vector J_model - J_meas, A/cm^2.
        model_current: J_model at the data voltages, A/cm^2.
        n_points: number of data points.
        cost: final least_squares cost (0.5 * sum of squared optimizer residuals).
        residual_space: the residual space actually used ("linear" or "log").
    """

    params: DiodeParams | TandemParams
    free_names: tuple[str, ...]
    success: bool
    message: str
    rmse: float
    r_squared: float
    max_abs_residual: float
    rmse_log: float | None
    residual: np.ndarray
    model_current: np.ndarray
    n_points: int
    cost: float
    residual_space: str


def default_specs(
    model: Literal["light", "dark"],
    free: set[str],
    initial: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, ParamSpec]:
    """Build a full ``{name: ParamSpec}`` map for a light or dark fit.

    For ``model="dark"`` the photocurrent is structurally excluded: ``j_ph`` is
    always injected as a fixed spec with value 0 and dropped from ``free`` if the
    caller mistakenly requested it. This guarantees a photocurrent term is never
    fitted to dark data.

    Args:
        model: "light" (all five parameters available) or "dark" (j_ph forced 0).
        free: set of parameter names to fit; the rest are held fixed.
        initial: optional per-parameter starting values / fixed values, overriding
            ``DEFAULT_INITIAL``.
        bounds: optional per-parameter (lower, upper) overrides of ``DEFAULT_BOUNDS``.

    Returns:
        Ordered dict (``PARAM_NAMES`` order) of ``ParamSpec``.
    """
    if model not in ("light", "dark"):
        raise ValueError(f"Unknown model {model!r}; expected 'light' or 'dark'.")

    initial = {**DEFAULT_INITIAL, **(initial or {})}
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    free = set(free)

    if model == "dark":
        # A photocurrent term is meaningless for dark data; never fit it.
        free.discard("j_ph")

    specs: dict[str, ParamSpec] = {}
    for name in PARAM_NAMES:
        if model == "dark" and name == "j_ph":
            specs[name] = ParamSpec(
                name="j_ph", free=False, value=0.0,
                lower=0.0, upper=0.0, log=False,
            )
            continue
        lower, upper = bounds[name]
        specs[name] = ParamSpec(
            name=name,
            free=name in free,
            value=float(initial[name]),
            lower=float(lower),
            upper=float(upper),
            log=name in LOG_PARAMS,
        )
    return specs


def _forward(value: float, log: bool) -> float:
    """Map a natural-unit value into fit space (log10 if ``log``)."""
    return float(np.log10(value)) if log else float(value)


def _inverse(theta: float, log: bool) -> float:
    """Map a fit-space value back to natural units (10**theta if ``log``)."""
    return float(10.0 ** theta) if log else float(theta)


def pack(
    specs: dict[str, ParamSpec],
    param_order: tuple[str, ...] = PARAM_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Assemble the free-parameter start vector and bounds in fit space.

    Args:
        specs: ``{name: ParamSpec}`` map.
        param_order: canonical parameter ordering; defaults to the single-diode
            ``PARAM_NAMES`` (the tandem fit passes its 10-name ordering).

    Returns:
        (theta0, lower, upper, free_names) where the arrays cover only free
        parameters, in ``param_order`` order, and log-space parameters have
        their value and bounds transformed with log10.
    """
    theta0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    free_names: list[str] = []
    for name in param_order:
        spec = specs.get(name)
        if spec is None or not spec.free:
            continue
        free_names.append(name)
        theta0.append(_forward(spec.value, spec.log))
        lower.append(_forward(spec.lower, spec.log))
        upper.append(_forward(spec.upper, spec.log))
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    # Clamp the start vector onto the feasible box: a user-supplied initial guess
    # (e.g. a slider value typed past a bound) must not make least_squares reject
    # x0 as infeasible. Only the seed is nudged, not the bounds.
    theta0_arr = np.clip(np.asarray(theta0, dtype=float), lower_arr, upper_arr)
    return (theta0_arr, lower_arr, upper_arr, tuple(free_names))


def unpack_values(
    theta: np.ndarray,
    specs: dict[str, ParamSpec],
    param_order: tuple[str, ...],
) -> dict[str, float]:
    """Rebuild the full ``{name: natural-unit value}`` map from a fit vector.

    Fixed parameters are taken verbatim from their spec value; free parameters
    are read from ``theta`` (in ``param_order`` order among the free set) and
    inverse-transformed. Model-agnostic — the caller decides which parameter
    container to build from the values.
    """
    values = {name: specs[name].value for name in param_order}
    free_names = [n for n in param_order if specs.get(n) is not None and specs[n].free]
    for name, t in zip(free_names, np.atleast_1d(theta)):
        values[name] = _inverse(float(t), specs[name].log)
    return values


def unpack(theta: np.ndarray, specs: dict[str, ParamSpec], temp_k: float) -> DiodeParams:
    """Rebuild ``DiodeParams`` from a fit-space vector plus the fixed specs."""
    values = unpack_values(theta, specs, PARAM_NAMES)
    return DiodeParams(
        j_ph=values["j_ph"], j_0=values["j_0"], n=values["n"],
        r_s=values["r_s"], r_sh=values["r_sh"], temp_k=temp_k,
    )


def resolve_residual_space(residual_space: ResidualSpace, kind: str) -> str:
    """Resolve "auto" to "log" for dark data and "linear" for light data."""
    if residual_space == "auto":
        return "log" if kind == "dark" else "linear"
    if residual_space not in ("linear", "log"):
        raise ValueError(
            f"Unknown residual_space {residual_space!r}; expected 'auto', 'linear', or 'log'."
        )
    return residual_space


def _linear_residual(model_current: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return model_current - measured


def _log_residual(model_current: np.ndarray, measured: np.ndarray) -> np.ndarray:
    model_mag = np.maximum(np.abs(model_current), CURRENT_FLOOR)
    meas_mag = np.maximum(np.abs(measured), CURRENT_FLOOR)
    return np.log10(model_mag) - np.log10(meas_mag)


def _make_residual(voltage, measured, specs, temp_k, space, penalty, predict=None):
    """Build the residual closure passed to ``least_squares``.

    Traps any forward-model failure or non-finite output and substitutes a large
    finite penalty, always returning a vector of length ``len(voltage)``.

    ``predict`` maps a fit-space vector theta to the model current at
    ``voltage``; when omitted it defaults to the single-diode forward model
    (``solve_current`` on ``unpack``-ed specs). The tandem fit supplies its own.
    """
    n_points = voltage.shape[0]
    residual_fn = _log_residual if space == "log" else _linear_residual
    if predict is None:
        def predict(theta: np.ndarray) -> np.ndarray:
            return solve_current(voltage, unpack(theta, specs, temp_k))

    def residuals(theta: np.ndarray) -> np.ndarray:
        try:
            model_current = predict(theta)
        except Exception:
            return np.full(n_points, penalty)
        r = residual_fn(model_current, measured)
        bad = ~np.isfinite(r)
        if bad.any():
            r = np.where(bad, penalty, r)
        return r

    return residuals


def _metrics(model_current: np.ndarray, measured: np.ndarray) -> tuple[float, float, float]:
    """Linear-space RMSE, R^2, and max absolute residual (A/cm^2)."""
    resid = model_current - measured
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    max_abs = float(np.max(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r_squared, max_abs


def fit_diode(
    voltage: np.ndarray,
    current: np.ndarray,
    temp_k: float,
    specs: dict[str, ParamSpec],
    *,
    kind: str = "light",
    residual_space: ResidualSpace = "auto",
    loss: str = "linear",
    penalty: float = PENALTY,
    max_nfev: int | None = None,
) -> FitResult:
    """Fit the free parameters in ``specs`` to measured (voltage, current) data.

    Args:
        voltage: measured voltage points (V).
        current: measured current density (A/cm^2), in the model sign convention.
        temp_k: fixed measurement temperature (K) — never fitted.
        specs: ``{name: ParamSpec}`` from ``default_specs`` (or hand-built).
        kind: "light" or "dark"; only used to resolve ``residual_space="auto"``.
        residual_space: "auto" | "linear" | "log".
        loss: ``least_squares`` loss ("linear", "soft_l1", "huber", ...).
        penalty: residual substituted for failed/non-finite evaluations.
        max_nfev: optional cap on function evaluations.

    Returns:
        A ``FitResult``. Never raises on optimizer failure — a failed fit is
        reported via ``success=False`` and the optimizer message.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    space = resolve_residual_space(residual_space, kind)

    return _fit_generic(
        voltage, current, specs,
        space=space, loss=loss, penalty=penalty, max_nfev=max_nfev,
        param_order=PARAM_NAMES,
        unpack_params=lambda theta: unpack(theta, specs, temp_k),
        predict=lambda theta: solve_current(voltage, unpack(theta, specs, temp_k)),
    )


def _fit_generic(
    voltage: np.ndarray,
    current: np.ndarray,
    specs: dict[str, ParamSpec],
    *,
    space: str,
    loss: str,
    penalty: float,
    max_nfev: int | None,
    param_order: tuple[str, ...],
    unpack_params: Callable[[np.ndarray], "DiodeParams | TandemParams"],
    predict: Callable[[np.ndarray], np.ndarray],
) -> FitResult:
    """Model-agnostic least-squares engine shared by the single-diode and
    tandem fits.

    ``unpack_params`` rebuilds the parameter container from a fit-space vector;
    ``predict`` maps a fit-space vector to the model current at ``voltage``.
    Everything else (packing, bounds, penalties, metrics) is common.
    """
    theta0, lower, upper, free_names = pack(specs, param_order)

    # All-fixed case: nothing to optimise, just evaluate and report.
    if theta0.size == 0:
        params = unpack_params(np.empty(0))
        model_current = predict(np.empty(0))
        return _build_result(
            params, free_names, True, "No free parameters; evaluated fixed model.",
            voltage, current, model_current, space, cost=0.0,
        )

    result = least_squares(
        _make_residual(voltage, current, None, None, space, penalty, predict=predict),
        theta0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss=loss,
        max_nfev=max_nfev,
    )

    params = unpack_params(result.x)
    # Recompute the model current cleanly (residual closure may have penalised).
    # Guard against a pathological final point so a fit never crashes the caller.
    success = bool(result.success)
    message = str(result.message)
    try:
        model_current = predict(result.x)
        if not np.all(np.isfinite(model_current)):
            raise ValueError("non-finite model current")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        model_current = np.full_like(voltage, np.nan)
        success = False
        message = f"{message} (final model evaluation failed: {exc})"
    return _build_result(
        params, free_names, success, message,
        voltage, current, model_current, space, cost=float(result.cost),
    )


def _build_result(
    params, free_names, success, message,
    voltage, current, model_current, space, cost,
) -> FitResult:
    rmse, r_squared, max_abs = _metrics(model_current, current)
    rmse_log = None
    if space == "log":
        log_resid = _log_residual(model_current, current)
        rmse_log = float(np.sqrt(np.mean(log_resid ** 2)))
    return FitResult(
        params=params,
        free_names=tuple(free_names),
        success=success,
        message=message,
        rmse=rmse,
        r_squared=r_squared,
        max_abs_residual=max_abs,
        rmse_log=rmse_log,
        residual=model_current - current,
        model_current=model_current,
        n_points=int(voltage.shape[0]),
        cost=cost,
        residual_space=space,
    )
