# Tandem Model (2-Terminal) — the "Tandem" page

This page models and fits a **monolithic 2-terminal tandem solar cell**: two
single-diode equivalent circuits connected in series. It reuses the
single-diode machinery everywhere the physics allows (parameters, temperature
model, data import, fitting engine, plots) and adds only what is genuinely
new: the series/current-matching constraint.

Code map: physics in `src/models/tandem.py`, fitting in
`src/models/tandem_fitting.py`, example data in
`src/models/examples_tandem.py`, UI in `pages/02_tandem.py`.

---

## 1. Circuit picture

```
        ┌──────────────────────────┐
   ─────┤  Top sub-cell (wide Eg)  ├─────┐        J (shared)
        │  j_ph, j_0, n, R_s, R_sh │     │   ──────────────►
        └──────────────────────────┘     │
        ┌──────────────────────────┐     │
   ─────┤ Bottom sub-cell (Si)     ├─────┘
        │  j_ph, j_0, n, R_s, R_sh │
        └──────────────────────────┘
```

Each sub-cell is the standard single-diode circuit from the Single Diode page
(current source + diode + shunt resistance, in series with R_s), with all
quantities area-normalised per the PV Lighthouse convention: current densities
in A/cm² (displayed as mA/cm²), resistances in Ω·cm². Two sub-cells × five
parameters = **10 tandem parameters**, named `top_j_ph … top_r_sh` and
`bot_j_ph … bot_r_sh` in the fitting code.

## 2. Current matching: how the terminal curve is built

Because the sub-cells are in **series**, the same current density J flows
through both (2-terminal current-matching constraint), and the terminal
voltage is the sum of the sub-cell voltages at that shared current:

```
V_tandem(J) = V_top(J) + V_bottom(J)
```

The single-diode equation has an exact closed-form **inverse** — voltage as a
function of current — via the Lambert W function (Jain & Kapoor, the same
reference used for the forward solver in `single_diode.py`):

```
A    = j_ph + j_0 − J
V(J) = A·R_sh − J·R_s − n·Vt · W[ (j_0·R_sh)/(n·Vt) · exp(A·R_sh/(n·Vt)) ]
```

So no iterative circuit solving is needed: the page sweeps a shared current
grid, evaluates each sub-cell's V(J) exactly, sums them, and resamples the
parametric (V, J) points onto a voltage grid (`tandem.tandem_iv_curve`).
Because V(J) is strictly decreasing, the resampling is well-posed.

Two consequences worth internalising:

- **Voc is exactly additive.** At J = 0 each sub-cell sits at its own
  open-circuit voltage, so `Voc_tandem = Voc_top + Voc_bottom`. This is why
  the default perovskite/Si stack shows Voc ≈ 1.9 V.
- **Jsc sits slightly *above* the smaller sub-cell photocurrent, not below.**
  At V = 0 the current-limited sub-cell is pushed into reverse bias by its
  partner, and its shunt resistance passes the excess current
  (ΔJ ≈ |V_reverse|/R_sh). With no reverse-breakdown model (a stated
  limitation, same as the single-diode page), the shunt is the only such
  path, so `min(j_ph) < Jsc < max(j_ph)`, typically within a few percent of
  `min(j_ph)`. Turn on **"Show sub-cell JV curves"** to watch the limiting
  sub-cell dip below zero volts near Jsc.

## 3. Numerics: Lambert W in log space

The W argument's exponent `A·R_sh/(n·Vt)` is of order 10³–10⁷ for realistic
shunt resistances — float64's `exp` overflows at ~709, so the argument can
never be formed directly. `tandem.solve_voltage` therefore works in log
space:

- moderate exponents (≤ 500): scipy's `lambertw` on the explicit argument;
- large exponents: `W(e^x)` satisfies `w + ln(w) = x`, solved by Newton
  iteration on `ln(w)` directly (`_log_lambertw_exp_large`). Returning
  `ln(w)` also sidesteps the catastrophic cancellation between the huge,
  nearly-equal terms `A·R_sh` and `n·Vt·W` — the difference is formed
  analytically as `n·Vt·(ln w − ln c)`.

The round-trip `solve_current(solve_voltage(J)) = J` is tested to 1e-12 A/cm²
across shunt resistances up to 1e8 Ω·cm² (`tests/models/test_tandem.py`).

## 4. Temperature model (per sub-cell)

The existing PVsyst-style adjustment
(`temperature.adjust_params_for_temperature`) is applied **independently to
each sub-cell**, each with its own bandgap and Jsc coefficient (set in the
sub-cell expanders, never fitted):

- `J_ph(T) = J_ph,ref · [1 + α·(T − T_ref)]`
- `J_0(T) = J_0,ref · (T/T_ref)³ · exp[(E_g/k_B)·(1/T_ref − 1/T)]`

Defaults: top E_g = 1.68 eV, α = +0.02 %/K (perovskite); bottom
E_g = 1.121 eV, α = +0.05 %/K (silicon). Because the two photocurrents drift
at different rates, temperature can change **which sub-cell limits the tandem
current** — a genuinely tandem-specific behaviour you can see by sweeping the
temperature slider with the sub-cell overlay on.

## 5. Fitting the 10-parameter model

The fit reuses the single-diode engine (`fitting.py`): bounded
`scipy.optimize.least_squares` (TRF), per-parameter free/fixed `ParamSpec`s,
multi-decade parameters (`*_j_0`, `*_r_sh`) fitted in log10 space, linear
residuals for light data and log residuals for dark data by default, and the
same penalty handling for failed forward evaluations. Only three things are
tandem-specific (`tandem_fitting.py`):

- the 10 prefixed parameter names and their bounds (`*_j_0` extends down to
  1e-22 A/cm² for the wide-gap top cell);
- dark fits structurally exclude **both** photocurrents (fixed at 0);
- the forward model: `tandem.solve_tandem_current` builds one dense exact
  parametric (V, J) table per evaluation and interpolates V → J. Since V(J)
  is monotonic this is well-posed, and one vectorised table is far cheaper
  inside `least_squares` than per-point root finding.

### Degeneracy — read this before fitting

Freeing all 10 parameters against a single terminal curve is **hopeless**:
sub-cell voltages add at shared current, so e.g. trading `top_j_0` against
`bot_j_0` (or the two series resistances against each other) produces
near-identical terminal curves. An excellent residual does *not* mean the
parameters are physically meaningful. The intended workflow, as on the
single-diode page:

1. Fix everything you know (photocurrents from EQE, resistances from other
   measurements, bandgaps from the materials).
2. Free a small subset — the page defaults to `top_j_0, top_n, bot_j_0,
   bot_n` ticked.
3. Sanity-check the fitted values against physically plausible ranges, and
   refit from different starting values to see whether the solution is
   stable.

### Example datasets

The load dialog ships two deterministic synthetic datasets generated from the
default perovskite/Si stack with fixed-seed noise (light: ~0.2 % of Jsc
additive; dark: log-normal), so you can practise the workflow and verify the
fit recovers the generating parameters.

## 6. Units, defaults, references

Same conventions as the rest of the app: V, A/cm², Ω·cm², temperatures in °C
in the UI and K internally; dark data is stored as −|J| (see
`data_import.py`).

Default reference parameters (25 °C):

| Parameter | Top (perovskite) | Bottom (silicon) |
|---|---|---|
| J_ph (mA/cm²) | 20.0 | 19.5 |
| J_0 (A/cm²) | 1e-16 | 1e-13 |
| n | 1.5 | 1.0 |
| R_s (Ω·cm²) | 1.0 | 0.5 |
| R_sh (Ω·cm²) | 2000 | 5000 |
| E_g (eV) | 1.68 | 1.121 |
| α (1/K) | 0.0002 | 0.0005 |

References:

- Jain & Kapoor, *Exact analytical solutions of the parameters of real solar
  cells using Lambert W-function*, Solar Energy Materials and Solar Cells —
  forward and inverse closed forms.
- PVsyst one-diode model docs — temperature adjustment.
- TandEx (Oxford eMat Lab, https://github.com/Oxford-eMat-Lab/TandEx) —
  reference tandem equivalent-circuit model for later cross-validation
  (Phase D).
