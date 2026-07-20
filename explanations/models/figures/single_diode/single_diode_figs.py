"""

Run from the repository root:


    fig1_parameter_fingerprints.png  - one-at-a-time sweeps of the five parameters
    fig2_local_ideality.png          - the m(V) diagnostic on synthetic dark curves
    fig3_example_data.png            - the repository's built-in light and dark data

"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.examples import DARK_JV_EXAMPLE, LIGHT_JV_EXAMPLE
from src.models.single_diode import (
    DiodeParams,
    iv_curve,
    key_metrics,
    local_ideality_factor,
    solve_current,
    thermal_voltage,
)

OUT = Path(__file__).resolve().parent
BASE = DiodeParams(j_ph=36e-3, j_0=1e-12, n=1.1, r_s=0.8, r_sh=2000.0)
mA = 1e3  # A/cm^2 -> mA/cm^2 for plotting


def replace(params: DiodeParams, **kw) -> DiodeParams:
    d = dict(j_ph=params.j_ph, j_0=params.j_0, n=params.n,
             r_s=params.r_s, r_sh=params.r_sh, temp_k=params.temp_k)
    d.update(kw)
    return DiodeParams(**d)


# --- Figure 1: parameter fingerprints ---------------------------------------

SWEEPS = [
    ("j_ph", [30e-3, 36e-3, 42e-3], "J_ph (mA/cm$^2$)", lambda v: f"{v*mA:.0f}"),
    ("j_0", [1e-14, 1e-12, 1e-10], "J$_0$ (A/cm$^2$)", lambda v: f"$10^{{{int(np.log10(v))}}}$"),
    ("n", [1.0, 1.3, 1.6], "n", lambda v: f"{v:.1f}"),
    ("r_s", [0.0, 1.5, 3.0], "R$_s$ ($\\Omega\\,$cm$^2$)", lambda v: f"{v:.1f}"),
    ("r_sh", [50.0, 300.0, 1e4], "R$_{sh}$ ($\\Omega\\,$cm$^2$)",
     lambda v: f"{v:.0f}" if v < 1e3 else f"$10^{{{int(np.log10(v))}}}$"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
for ax, (name, values, label, fmt) in zip(axes.flat, SWEEPS):
    for value, colour in zip(values, ("C0", "C1", "C2")):
        p = replace(BASE, **{name: value})
        v, j = iv_curve(p, n_points=400)
        ax.plot(v, j * mA, color=colour, label=fmt(value))
    ax.set_title(label)
    ax.set_xlabel("V (V)")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-2, 45)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8, title=None)
axes.flat[0].set_ylabel("J (mA/cm$^2$)")
axes.flat[3].set_ylabel("J (mA/cm$^2$)")
# Sixth panel: all five fingerprints summarised as annotated base curve.
ax = axes.flat[5]
v, j = iv_curve(BASE, n_points=400)
m = key_metrics(v, j)
ax.plot(v, j * mA, "k")
ax.plot([0], [m["jsc"] * mA], "C0o")
ax.plot([m["voc"]], [0], "C1o")
ax.plot([m["vmp"]], [m["jmp"] * mA], "C2o")
ax.annotate(f"J$_{{sc}}$ = {m['jsc']*mA:.1f}", (0.02, m["jsc"] * mA + 1), fontsize=8)
ax.annotate(f"V$_{{oc}}$ = {m['voc']:.3f} V", (m["voc"] - 0.21, 2), fontsize=8)
ax.annotate(f"MPP ({m['vmp']:.3f} V, {m['jmp']*mA:.1f})",
            (m["vmp"] - 0.33, m["jmp"] * mA - 4), fontsize=8)
ax.set_title(f"base device: FF = {m['fill_factor']:.3f}, $\\eta$ = {m['efficiency']:.1%}")
ax.set_xlabel("V (V)")
ax.set_xlim(0, 0.75)
ax.axhline(0, color="k", lw=0.5)
fig.suptitle("One-at-a-time parameter fingerprints on the light JV curve", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "fig1_parameter_fingerprints.png", dpi=150)
plt.close(fig)

# --- Figure 2: the m(V) diagnostic ------------------------------------------

vt = thermal_voltage(298.15)
voltage = np.linspace(0.0, 0.72, 800)
cases = {
    "ideal, n = 1.37": DiodeParams(0.0, 1e-12, 1.37, 0.0, 1e12),
    "shunted: R$_{sh}$ = 500": DiodeParams(0.0, 1e-12, 1.0, 0.0, 500.0),
    "series: R$_s$ = 2": DiodeParams(0.0, 1e-12, 1.0, 2.0, 1e12),
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for (label, p), colour in zip(cases.items(), ("C0", "C1", "C2")):
    j = solve_current(voltage, p)
    ax1.semilogy(voltage, np.abs(j) * mA, color=colour, label=label)
    ax2.plot(voltage, local_ideality_factor(voltage, j, p.temp_k), color=colour)
# The dark example dataset, through the same diagnostic.
ds = DARK_JV_EXAMPLE
m_data = local_ideality_factor(ds.voltage, ds.current, ds.temp_k)
ax1.semilogy(ds.voltage, np.abs(ds.current) * mA, "k.", ms=2.5,
             label="dark example data")
ax2.plot(ds.voltage, m_data, "k.", ms=2.5)
ax2.plot(voltage, voltage / vt, ":", color="C1", lw=1, label="m = V/V$_t$")
ax2.axhline(1.37, color="C0", ls=":", lw=1)
ax1.set_xlabel("V (V)"); ax1.set_ylabel("|J| (mA/cm$^2$)")
ax1.set_ylim(1e-7, 1e3); ax1.legend(fontsize=8)
ax1.set_title("dark curves, semilog")
ax2.set_xlabel("V (V)"); ax2.set_ylabel("local ideality m(V)")
ax2.set_ylim(0, 8); ax2.legend(fontsize=8)
ax2.set_title("the same curves as m(V)")
fig.tight_layout()
fig.savefig(OUT / "fig2_local_ideality.png", dpi=150)
plt.close(fig)

# --- Figure 3: the example datasets (tier R) --------------------------------

light, dark = LIGHT_JV_EXAMPLE, DARK_JV_EXAMPLE
lm = key_metrics(light.voltage, light.current)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(light.voltage, light.current * mA, "C0.-", ms=4, lw=0.8)
ax1.set_xlabel("V (V)"); ax1.set_ylabel("J (mA/cm$^2$)")
ax1.set_title(
    f"light example: J$_{{sc}}$ = {lm['jsc']*mA:.2f} mA/cm$^2$, "
    f"V$_{{oc}}$ = {lm['voc']*1e3:.1f} mV, FF = {lm['fill_factor']:.3f}"
)
ax2.semilogy(dark.voltage, np.abs(dark.current) * mA, "C3.-", ms=3, lw=0.5)
ax2.set_xlabel("V (V)"); ax2.set_ylabel("|J| (mA/cm$^2$)")
ax2.set_title(f"dark example: {dark.voltage.size} points, "
              f"{np.abs(dark.current).min()*1e6:.2f} $\\mu$A/cm$^2$ "
              f"to {np.abs(dark.current).max()*mA:.0f} mA/cm$^2$")
fig.suptitle("The repository's built-in measured data (tier R)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig3_example_data.png", dpi=150)
plt.close(fig)

# --- Numbers quoted in the chapter ------------------------------------------

def implicit_residual(v, j, p):
    nvt = p.n * thermal_voltage(p.temp_k)
    vj = v + j * p.r_s
    return p.j_ph - p.j_0 * (np.exp(vj / nvt) - 1.0) - vj / p.r_sh - j

print("=== numbers quoted in chapter 1 ===")
print(f"Vt(298.15 K) = {vt*1e3:.4f} mV")

v, j = iv_curve(BASE, n_points=400)
print(f"base device implicit-eq residual max = "
      f"{np.abs(implicit_residual(v, j, BASE)).max():.3e} A/cm^2")

m = key_metrics(v, j)
print(f"base device: Jsc = {m['jsc']*mA:.3f} mA/cm2, Voc = {m['voc']*1e3:.2f} mV, "
      f"FF = {m['fill_factor']:.4f}, eta = {m['efficiency']:.4f}")
print(f"  Jph - Jsc = {(BASE.j_ph - m['jsc'])*1e6:.3f} uA/cm2")

def voc_of(p):
    return brentq(lambda x: solve_current(np.array([x]), p)[0], 0.05, 2.0,
                  xtol=1e-14)

voc0 = voc_of(replace(BASE, r_s=0.0))
print("Voc vs Rs (brentq on the solver):")
for r_s in (0.0, 0.5, 2.0, 10.0):
    print(f"  Rs = {r_s:5.1f}: Voc = {voc_of(replace(BASE, r_s=r_s)):.15f} V"
          f"  (shift {voc_of(replace(BASE, r_s=r_s)) - voc0:+.2e} V)")

analytic = BASE.n * vt * np.log(BASE.j_ph / BASE.j_0 + 1.0)
big_rsh = voc_of(replace(BASE, r_sh=1e9))
print(f"analytic Voc (Rsh->inf) = {analytic*1e3:.4f} mV; "
      f"solver at Rsh=1e9: {big_rsh*1e3:.4f} mV; "
      f"diff = {abs(big_rsh-analytic)*1e9:.3f} nV")
print(f"Voc at Rsh = 2000: {voc0*1e3:.4f} mV "
      f"(shunt costs {(analytic-voc0)*1e3:.3f} mV)")

vv = np.linspace(0.0, 0.6, 100)
gap0 = solve_current(vv, replace(BASE, r_s=0.0)) - \
    solve_current(vv, replace(BASE, r_s=0.0, j_ph=0.0))
gap1 = solve_current(vv, replace(BASE, r_s=1.0)) - \
    solve_current(vv, replace(BASE, r_s=1.0, j_ph=0.0))
print(f"superposition gap - Jph: Rs=0: max {np.abs(gap0-BASE.j_ph).max():.2e}; "
      f"Rs=1: max {np.abs(gap1-BASE.j_ph).max():.2e} A/cm2")

print("Voc ladder (Rs=0, Rsh=1e9), step per decade of J0 "
      f"(n*Vt*ln10 = {1.1*vt*np.log(10)*1e3:.3f} mV):")
prev = None
for j_0 in (1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
    p = replace(BASE, j_0=j_0, r_s=0.0, r_sh=1e9)
    voc = voc_of(p)
    step = "" if prev is None else f"  step = {(voc-prev)*1e3:.3f} mV"
    print(f"  J0 = {j_0:.0e}: Voc = {voc*1e3:.3f} mV{step}")
    prev = voc

lm = key_metrics(light.voltage, light.current)
print(f"light example: {light.voltage.size} pts, Jsc = {lm['jsc']*mA:.3f} mA/cm2, "
      f"Voc = {lm['voc']*1e3:.2f} mV, FF = {lm['fill_factor']:.4f}, "
      f"pmax = {lm['pmax']*mA:.3f} mW/cm2, eta = {lm['efficiency']:.4f}")
window = (ds.voltage > 0.30) & (ds.voltage < 0.40)
print(f"dark example m(V) in 0.30-0.40 V: "
      f"median {np.nanmedian(m_data[window]):.3f}, "
      f"range {np.nanmin(m_data[window]):.3f}-{np.nanmax(m_data[window]):.3f}")

# Overflow boundary of the Lambert-W argument.
p = BASE
nvt = p.n * vt
v_over = 709.78 * nvt * (p.r_s + p.r_sh) / p.r_sh - p.r_s * (p.j_ph + p.j_0)
print(f"Lambert-W argument overflow (exp arg > 709.78) at V ~ {v_over:.1f} V "
      "for the base device")
print("figures written to", OUT)
