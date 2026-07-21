from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tandem-book-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.fitting.uncertainty import estimate_fit_uncertainty
from src.models.examples import DARK_JV_EXAMPLE, LIGHT_JV_EXAMPLE
from src.models.fitting import default_specs, fit_diode, unpack
from src.models.single_diode import solve_current


OUT = Path(__file__).resolve().parent
FREE_LIGHT = {"j_ph", "j_0", "n", "r_s", "r_sh"}


def multistart_light(n_starts: int = 30):
    """Fit the measured light curve from deliberately dispersed starting points."""
    rng = np.random.default_rng(20260720)
    rows = []
    for _ in range(n_starts):
        initial = {
            "j_ph": rng.uniform(0.034, 0.040),
            "j_0": 10.0 ** rng.uniform(-15.0, -7.0),
            "n": rng.uniform(0.85, 2.2),
            "r_s": rng.uniform(0.05, 5.0),
            "r_sh": 10.0 ** rng.uniform(2.0, 5.0),
        }
        specs = default_specs("light", free=FREE_LIGHT, initial=initial)
        fit = fit_diode(
            LIGHT_JV_EXAMPLE.voltage,
            LIGHT_JV_EXAMPLE.current,
            LIGHT_JV_EXAMPLE.temp_k,
            specs,
            kind="light",
            max_nfev=5000,
        )
        rows.append((fit, specs, initial))
    successful = [row for row in rows if row[0].success]
    return sorted(successful, key=lambda row: row[0].rmse), rows


def fit_dark_in_both_spaces():
    """Fit the measured dark curve with linear- and log-current residuals."""
    results = {}
    for space in ("linear", "log"):
        specs = default_specs(
            "dark", free={"j_0", "n", "r_s", "r_sh"}
        )
        results[space] = fit_diode(
            DARK_JV_EXAMPLE.voltage,
            DARK_JV_EXAMPLE.current,
            DARK_JV_EXAMPLE.temp_k,
            specs,
            kind="dark",
            residual_space=space,
            max_nfev=10_000,
        )
    return results


def figure_residual_spaces(dark_fits):
    voltage = DARK_JV_EXAMPLE.voltage
    measured = np.abs(DARK_JV_EXAMPLE.current)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].semilogy(voltage, measured, "o", ms=3, color="black", label="measured")
    colours = {"linear": "#d95f02", "log": "#1b9e77"}
    for space, fit in dark_fits.items():
        axes[0].semilogy(
            voltage,
            np.abs(fit.model_current),
            lw=2,
            color=colours[space],
            label=f"{space}-residual fit",
        )
        log_residual = (
            np.log10(np.maximum(np.abs(fit.model_current), 1e-9))
            - np.log10(np.maximum(measured, 1e-9))
        )
        axes[1].plot(voltage, log_residual, lw=1.8, color=colours[space], label=space)
    axes[0].set_ylabel(r"$|J|$ (A cm$^{-2}$)")
    axes[0].set_title("One dark curve, two definitions of 'close'")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, which="both")
    axes[1].axhline(0.0, color="0.25", lw=1)
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel(r"$\log_{10}|J_{model}|-\log_{10}|J_{meas}|$")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = OUT / "fig1_residual_spaces.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def figure_correlation(uncertainty):
    labels = [r"$J_{ph}$", r"$\log_{10}J_0$", r"$n$", r"$R_s$", r"$\log_{10}R_{sh}$"]
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(uncertainty.correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    for row in range(5):
        for column in range(5):
            value = uncertainty.correlation[row, column]
            colour = "white" if abs(value) > 0.65 else "black"
            ax.text(column, row, f"{value:+.5f}", ha="center", va="center", color=colour)
    ax.set_title("Local parameter correlations: measured light JV")
    fig.colorbar(image, ax=ax, label="correlation")
    fig.tight_layout()
    path = OUT / "fig2_correlation_matrix.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def figure_weak_direction(best_fit, best_specs, uncertainty):
    theta = np.array([
        best_fit.params.j_ph,
        np.log10(best_fit.params.j_0),
        best_fit.params.n,
        best_fit.params.r_s,
        np.log10(best_fit.params.r_sh),
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(uncertainty.covariance_fit)
    direction = eigenvectors[:, -1] * np.sqrt(eigenvalues[-1])
    if direction[2] < 0:
        direction *= -1

    offsets = np.linspace(-2.0, 2.0, 41)
    j0, ideality, series, rmse = [], [], [], []
    curves = {}
    voltage = LIGHT_JV_EXAMPLE.voltage
    measured = LIGHT_JV_EXAMPLE.current
    for offset in offsets:
        params = unpack(theta + offset * direction, best_specs, LIGHT_JV_EXAMPLE.temp_k)
        model = solve_current(voltage, params)
        j0.append(params.j_0)
        ideality.append(params.n)
        series.append(params.r_s)
        rmse.append(np.sqrt(np.mean((model - measured) ** 2)))
        if np.isclose(offset, -1.0) or np.isclose(offset, 0.0) or np.isclose(offset, 1.0):
            curves[round(float(offset))] = model

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    scatter = axes[0].scatter(
        ideality, np.asarray(j0) / 1e-11, c=np.asarray(series), cmap="viridis", s=35
    )
    axes[0].set_xlabel("ideality factor n")
    axes[0].set_ylabel(r"$J_0$ ($10^{-11}$ A cm$^{-2}$)")
    axes[0].set_title("Weakest local direction")
    fig.colorbar(scatter, ax=axes[0], label=r"$R_s$ ($\Omega$ cm$^2$)")
    axes[0].grid(alpha=0.2)

    colours = {-1: "#7570b3", 0: "#1b9e77", 1: "#d95f02"}
    for offset in (-1, 0, 1):
        axes[1].plot(
            voltage,
            (curves[offset] - measured) * 1e6,
            color=colours[offset],
            lw=1.8,
            label=f"{offset:+d} local s.d.",
        )
    axes[1].axhline(0.0, color="0.25", lw=1)
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel(r"residual ($\mu$A cm$^{-2}$)")
    axes[1].set_title("Different parameters, nearly the same curve")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = OUT / "fig3_j0_n_rs_ridge.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, direction, offsets, np.asarray(rmse)


def main():
    successful, all_runs = multistart_light()
    if not successful:
        raise RuntimeError("No light-curve fit converged")
    best_fit, best_specs, _ = successful[0]
    uncertainty = estimate_fit_uncertainty(
        best_fit,
        LIGHT_JV_EXAMPLE.voltage,
        LIGHT_JV_EXAMPLE.current,
        LIGHT_JV_EXAMPLE.temp_k,
        best_specs,
        kind="light",
    )
    dark_fits = fit_dark_in_both_spaces()

    paths = [
        figure_residual_spaces(dark_fits),
        figure_correlation(uncertainty),
    ]
    ridge_path, weak_step, offsets, ridge_rmse = figure_weak_direction(
        best_fit, best_specs, uncertainty
    )
    paths.append(ridge_path)

    fits = [row[0] for row in successful]
    best = best_fit.params
    
    print(f"repo data: light N={LIGHT_JV_EXAMPLE.voltage.size}, dark N={DARK_JV_EXAMPLE.voltage.size}")
    print(f"multistart: {len(successful)}/{len(all_runs)} converged")
    print(
        "multistart RMSE range: "
        f"{min(f.rmse for f in fits):.9e} to {max(f.rmse for f in fits):.9e} A/cm^2"
    )
    print(
        "best light fit: "
        f"Jph={best.j_ph:.9g}, J0={best.j_0:.9g}, n={best.n:.9g}, "
        f"Rs={best.r_s:.9g}, Rsh={best.r_sh:.9g}, RMSE={best_fit.rmse:.9e} A/cm^2"
    )
    print(f"rank: {uncertainty.rank}/{len(uncertainty.free_names)}")
    print(f"Jacobian condition number: {uncertainty.condition_number:.6g}")
    print(f"residual sigma-hat: {np.sqrt(uncertainty.residual_variance):.9e} A/cm^2")
    for name, estimate, se_fit, se in zip(
        uncertainty.free_names,
        uncertainty.estimates,
        uncertainty.standard_errors_fit,
        uncertainty.standard_errors,
    ):
        print(f"  {name:4s} = {estimate:.9g}; SE_fit={se_fit:.6g}; SE_natural={se:.6g}")
    j0_index = uncertainty.free_names.index("j_0")
    n_index = uncertainty.free_names.index("n")
    rs_index = uncertainty.free_names.index("r_s")
    print(f"corr(log10 J0, n) = {uncertainty.correlation[j0_index, n_index]:.9f}")
    print(f"corr(log10 J0, Rs) = {uncertainty.correlation[j0_index, rs_index]:.9f}")
    print(f"corr(n, Rs) = {uncertainty.correlation[n_index, rs_index]:.9f}")
    print("one-local-s.d. weak step [Jph, log10J0, n, Rs, log10Rsh]:")
    print("  " + np.array2string(weak_step, precision=7))
    for offset in (-1.0, 0.0, 1.0):
        idx = int(np.argmin(np.abs(offsets - offset)))
        print(f"ridge RMSE at {offset:+.0f} s.d.: {ridge_rmse[idx]:.9e} A/cm^2")
    for space, fit in dark_fits.items():
        p = fit.params
        print(
            f"dark {space:6s}: success={fit.success}, J0={p.j_0:.9g}, n={p.n:.9g}, "
            f"Rs={p.r_s:.9g}, Rsh={p.r_sh:.9g}, linear_RMSE={fit.rmse:.9e}, "
            f"log_RMSE={fit.rmse_log}"
        )
    for path in paths:
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
