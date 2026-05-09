"""Generate figures for the Systrophe time-travel whitepaper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from systrophe import (
    SystrophePair,
    VanStockumInterior,
    find_single_cylinder_windows,
    harness_time_loop,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fig_F_L_oscillation():
    """Single-cylinder F and L oscillation in the supercritical exterior."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.geomspace(1.001, 60.0, 4001)
    F = vs.analytic_exterior_F(r)
    L = vs.analytic_exterior_L(r)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax[0].axhline(0, color="0.7", lw=0.7)
    ax[0].plot(np.log(r), F, color="C0", lw=1.0)
    ax[0].set_ylabel(r"$F(r) = -g_{tt}$")
    ax[0].set_title(r"Tipler sinusoid: $a = \omega R = 1$, $\alpha = \sqrt{3}$")
    ax[1].axhline(0, color="0.7", lw=0.7)
    ax[1].fill_between(np.log(r), L, 0, where=(L < 0), color="C3", alpha=0.25, label="CTC region ($L < 0$)")
    ax[1].plot(np.log(r), L, color="C3", lw=1.0)
    ax[1].set_xlabel(r"$\ln(r/R)$")
    ax[1].set_ylabel(r"$L(r) = g_{\phi\phi}$")
    ax[1].set_yscale("symlog", linthresh=1.0)
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "F_L_oscillation.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_omega_sector_diagram():
    """Plot the timelike-Omega sectors at the deepest point of band 1."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    w = find_single_cylinder_windows(vs, r_min=1.001, r_max=10.0)[0]
    F, K, L, r = w.F_at_min, w.K_at_min, w.L_min, w.r_min_L
    Om = np.linspace(-3, 3, 5001)
    s = F - 2 * K * Om - L * Om * Om
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="0.7", lw=0.7)
    ax.fill_between(Om, s, 0, where=(s > 0), color="C2", alpha=0.2, label="timelike orbit")
    ax.fill_between(Om, s, 0, where=(s < 0), color="C1", alpha=0.2, label="spacelike orbit")
    ax.plot(Om, s, color="k", lw=1.0)
    ax.axvline(w.omega_bounds_at_min[0], color="C1", linestyle="--", lw=0.8)
    ax.axvline(w.omega_bounds_at_min[1], color="C1", linestyle="--", lw=0.8)
    ax.set_xlabel(r"$\Omega = d\phi/dt$")
    ax.set_ylabel(r"$F - 2K\Omega - L\Omega^2$")
    ax.set_title(rf"Timelike-orbit sector at $r = {r:.3f}$ (CTC band 1, $a = 1$)")
    ax.legend(loc="upper center")
    fig.tight_layout()
    out = FIG_DIR / "omega_sector.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_offset_sweep():
    """CTC log-measure as a function of the SystrophePair phase offset."""
    cyl = VanStockumInterior(omega=1.5, R=1.0)
    pair = SystrophePair.from_cylinders(cyl, cyl, delta_offset=0.0)
    offsets = np.linspace(0, 2 * np.pi, 121)
    sweep = pair.offset_sweep(r_min=1.05, r_max=20.0, offsets=offsets)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(offsets / np.pi, sweep["log_measures"], color="C4", lw=1.2)
    ax.axvline(1.0, color="0.7", lw=0.7)
    ax.set_xlabel(r"phase offset $\delta_2 - \delta_1$ (units of $\pi$)")
    ax.set_ylabel(r"total $\sum \ln(r_{\rm out}/r_{\rm in})$ over CTC bands")
    ax.set_title(r"Off-set Tipler sinusoid: CTC log-measure vs phase offset")
    fig.tight_layout()
    out = FIG_DIR / "offset_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_pair_band_tuning():
    """Show CTC band positions of the pair as the offset varies."""
    cyl = VanStockumInterior(omega=1.5, R=1.0)
    pair = SystrophePair.from_cylinders(cyl, cyl, delta_offset=0.0)
    offsets = np.linspace(0, np.pi - 0.05, 24)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, off in enumerate(offsets):
        p = SystrophePair.from_cylinders(cyl, cyl, delta_offset=float(off))
        bands = p.ctc_bands(r_min=1.05, r_max=20.0)
        for r_in, r_out in bands:
            ax.plot([np.log(r_in), np.log(r_out)], [off, off], color="C3", lw=2.5, alpha=0.7)
    ax.set_xlabel(r"$\ln(r/R)$")
    ax.set_ylabel(r"phase offset $\delta_2 - \delta_1$")
    ax.set_title("Pair-tuned CTC bands")
    fig.tight_layout()
    out = FIG_DIR / "pair_band_tuning.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_time_travel_balance():
    """Plot proper time vs coordinate time for backward-time orbits."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    w = find_single_cylinder_windows(vs, r_min=1.001, r_max=10.0)[0]
    targets = np.linspace(-5.0, -0.05, 80)
    coord = []
    proper = []
    for t in targets:
        try:
            res = harness_time_loop(w, target_dt_per_rev=t, n_revolutions=1)
            coord.append(res["dt_per_revolution"])
            proper.append(res["dtau_per_revolution"])
        except ValueError:
            coord.append(np.nan)
            proper.append(np.nan)
    coord = np.array(coord)
    proper = np.array(proper)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(coord, proper, color="C5", lw=1.4)
    ax.axvline(0, color="0.7", lw=0.7)
    ax.set_xlabel(r"coord. time per rev.  $\Delta t = 2\pi/\Omega$")
    ax.set_ylabel(r"proper time per rev.  $\Delta\tau$")
    ax.set_title(r"Time-travel orbit (band 1, $r \approx 3.35$, $a = 1$)")
    fig.tight_layout()
    out = FIG_DIR / "time_travel_balance.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_off_axis_ctc_map():
    """2D CTC map for an off-axis Systrophe pair."""
    from systrophe.off_axis import OffAxisPair

    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=4.0)
    m = pair.ctc_map_2d(x_min=-3, x_max=8, y_min=-4, y_max=4, nx=181, ny=141)
    fig, ax = plt.subplots(figsize=(7, 4.0))
    # Show g_phiphi sign (red where CTC, blue otherwise)
    Z = m["g_phiphi_cyl1"]
    # Symlog scale for visualization
    vmax = float(np.nanpercentile(np.abs(Z), 95))
    im = ax.pcolormesh(
        m["x"], m["y"], Z,
        shading="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=r"$g_{\phi_1\phi_1}$")
    # Mark cylinder positions
    ax.plot(0, 0, marker="o", color="k", markersize=8, label="cyl 1 axis")
    ax.plot(4, 0, marker="o", color="k", markersize=8, fillstyle="none", label="cyl 2 axis")
    # CTC contour
    ax.contour(m["x"], m["y"], Z, levels=[0], colors="white", linewidths=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / R$")
    ax.set_ylabel(r"$y / R$")
    ax.set_title(
        rf"Off-axis pair, $a = 1$ both, separation $d = 4 R$"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "off_axis_ctc_map.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_lp_robust_demo():
    """Demonstrate machine-precision agreement of the robust solver at high a."""
    from systrophe.lp_robust import integrate_lp_robust

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for omega, color in [(0.7, "C0"), (1.0, "C1"), (1.5, "C2"), (2.5, "C3")]:
        sol = integrate_lp_robust(omega_dust=omega, R=1.0, r_max=20.0, n_samples=2001)
        u = np.log(sol.r / sol.R)
        ax.plot(u, sol.F, color=color, lw=1.0, label=rf"$a = {omega:.1f}$")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel(r"$\ln(r / R)$")
    ax.set_ylabel(r"$F(r)$")
    ax.set_title(r"Regime-dispatching robust solver: machine precision at any $a$")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    out = FIG_DIR / "lp_robust_demo.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    figs = [
        fig_F_L_oscillation(),
        fig_omega_sector_diagram(),
        fig_offset_sweep(),
        fig_pair_band_tuning(),
        fig_time_travel_balance(),
        fig_off_axis_ctc_map(),
        fig_lp_robust_demo(),
    ]
    for f in figs:
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
