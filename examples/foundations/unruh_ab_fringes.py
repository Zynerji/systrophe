"""Unruh-modulated Aharonov-Bohm fringes: observer-dependent visibility.

Geometry: same source / two-slit / detector layout as
``double_slit_ab_demo.py``, with a van Stockum cylinder providing the
AB phase. Now the *observer* sits on a circular orbit at radius
r_observer and angular velocity Omega_obs; that observer's proper
acceleration is a = Omega^2 r_observer / sqrt(F), which gives an
Unruh temperature T_U = a / (2 pi). The thermal bath dephases the
interferometer mode by eta_thermal(T_U).

Total visibility:
    V_total = cos(pi * kappa / 2) * exp(-gamma_tau * 2 nbar(omega, T_U)).

Demonstrated:
  [1] inertial observer (no orbit / T_U = 0): full AB fringes.
  [2] near-horizon observer (large T_U): fringes wash out.
  [3] sweep of V_total over (kappa, T_U).
  [4] V_total along the observer orbit-radius axis.

Run:
    python examples/unruh_ab_fringes.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from systrophe.foundations.aharonov_bohm_ctc import aharonov_bohm_phase
from systrophe.foundations.unruh_modulated_ab import (
    observer_unruh_temperature,
    sweep_observer_radii,
    thermal_decoherence,
    total_visibility,
    unruh_ab_intensity,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def main() -> None:
    # --- geometry ------------------------------------------------------
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_loop = 2.5 * vs.R           # AB-flux enclosure radius
    k, d, D_arm = 60.0, 0.30, 5.0
    omega_mode = 1.0              # interferometer mode frequency
    gamma_tau = 0.5               # dimensionless coupling-time
    coupling = 0.0                # no intentional which-way coupling
    Omega_obs = 1.0               # observer orbital frequency
    phi_ab = aharonov_bohm_phase(vs, r_loop)

    print("van Stockum exterior:")
    print(f"  omega = {vs.omega}, R = {vs.R}, alpha = {vs.alpha:.4f}")
    print(f"  AB enclosure r_loop = {r_loop}, phi_AB = {phi_ab:.4f} rad")
    print()

    # --- observer Unruh temperatures along the orbit -------------------
    r_obs_grid = np.linspace(1.05, 1.85, 9)
    print("Observer  r_obs   T_U(r_obs)   V_total(kappa=0)")
    for r in r_obs_grid:
        T = observer_unruh_temperature(vs, float(r), Omega_obs)
        V = total_visibility(coupling, T, omega=omega_mode, gamma_tau=gamma_tau)
        print(f"          {r:5.3f}  {T:9.4f}    {V:6.4f}")
    print()

    # --- 1. inertial observer (T_U = 0) --------------------------------
    print("[1] inertial observer  (T_U = 0):")
    T_in = 0.0
    V_in = total_visibility(coupling, T_in,
                            omega=omega_mode, gamma_tau=gamma_tau)
    print(f"    eta_thermal = {thermal_decoherence(T_in):.4f},  "
          f"V_total = {V_in:.4f}")
    print()

    # --- 2. near-horizon observer --------------------------------------
    r_hot = 1.80
    T_hot = observer_unruh_temperature(vs, r_hot, Omega_obs)
    V_hot = total_visibility(coupling, T_hot,
                             omega=omega_mode, gamma_tau=gamma_tau)
    print(f"[2] near-horizon observer  (r_obs = {r_hot}):")
    print(f"    T_U = {T_hot:.4f},  V_total = {V_hot:.4f}  "
          f"({100*(1-V_hot/V_in):.1f}% fringe contrast loss vs inertial)")
    print()

    save_path = _make_plot(
        vs=vs, r_loop=r_loop, k=k, d=d, D_arm=D_arm,
        phi_ab=phi_ab, coupling=coupling,
        omega_mode=omega_mode, gamma_tau=gamma_tau, Omega_obs=Omega_obs,
    )
    if save_path is not None:
        print(f"[3] plot saved to {save_path}")
    else:
        print("[3] matplotlib unavailable; skipping plot")


def _make_plot(
    *, vs: VanStockumInterior, r_loop: float, k: float, d: float,
    D_arm: float, phi_ab: float, coupling: float,
    omega_mode: float, gamma_tau: float, Omega_obs: float,
) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    y_fine = np.linspace(-0.5, 0.5, 401)

    # (a) fringe contrast: inertial vs two accelerated observers ---------
    r_obs_samples = [1.10, 1.50, 1.80]
    Ts = [observer_unruh_temperature(vs, r, Omega_obs) for r in r_obs_samples]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    ax = axes[0]
    ax.plot(
        y_fine,
        unruh_ab_intensity(
            y_fine, k, d, D_arm, phi_ab=phi_ab,
            coupling=coupling, T_U=0.0,
            omega=omega_mode, gamma_tau=gamma_tau,
        ),
        lw=1.6, label="inertial  (T_U = 0)",
    )
    for r, T in zip(r_obs_samples, Ts):
        I = unruh_ab_intensity(
            y_fine, k, d, D_arm, phi_ab=phi_ab,
            coupling=coupling, T_U=T,
            omega=omega_mode, gamma_tau=gamma_tau,
        )
        ax.plot(y_fine, I, lw=1.4,
                label=f"r_obs={r:.2f}  T_U={T:5.2f}")
    ax.set_xlabel("detector y")
    ax.set_ylabel("|ψ|²")
    ax.set_title("(a) fringes: inertial vs accelerated")
    ax.legend(fontsize=7, loc="lower center")
    ax.set_ylim(-0.1, 4.3)

    # (b) V_total over (kappa, T_U) -------------------------------------
    ax = axes[1]
    kappa_grid = np.linspace(0.0, 1.0, 60)
    T_grid = np.linspace(0.0, 5.0, 60)
    K, T = np.meshgrid(kappa_grid, T_grid, indexing="xy")
    V = np.vectorize(
        lambda kk, tt: total_visibility(
            float(kk), float(tt), omega=omega_mode, gamma_tau=gamma_tau,
        )
    )(K, T)
    im = ax.imshow(
        V, origin="lower", aspect="auto",
        extent=[kappa_grid[0], kappa_grid[-1], T_grid[0], T_grid[-1]],
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("which-way coupling κ")
    ax.set_ylabel("Unruh temperature T_U")
    ax.set_title("(b) V_total(κ, T_U)")
    fig.colorbar(im, ax=ax, label="V_total")

    # (c) V_total along the orbit radius --------------------------------
    ax = axes[2]
    r_obs_grid = np.linspace(1.05, 1.85, 200)
    out = sweep_observer_radii(
        vs, r_obs_grid, coupling=0.0, angular_velocity=Omega_obs,
        omega=omega_mode, gamma_tau=gamma_tau,
    )
    ax.plot(out["r_observer"], out["T_U"], lw=1.6, label="T_U(r_obs)")
    ax2 = ax.twinx()
    ax2.plot(out["r_observer"], out["V_total"], lw=1.6, color="tab:orange",
             label="V_total")
    ax.set_xlabel("orbit radius r_obs")
    ax.set_ylabel("T_U")
    ax2.set_ylabel("V_total")
    ax.set_yscale("log")
    ax.set_title("(c) observer orbit: T_U(r) vs V_total(r)")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(-0.05, 1.05)

    # (d) heatmap I(y, T_U) at fixed kappa ------------------------------
    ax = axes[3]
    T_grid2 = np.linspace(0.0, 3.0, 240)
    I_mat = np.array([
        unruh_ab_intensity(
            y_fine, k, d, D_arm, phi_ab=phi_ab,
            coupling=coupling, T_U=float(t),
            omega=omega_mode, gamma_tau=gamma_tau,
        )
        for t in T_grid2
    ])
    im = ax.imshow(
        I_mat, origin="lower", aspect="auto",
        extent=[y_fine[0], y_fine[-1], T_grid2[0], T_grid2[-1]],
        cmap="magma", vmin=0.0, vmax=4.0,
    )
    ax.set_xlabel("detector y")
    ax.set_ylabel("T_U")
    ax.set_title("(d) fringe map vs Unruh T_U  (κ=0)")
    fig.colorbar(im, ax=ax, label="|ψ|²")

    fig.suptitle(
        "Unruh-modulated AB double-slit on a van Stockum cylinder",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


if __name__ == "__main__":
    main()
