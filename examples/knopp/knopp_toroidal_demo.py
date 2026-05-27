"""Demo: Toroidal Knopp Drive (counter-rotating Kerr binary backend).

Replaces the infinite Tipler cylinder with the finite-length toroidal
realization from Aguilera Katayama (April 2026): two near-extremal Kerr
black holes with maximally antiparallel spins, with the inter-horizon
toroidal domain providing the effective frame-dragging that the original
Bonnor Case III exterior supplied. Inside the toroidal CTC band the
Knopp composite exotic-matter budget collapses to zero (classically).

Run:
    python examples/knopp/knopp_toroidal_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systrophe.knopp.knopp_toroidal import (
    EffectiveToroidalKerrBinary,
    knopp_toroidal_budget,
    summarise_toroidal_budget,
)


def main() -> None:
    k_crit = EffectiveToroidalKerrBinary.critical_k()
    print(f"Toroidal Knopp Drive: counter-rotating maximal-spin Kerr binary")
    print(f"  band-existence threshold: k_crit = 2 M / d >= {k_crit:.4f}")
    print(f"  i.e., d <= {1.0/k_crit*2:.4f} M for the linear-LT band to exist.")
    print()

    # ---- subcritical: update.txt's quoted M=1, d=10 example ---------
    print("[1] subcritical binary (update.txt's quoted parameters, k = 0.2)")
    b_sub = knopp_toroidal_budget(M=1.0, d=10.0, rho_orbit=6.0)
    print(summarise_toroidal_budget(b_sub))
    print()

    # ---- critical: tight binary, in band ----------------------------
    print("[2] tight binary in band (k = 1, M = 1, d = 2, rho_orbit = 1.5)")
    b_in = knopp_toroidal_budget(M=1.0, d=2.0, rho_orbit=1.5,
                                  Q=100.0, epsilon_horn=0.01)
    print(summarise_toroidal_budget(b_in))
    print()

    # ---- sweep rho_orbit through the band -------------------------------
    print("[3] sweep rho_orbit through the band (M=1, d=2):")
    print(" rho     T_eff   gate    composite E_neg   final E_neg   inside")
    for rho in np.linspace(0.3, 4.0, 13):
        bb = knopp_toroidal_budget(M=1.0, d=2.0, rho_orbit=float(rho),
                                    Q=100.0, epsilon_horn=0.01)
        print(f" {rho:5.2f}  {bb.t_eff:6.4f}  {bb.tipler_gate_factor:6.4f}  "
              f"{bb.composite_E_neg:.4e}      {bb.final_E_neg:.4e}   "
              f"{bb.inside_ctc_band!s}")
    print()

    save_path = _make_plot()
    if save_path is not None:
        print(f"plot saved to {save_path}")


def _make_plot() -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) T_eff(rho) for a few binary configs (varying d/M ratio)
    ax = axes[0]
    rho_grid = np.linspace(0.05, 5.0, 401)
    for d in (2.0, 2.5, 4.0, 10.0):
        binary = EffectiveToroidalKerrBinary(M=1.0, d=d, chi=1.0)
        T = np.array([binary.t_eff(float(r), include_phi=False)
                      for r in rho_grid])
        k = 2.0 * binary.M / binary.d
        in_band = binary.has_toroidal_ctc_band(include_phi=False)
        marker = " (BAND)" if in_band else " (no band)"
        ax.plot(rho_grid, T, lw=1.6,
                label=f"d/M={d:.1f}, k={k:.2f}{marker}")
    ax.axhline(1.0, color="black", ls="--", lw=1.0, label="T_eff = 1")
    ax.set_xlabel("rho")
    ax.set_ylabel("T_eff (linear LT)")
    ax.set_title("(a) effective Tipler tilt")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0.0, 2.0)

    # (b) Final E_neg through the band
    ax = axes[1]
    rho_grid_b = np.linspace(0.2, 4.0, 80)
    E_classical = []
    E_final = []
    for rho in rho_grid_b:
        bb = knopp_toroidal_budget(M=1.0, d=2.0, rho_orbit=float(rho),
                                    Q=100.0, epsilon_horn=0.01)
        E_classical.append(bb.composite_E_neg)
        E_final.append(bb.final_E_neg)
    ax.plot(rho_grid_b, E_classical, lw=1.6, label="classical composite")
    ax.plot(rho_grid_b, E_final, lw=1.6, ls="--", label="+ back-reaction")
    binary = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    e_in, e_out = binary.ctc_band_edges(include_phi=False)
    if e_in is not None:
        ax.axvspan(e_in, e_out, color="green", alpha=0.15, label="CTC band")
    ax.set_xlabel("rho_orbit")
    ax.set_ylabel("|E_neg|")
    ax.set_title("(b) exotic-matter budget through the band")
    ax.legend(fontsize=8)

    # (c) k = 2M/d -> band-existence threshold
    ax = axes[2]
    k_grid = np.linspace(0.1, 2.0, 200)
    band_exists = []
    for k in k_grid:
        M = 1.0
        d = 2.0 * M / k
        binary = EffectiveToroidalKerrBinary(M=M, d=d, chi=1.0)
        band_exists.append(int(binary.has_toroidal_ctc_band(include_phi=False)))
    ax.fill_between(k_grid, 0, band_exists, color="green", alpha=0.4,
                    label="band exists")
    k_crit = EffectiveToroidalKerrBinary.critical_k()
    ax.axvline(k_crit, color="black", ls="--", lw=1.4,
               label=f"k_crit = {k_crit:.3f}")
    ax.set_xlabel("k = 2 M / d")
    ax.set_ylabel("band exists (0/1)")
    ax.set_title("(c) band-existence threshold")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Toroidal Knopp Drive (counter-rotating Kerr binary)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
