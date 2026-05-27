"""Demo: LQG + holographic-complexity probes of the Toroidal Knopp Drive.

The "next layer" from update.txt: apply LQG area/volume discretization
and the Complexity = Volume / Complexity = Action proxies to the
*finite* toroidal CTC band of EffectiveToroidalKerrBinary. Both probes
are ill-defined on the infinite Tipler cylinder but finite on the
binary-Kerr toroidal realisation.

Run:
    python examples/knopp/knopp_toroidal_quantum_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_quantum import (
    summarise_toroidal_quantum,
    toroidal_band_complexity,
    toroidal_band_lqg_discretization,
    toroidal_quantum_diagnostics,
)


def main() -> None:
    print("=== Toroidal Knopp Drive: quantum-gravity probes ===\n")

    # [1] supercritical k=1 binary, in band
    print("[1] working binary (M=1, d=2, k=1, IN BAND)")
    binary_a = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    print(summarise_toroidal_quantum(toroidal_quantum_diagnostics(binary_a)))
    print()

    # [2] subcritical
    print("[2] subcritical binary (M=1, d=10, k=0.2 — NO BAND)")
    binary_b = EffectiveToroidalKerrBinary(M=1.0, d=10.0, chi=1.0)
    print(summarise_toroidal_quantum(toroidal_quantum_diagnostics(binary_b)))
    print()

    # [3] sweep d/M -> show how the band collapses
    print("[3] sweep d/M -> band width and complexity")
    print(" d/M      k         band-width    V_band     C_V      C_A   "
          "  Lloyd dC/dt   traversal_time")
    for dM in np.linspace(1.0, 3.0, 11):
        binary = EffectiveToroidalKerrBinary(M=1.0, d=float(dM), chi=1.0)
        k = 2.0 / dM
        lqg = toroidal_band_lqg_discretization(binary)
        comp = toroidal_band_complexity(binary)
        tau = comp.cv_traversal_time
        tau_str = f"{tau:.3f}" if tau is not None else "  None"
        print(f" {dM:5.2f}   {k:6.4f}   {lqg.band_width_proper:8.4f}     "
              f"{comp.band_volume:8.3e}  {comp.cv_proxy:8.3e}  "
              f"{comp.ca_proxy:8.3e}   {comp.lloyd_growth_rate_max:.3e}   {tau_str}")
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

    # Sweep d/M over the regime where the band shrinks to nothing.
    dMs = np.linspace(1.0, 3.0, 41)
    band_widths, V_bands, CVs, CAs, j_inners, j_outers = [], [], [], [], [], []
    for dM in dMs:
        binary = EffectiveToroidalKerrBinary(M=1.0, d=float(dM), chi=1.0)
        d = toroidal_quantum_diagnostics(binary)
        if d.has_band:
            band_widths.append(d.lqg.band_width_proper)
            V_bands.append(d.complexity.band_volume)
            CVs.append(d.complexity.cv_proxy)
            CAs.append(d.complexity.ca_proxy)
            j_inners.append(d.lqg.j_inner)
            j_outers.append(d.lqg.j_outer)
        else:
            band_widths.append(0.0)
            V_bands.append(0.0)
            CVs.append(0.0)
            CAs.append(0.0)
            j_inners.append(0.0)
            j_outers.append(0.0)

    k_grid = 2.0 / dMs
    k_crit = EffectiveToroidalKerrBinary.critical_k()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) band width and V_band vs d/M
    ax = axes[0]
    ax.plot(dMs, band_widths, lw=1.6, label="band width Δρ")
    ax2 = ax.twinx()
    ax2.plot(dMs, V_bands, lw=1.6, color="tab:orange", label="V_band")
    ax.axvline(2.0 / k_crit, color="black", ls="--", lw=1.0,
               label=f"d/M = 1/k_crit ≈ {2.0/k_crit:.3f}")
    ax.set_xlabel("d / M")
    ax.set_ylabel("band width Δρ")
    ax2.set_ylabel("V_band")
    ax.set_title("(a) toroidal band geometry")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)

    # (b) LQG spins
    ax = axes[1]
    ax.plot(dMs, j_inners, lw=1.6, marker="o", ms=3, label="j_inner")
    ax.plot(dMs, j_outers, lw=1.6, marker="s", ms=3, label="j_outer")
    ax.axvline(2.0 / k_crit, color="black", ls="--", lw=1.0,
               label=f"band ends at d/M ≈ {2.0/k_crit:.3f}")
    ax.set_xlabel("d / M")
    ax.set_ylabel("LQG spin j (quantized)")
    ax.set_title("(b) LQG boundary spins")
    ax.legend(loc="upper right", fontsize=8)

    # (c) Complexity C_V, C_A
    ax = axes[2]
    ax.plot(dMs, CVs, lw=1.6, label="C_V (volume proxy)")
    ax.plot(dMs, CAs, lw=1.6, label="C_A (action proxy)")
    ax.axvline(2.0 / k_crit, color="black", ls="--", lw=1.0)
    ax.set_xlabel("d / M")
    ax.set_ylabel("complexity proxy")
    ax.set_title("(c) holographic complexity vs binary separation")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("symlog", linthresh=1e-3)

    fig.suptitle(
        "Toroidal Knopp Drive: LQG discretization + holographic complexity"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
