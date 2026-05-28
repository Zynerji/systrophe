"""Demo: stability + GW signature of the Toroidal Knopp binary.

The framework's most speculative claim is that two near-extremal,
maximally counter-rotating Kerr black holes hold together long enough
to host the toroidal CTC band. This demo quantifies the answer
(classical-GR, leading-order spin-spin):

Headline numerical finding: the 'working configuration' (M=1, d=2M,
chi=1) is **dynamically catastrophic** -- the spin-spin coupling is
non-perturbative (|E_SS| / |E_orb| = 1) and the binary merges in
~0.012 orbits. The CTC band cannot persist at this parameter point
on dynamical grounds. The framework is FALSIFIED for the working
configuration unless the binary is held by something other than
gravity.

Wider binaries do survive many orbits but lie below the band-existence
threshold (k = 2M/d < k_crit ~ 0.806), so they have no CTC band at all.

Run:
    python examples/knopp/knopp_toroidal_stability_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_stability import (
    detector_classification,
    gw_frequency_in_hz,
    stability_report,
    summarise_stability,
    time_to_merger_in_seconds,
)


def main() -> None:
    print("=== Toroidal Knopp Drive: binary stability + GW signature ===\n")

    # [1] working configuration at several mass scales
    print("[1] working configuration M=1, d=2M, chi=1 at several mass scales")
    print(" M_solar     f_GW (Hz)    t_merger (s)   detector     n_orbits")
    binary = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    for M_solar in (1.0, 10.0, 1e3, 1e6, 1e9, 1e11):
        r = stability_report(binary, M_solar=M_solar)
        print(f"  {M_solar:.0e}      {r.gw_frequency_hz:.3e}    "
              f"{r.merger_time_seconds:.3e}    {r.detector_band:11s}  "
              f"{r.band_lifetime_vs_ctc_window:.3e}")
    print()

    # [2] band-vs-stability tradeoff: scan d/M
    print("[2] sweep d/M -> band existence + binary lifetime")
    print(" d/M    k       has_band   n_orbits     t_merger (s, M=10 M_sun)")
    for dM in (1.5, 2.0, 2.5, 5.0, 10.0, 50.0, 100.0):
        b = EffectiveToroidalKerrBinary(M=1.0, d=float(dM), chi=1.0)
        r = stability_report(b, M_solar=10.0)
        k = 2.0 / dM
        print(f"  {dM:5.1f}  {k:.4f}  {r.has_band!s:9s}  "
              f"{r.band_lifetime_vs_ctc_window:.3e}    "
              f"{r.merger_time_seconds:.3e}")
    print()

    # [3] full report for the canonical case
    print("[3] full report for the M=1, d=2M, M_solar=10 case (LIGO band):")
    r = stability_report(
        EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0), M_solar=10.0,
    )
    print(summarise_stability(r))
    print()

    # [4] honest verdict
    print("[4] verdict (classical-GR, leading-order quadrupole + spin-spin):")
    print("  - At the working configuration (M=1, d=2M, k=1):")
    print("      |E_SS| / |E_orb| = 1.00  (NON-PERTURBATIVE)")
    print("      n_orbits to merger = 0.012")
    print("    -> Binary merges in << 1 orbit. CTC band cannot persist.")
    print("  - At wider, sub-critical binaries (k < k_crit ~ 0.806):")
    print("      The binary survives many orbits, but NO TOROIDAL CTC BAND.")
    print("  -> The toroidal Knopp Drive is FALSIFIED in its classical-GR")
    print("     reading. Any rescue requires non-gravitational binding (a")
    print("     yet-unmodelled force/structure) or a beyond-leading-order")
    print("     GR effect that prevents inspiral.")
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

    dMs = np.linspace(1.0, 100.0, 200)
    n_orbits = []
    has_band = []
    f_hz_10 = []
    for dM in dMs:
        b = EffectiveToroidalKerrBinary(M=1.0, d=float(dM), chi=1.0)
        r = stability_report(b, M_solar=10.0)
        n_orbits.append(r.band_lifetime_vs_ctc_window)
        has_band.append(int(r.has_band))
        f_hz_10.append(r.gw_frequency_hz)

    # critical d/M for band existence
    from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary as B
    k_crit = B.critical_k()
    d_crit = 2.0 / k_crit

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) n_orbits vs d/M
    ax = axes[0]
    ax.loglog(dMs, n_orbits, lw=1.6, label="n_orbits to merger")
    ax.axhline(1.0, color="black", ls=":", lw=1.0, label="1 orbit")
    ax.axvline(d_crit, color="green", ls="--", lw=1.4,
               label=f"d/M = 1/k_crit ~ {d_crit:.3f}")
    ax.set_xlabel("d / M")
    ax.set_ylabel("n_orbits to merger")
    ax.set_title("(a) binary lifetime")
    ax.legend(fontsize=8)

    # (b) GW frequency vs d/M, at M_solar = 10
    ax = axes[1]
    ax.loglog(dMs, f_hz_10, lw=1.6, label="M = 10 M_sun")
    ax.axhspan(10, 5000, color="C2", alpha=0.15, label="LIGO band")
    ax.axhspan(1e-4, 1e-1, color="C1", alpha=0.15, label="LISA band")
    ax.axvline(d_crit, color="green", ls="--", lw=1.4)
    ax.set_xlabel("d / M")
    ax.set_ylabel("f_GW (Hz)")
    ax.set_title("(b) GW signature")
    ax.legend(fontsize=8, loc="upper right")

    # (c) viability map: band-existence x n_orbits-above-1
    ax = axes[2]
    has_band_arr = np.array(has_band)
    n_orbits_arr = np.array(n_orbits)
    viable = (has_band_arr == 1) & (n_orbits_arr > 1.0)
    ax.fill_between(dMs, 0.0, has_band_arr, color="green", alpha=0.25,
                    label="has CTC band")
    ax.fill_between(dMs, 0.0, (n_orbits_arr > 1.0).astype(float),
                    color="blue", alpha=0.25, label="n_orbits > 1")
    if viable.any():
        ax.fill_between(dMs, 0.0, viable.astype(float),
                        color="red", alpha=0.35, label="VIABLE (both)")
    ax.axvline(d_crit, color="black", ls="--", lw=1.4)
    ax.set_xlabel("d / M")
    ax.set_ylabel("indicator")
    ax.set_title("(c) viability map")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Toroidal Knopp binary: stability + GW signature"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
