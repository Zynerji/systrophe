"""Demo: self-consistent NK back-reaction on the toroidal CTC band.

Replaces the dimensional `lambda / M^4` proxy in knopp_toroidal.py
with a true Newton-Kantorovich iteration on the band edges under a
Polyakov-Boulware vacuum source. Reports the back-reacted band
geometry as a function of the dimensionless coupling lambda, plus
the bisection-derived critical lambda at which the band closes.

Run:
    python examples/knopp/knopp_toroidal_backreaction_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_backreaction import (
    backreacted_band,
    backreaction_diagnostic,
    critical_lambda,
    q_threshold_from_balance,
    summarise_backreaction,
    t_kk_polyakov_boulware,
)


def main() -> None:
    print("=== Toroidal Knopp Drive: NK semiclassical back-reaction ===\n")

    binary = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    print(f"Working binary: M=1.0, d=2.0, chi=1.0 (k=1 > k_crit ~ 0.806)")
    print()

    print("[1] Polyakov-Boulware <T_kk>_B across the band:")
    for rho in np.linspace(0.5, 4.0, 8):
        Tkk = t_kk_polyakov_boulware(binary, float(rho))
        print(f"   rho = {rho:.2f}   <T_kk>_B = {Tkk:+.4e}")
    print()

    print("[2] sweep lambda -> back-reacted band edges")
    print("  lambda   rho_in_BR   rho_out_BR   width    closed?")
    for lam in [0.0, 1.0, 5.0, 10.0, 20.0, 21.5, 22.0, 50.0, 100.0]:
        b = backreacted_band(binary, lam=lam)
        if b.band_closed:
            print(f"  {lam:6.2f}    -          -          -        True")
        else:
            print(f"  {lam:6.2f}    {b.rho_in_BR:.4f}     "
                  f"{b.rho_out_BR:.4f}     {b.band_width_BR:.4f}   False")
    print()

    lam_c = critical_lambda(binary)
    print(f"[3] critical lambda (bisection):  lambda_crit ~= {lam_c:.4g}")
    print()

    Q_thr = q_threshold_from_balance(binary, E_krasnikov=1.0, omega_0=1.0)
    print(f"[4] Q-threshold from energy balance: Q_thr ~= {Q_thr:.4g}")
    print("    (above Q_thr the cavity absorbs the BR flux; "
          "below, band collapses)")
    print()

    print("[5] Full diagnostic at lambda = 1:")
    d = backreaction_diagnostic(binary, lam=1.0)
    print(summarise_backreaction(d))

    save_path = _make_plot()
    if save_path is not None:
        print()
        print(f"plot saved to {save_path}")


def _make_plot() -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    binary = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)

    # (a) Polyakov T_kk(rho)
    rho_grid = np.linspace(0.4, 5.0, 200)
    Tkk = np.array([t_kk_polyakov_boulware(binary, float(r)) for r in rho_grid])

    # (b) BR band edges vs lambda
    lam_grid = np.linspace(0.0, 50.0, 80)
    rho_in_BR = []
    rho_out_BR = []
    closed_lam = []
    for lam in lam_grid:
        b = backreacted_band(binary, lam=float(lam))
        if b.band_closed:
            closed_lam.append(lam)
            rho_in_BR.append(np.nan)
            rho_out_BR.append(np.nan)
        else:
            rho_in_BR.append(b.rho_in_BR)
            rho_out_BR.append(b.rho_out_BR)

    # (c) critical lambda
    lam_c = critical_lambda(binary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(rho_grid, Tkk, lw=1.6)
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is not None:
        ax.axvspan(edges[0], edges[1], color="green", alpha=0.15,
                   label="classical band")
    ax.set_xlabel("rho")
    ax.set_ylabel("<T_kk>_B (Polyakov-Boulware)")
    ax.set_title("(a) vacuum stress across the band")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(lam_grid, rho_in_BR, lw=1.6, label="rho_inner^BR")
    ax.plot(lam_grid, rho_out_BR, lw=1.6, label="rho_outer^BR")
    if edges[0] is not None:
        ax.axhline(edges[0], color="C0", ls="--", lw=1.0, alpha=0.6,
                   label="classical inner")
        ax.axhline(edges[1], color="C1", ls="--", lw=1.0, alpha=0.6,
                   label="classical outer")
    ax.axvline(lam_c, color="red", ls="--", lw=1.4,
               label=f"lambda_crit = {lam_c:.2f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("band edge rho")
    ax.set_title("(b) NK back-reacted band edges")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    band_widths = []
    for r_in, r_out in zip(rho_in_BR, rho_out_BR):
        if not (np.isnan(r_in) or np.isnan(r_out)):
            band_widths.append(r_out - r_in)
        else:
            band_widths.append(0.0)
    ax.plot(lam_grid, band_widths, lw=1.6)
    ax.axvline(lam_c, color="red", ls="--", lw=1.4,
               label=f"lambda_crit = {lam_c:.2f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("BR band width")
    ax.set_title("(c) band collapse under back-reaction")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Toroidal Knopp Drive: NK self-consistent semiclassical back-reaction"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
