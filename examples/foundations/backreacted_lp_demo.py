"""Demo: toy iterative semiclassical back-reaction on supercritical LP.

Sweeps ell^2 across several decades and reports, for each, how the
chronology horizon r_H shifts under the iterative Polyakov-driven
correction. The honest message: this is a 2D-Polyakov toy that captures
the *direction* and *threshold* of back-reaction; it is not the full
4D self-consistent Einstein-quantum equation.

Run:
    python examples/foundations/backreacted_lp_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systrophe.ctc.backreacted_lp import (
    classical_horizon,
    horizon_shift,
    iterate_backreaction,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def main() -> None:
    vs = VanStockumInterior(omega=1.0, R=1.0)
    print(f"Supercritical LP exterior (omega=R=1, alpha = sqrt(3))")
    print(f"  classical r_H = {classical_horizon(vs):.4f}")
    print()

    print(" ell^2      verdict      iters   r_H_BR    shift")
    runs = {}
    for ell2 in [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        res = iterate_backreaction(
            vs, ell2=float(ell2), n_iter=50, damping=0.2,
        )
        s = horizon_shift(res, vs)
        runs[ell2] = (res, s)
        rH_str = (f"{s['r_H_backreacted']:.4f}"
                  if s['r_H_backreacted'] is not None else "  None")
        sh_str = (f"{s['shift']:+.4f}"
                  if s['shift'] is not None else "  None")
        print(f" {ell2:9.2e}  {res.verdict:12s}  {len(res.steps):3d}    "
              f"{rH_str}   {sh_str}")
    print()
    print("Interpretation:")
    print("  - ell^2 = 0 recovers the classical horizon (numerical noise floor).")
    print("  - small ell^2 shifts r_H INWARD: the back-reaction pulls the")
    print("    chronology horizon toward the cylinder R, shrinking the CTC region.")
    print("  - above a threshold ell^2_crit ~ 1e-4, the horizon is")
    print("    'swallowed' (driven to r < r_min, i.e., into the source).")
    print("  - this is consistent with Hawking's chronology-protection")
    print("    direction (quantum effects censor the exterior CTC region)")
    print("    but is a 2D Polyakov TOY, not a 4D self-consistent claim.")
    print()

    save_path = _make_plot(vs, runs)
    if save_path is not None:
        print(f"plot saved to {save_path}")


def _make_plot(vs, runs) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) F(r) profiles for several ell^2
    ax = axes[0]
    color_map = plt.get_cmap("viridis")
    for i, (ell2, (res, _)) in enumerate(runs.items()):
        color = color_map(i / max(1, len(runs) - 1))
        if res.steps:
            last = res.steps[-1]
            ax.plot(res.r_grid, last.F_grid, lw=1.4, color=color,
                    label=f"ell²={ell2:.0e}")
    ax.axvline(classical_horizon(vs), color="tab:red", ls="--", lw=1.0,
               label=f"classical r_H = {classical_horizon(vs):.3f}")
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("r")
    ax.set_ylabel("F(r)")
    ax.set_title("(a) back-reacted F profiles")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-1.2, 1.4)

    # (b) r_H shift vs ell^2
    ax = axes[1]
    ell2s = []
    shifts = []
    for ell2, (res, s) in runs.items():
        ell2s.append(float(ell2) + 1e-12)
        shifts.append(s["shift"] if s["shift"] is not None else float("nan"))
    ax.semilogx(ell2s, shifts, marker="o", lw=1.6)
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("ell^2")
    ax.set_ylabel("r_H(back-reacted) - r_H(classical)")
    ax.set_title("(b) horizon shift vs back-reaction strength")

    # (c) r_H trajectory for one representative ell^2
    ax = axes[2]
    target_ell2 = 1e-4
    if target_ell2 in runs:
        res, _ = runs[target_ell2]
        history = [h if h is not None else np.nan for h in res.r_H_history]
        ax.plot(range(1, len(history) + 1), history, marker="o", lw=1.6)
        ax.axhline(classical_horizon(vs), color="tab:red", ls="--", lw=1.0,
                   label=f"classical r_H = {classical_horizon(vs):.3f}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("r_H(n)")
        ax.set_title(f"(c) r_H trajectory at ell²={target_ell2:.0e}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Toy iterative semiclassical back-reaction on supercritical LP",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
