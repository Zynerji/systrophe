"""Four-panel summary figure for all Millennium-problem catcher
explorations landed in Systrophē.

Panels:
  A: Riemann zero spacing distribution (N=500), Wigner surmise overlay.
  B: 3-SAT P(SAT) vs alpha, derivative-catcher sharp features marked at alpha_c=4.27.
  C: Goldbach comet (g(n) vs n) coloured by n mod 6, conjecture-verified band.
  D: GUE null reference -- per_q novel rate vs N for Riemann RH-consistency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
EX = HERE.parent / "examples"


def panel_a(ax):
    """Riemann spacing distribution at N=500."""
    try:
        from mpmath import mp, zetazero
    except ImportError:
        ax.set_title("Panel A: mpmath unavailable")
        return
    mp.dps = 15
    zeros = np.array([float(zetazero(k).imag) for k in range(1, 501)])
    spacings = np.diff(zeros)
    t_c = (zeros[:-1] + zeros[1:]) / 2
    mean_local = 2 * math.pi / np.log(t_c / (2 * math.pi))
    s_norm = spacings / mean_local

    bins = np.linspace(0, 3, 31)
    counts, edges = np.histogram(s_norm, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, counts, width=edges[1] - edges[0], color="tab:blue",
           alpha=0.6, label="First 500 zeta zeros")
    # Wigner surmise (GUE)
    s_grid = np.linspace(0, 3, 200)
    wigner = (32.0 / math.pi ** 2) * s_grid ** 2 * np.exp(-4.0 * s_grid ** 2 / math.pi)
    ax.plot(s_grid, wigner, "r-", lw=2, label="GUE Wigner surmise")
    ax.set_xlabel("Normalised spacing $s$")
    ax.set_ylabel("Density")
    ax.set_title("A: Riemann zeta-zero spacings (N=500) — RH-consistent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def panel_b(ax):
    """3-SAT P(SAT) vs alpha with derivative-catcher sharp at alpha_c=4.27."""
    sat_path = EX / "millennium_sat_phase_transition_results.json"
    if not sat_path.exists():
        ax.set_title("Panel B: SAT data not yet generated")
        return
    j = json.loads(sat_path.read_text())
    alphas = [r["alpha"] for r in j["per_alpha"]]
    p_sat = [r["sat_fraction"] for r in j["per_alpha"]]
    ax.plot(alphas, p_sat, "o-", color="tab:green", lw=2,
            markersize=6, label="P(SAT) (n=20 vars, 60 inst/$\\alpha$)")
    centre = j["derivative_catcher"]["estimated_transition_centre"]
    if centre is not None:
        ax.axvline(centre, color="tab:red", linestyle="--", lw=2,
                   label=f"Derivative catcher centre: $\\alpha={centre:.3f}$")
    ax.axvline(4.267, color="tab:purple", linestyle=":", lw=1.5,
               label="Conjectured $\\alpha_c \\approx 4.267$")
    ax.set_xlabel("$\\alpha = m/n$")
    ax.set_ylabel("P(SAT)")
    ax.set_title("B: 3-SAT phase transition — catcher recovers $\\alpha_c$")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def panel_c(ax):
    """Goldbach comet, coloured by n mod 6."""
    gold_path = EX / "millennium_goldbach_catcher_n1000_results.json"
    if not gold_path.exists():
        ax.set_title("Panel C: Goldbach data not yet generated")
        return
    # Recompute (lighter than reloading from JSON which is summary only)
    import sys
    sys.path.insert(0, str(EX))
    from millennium_goldbach_catcher import compute_goldbach_comet

    data = compute_goldbach_comet(n_max=1000)
    evens = np.array(data["evens"])
    g = data["g_values"]
    for m, col in zip((0, 2, 4), ("tab:red", "tab:blue", "tab:green")):
        mask = evens % 6 == m
        ax.scatter(evens[mask], g[mask], s=8, alpha=0.6,
                   color=col, label=f"n mod 6 = {m}")
    ax.set_xlabel("Even integer $n$")
    ax.set_ylabel("$g(n)$ = Goldbach reps")
    ax.set_title("C: Goldbach comet up to $n=1000$ — conjecture holds, 3 bands")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def panel_d(ax):
    """GUE null reference: per_q novel rate vs N."""
    null_path = EX / "millennium_riemann_null_gue_results.json"
    if not null_path.exists():
        ax.set_title("Panel D: GUE null data not yet generated")
        return
    j = json.loads(null_path.read_text())
    summary = j.get("summary_per_N", {})
    Ns = sorted(int(k) for k in summary.keys())
    pq_rates = [100.0 * summary[str(n)]["per_q_novel_count"]
                / summary[str(n)]["total"] for n in Ns]
    sc_rates = [100.0 * summary[str(n)]["scan_novel_count"]
                / summary[str(n)]["total"] for n in Ns]
    ax.plot(Ns, sc_rates, "o-", color="tab:blue",
            label="scan_novelty flag rate (GUE null)")
    ax.plot(Ns, pq_rates, "s-", color="tab:red",
            label="per_quantity flag rate (GUE null)")
    # Mark the Riemann observations
    ax.scatter([100, 200, 500], [100, 100, 100], marker="x",
               color="tab:blue", s=80, label="Riemann scan novel")
    ax.scatter([500], [100], marker="*", color="tab:red",
               s=120, label="Riemann per_q novel")
    ax.set_xlabel("N (number of zeros / spacings)")
    ax.set_ylabel("Flag rate (%)")
    ax.set_title("D: GUE null reference — Riemann within finite-N fluctuation")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 105)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.suptitle("Systrophē Millennium-problem catcher explorations (v0.19.0)",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_dir = HERE.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "millennium_summary.png", dpi=160)
    fig.savefig(out_dir / "millennium_summary.pdf")
    print(f"Wrote {out_dir / 'millennium_summary.png'}")
    print(f"Wrote {out_dir / 'millennium_summary.pdf'}")


if __name__ == "__main__":
    main()
