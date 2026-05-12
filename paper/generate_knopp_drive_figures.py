"""Generate the four figures for paper/knopp_drive.pdf.

Reads the published JSON results from examples/ and experiments/results/
and renders publication-grade figures into paper/figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_tipler_gate_factor() -> None:
    """Figure 1: Tipler gate factor vs r for omega=1, R=1.

    Shows the gate factor's zeros (inside CTC bands) and nonzero
    plateaus (between bands).
    """
    from systrophe.tipler_krasnikov_hybrid import tipler_tilt_at
    from systrophe.vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=1.0, R=1.0)
    rs = np.linspace(1.05, 12.0, 400)
    tilts = np.array([tipler_tilt_at(vs, float(r)) for r in rs])
    gate = np.clip(1.0 - tilts, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(rs, gate, color="C0", linewidth=1.8, label="gate factor $(1-t(r))_+$")
    ax.fill_between(rs, 0, 1, where=(gate == 0), color="C0", alpha=0.15,
                     label="CTC band (gate=0)")
    ax.set_xlabel("orbit radius $r$")
    ax.set_ylabel("Tipler gate factor")
    ax.set_title("Tipler CTC-band gating: exotic matter is forced to zero "
                  "inside each band\n($\\omega=1$, $R=1$)")
    ax.set_xlim(rs[0], rs[-1])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "knopp_tipler_gate.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_knopp_drive_E_neg_vs_r() -> None:
    """Figure 2: Composite Knopp |E_neg| vs r at fixed Q.

    Confirms the headline shortcut: zero exotic matter inside each
    Tipler CTC band.
    """
    from systrophe.knopp_drive import KnoppDriveConfig, knopp_budget

    rs = np.linspace(1.05, 12.0, 200)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for Q in (10.0, 100.0):
        E_neg = []
        for r in rs:
            b = knopp_budget(KnoppDriveConfig(Q=Q, r_orbit=float(r)))
            E_neg.append(abs(b.composite_E_neg))
        ax.semilogy(rs, np.clip(E_neg, 1e-12, None),
                     linewidth=1.6, label=f"$Q={int(Q)}$")
    ax.set_xlabel("orbit radius $r$")
    ax.set_ylabel("composite $|E_{\\mathrm{neg}}|$ (geometric units)")
    ax.set_title("Knopp Drive exotic-matter budget: zero inside Tipler bands, "
                  "$1/Q^2$ scaling outside")
    ax.set_xlim(rs[0], rs[-1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out_path = FIG_DIR / "knopp_E_neg_vs_r.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_marrakesh_batch_6_hw() -> None:
    """Figure 3: Hardware confirmation of Knopp Drive on ibm_marrakesh.

    P(data=1) sim vs HW across 8 r-points spanning the first CTC band
    exit.
    """
    hw_path = REPO / "experiments" / "results" / "marrakesh_batch6_hw_analysis.json"
    if not hw_path.exists():
        print(f"skip (missing {hw_path})")
        return
    d = json.loads(hw_path.read_text())
    pc = d["per_circuit"]
    r_vals = [c["r"] for c in pc]
    p_sim = [c["P_data1_predicted"] for c in pc]
    p_hw = [c["P_data1_observed"] for c in pc]
    gate = [c["tipler_gate_factor"] for c in pc]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(r_vals, p_sim, "o-", color="C0", label="simulator", linewidth=1.8)
    ax.plot(r_vals, p_hw, "s--", color="C3", label="ibm\\_marrakesh HW",
             linewidth=1.8, markersize=8)
    ax.fill_between(r_vals, 0, 1,
                     where=[g == 0 for g in gate], color="C0", alpha=0.1,
                     label="inside CTC band")
    ax.set_xlabel("orbit radius $r$")
    ax.set_ylabel("$P(\\mathrm{data}=1)$")
    ax.set_title("Hardware confirmation: Knopp Drive CTC-band gating on "
                  "ibm\\_marrakesh (batch 6)")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "knopp_marrakesh_batch6.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_warp_comparison_bar() -> None:
    """Figure 4: Bar chart comparing exotic-matter requirements
    across the five canonical warp-drive families at Earth-Mars
    distance L = 0.52.
    """
    comp_path = REPO / "examples" / "warp_drive_comparison_results.json"
    if not comp_path.exists():
        print(f"skip (missing {comp_path})")
        return
    d = json.loads(comp_path.read_text())
    rows = d["comparison"]
    labels = [r["family"] for r in rows]
    e_negs = [r["E_neg"] for r in rows]
    # Replace zeros with a tiny floor for log-scale plotting
    e_plot = [max(e, 1e-12) for e in e_negs]
    colors = ["C0" if "Knopp" in lbl else "C7" for lbl in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    y = np.arange(len(labels))
    ax.barh(y, e_plot, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl.replace("_", "\\_") for lbl in labels], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("integrated $|E_{\\mathrm{neg}}|$ (geometric units, "
                   "$L = 0.52$ Earth-Mars equivalent)")
    ax.set_title("Knopp Drive vs the canonical warp-drive families "
                  "(zero exotic matter inside CTC band)")
    ax.axvline(1e-12, color="C0", linestyle="--", alpha=0.6,
                label="zero (Knopp inside band)")
    ax.grid(True, alpha=0.3, axis="x", which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = FIG_DIR / "knopp_warp_comparison.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    fig_tipler_gate_factor()
    fig_knopp_drive_E_neg_vs_r()
    fig_marrakesh_batch_6_hw()
    fig_warp_comparison_bar()


if __name__ == "__main__":
    main()
