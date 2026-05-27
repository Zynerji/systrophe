"""Single-figure summary of all Systrophe Heron-r2 QEC results.

Panels:
  A: d=3 Steane round sweep -- sub-threshold (logical < bare always)
  B: d=5 + d=7 Z-memory round sweep with Dijkstra-MWPM -- break-even
  C: d=7 long-rounds -- break-even at small n, methodological issue
     with bare baseline at long n
  D: logical X_L transversal gate -- symmetric P(L=0) and P(L=1)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def panel_a(ax):
    """d=3 Steane sub-threshold."""
    try:
        sw = json.loads(Path(__file__).parent.parent.joinpath(
            "experiments/results/steane_round_sweep_analysis.json"
        ).read_text())
        paired = sw["paired_by_n_rounds"]
        nr = sorted(int(k) for k in paired.keys())
        log = [paired[str(n)]["steane"]["logical_zero_rate"] for n in nr]
        bare = [paired[str(n)]["bare"]["physical_zero_rate"] for n in nr]
    except Exception as e:
        ax.set_title(f"A: Steane d=3 (data unavailable: {e})")
        return
    ax.plot(nr, log, "o-", color="tab:blue", lw=2, label="logical (Steane d=3)")
    ax.plot(nr, bare, "s-", color="tab:red", lw=2, label="bare qubit")
    ax.axhline(0.5, color="gray", linestyle=":", label="random")
    ax.set_xlabel("n_rounds")
    ax.set_ylabel("P(Z = 0)")
    ax.set_title("A: d=3 Steane code -- sub-threshold (logical < bare)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.4, 1.0)


def panel_b(ax):
    """d=5 + d=7 Z-memory with Dijkstra-MWPM."""
    try:
        d5 = json.loads(Path(__file__).parent.parent.joinpath(
            "experiments/results/surface_code_d5_dijkstra_mwpm_analysis.json"
        ).read_text())["paired_by_n_rounds"]
        d7 = json.loads(Path(__file__).parent.parent.joinpath(
            "experiments/results/surface_code_d7_and_d5_high_shots_dijkstra_analysis.json"
        ).read_text())["paired_by_d_n"]
    except Exception as e:
        ax.set_title(f"B: d=5 + d=7 (data unavailable: {e})")
        return

    # d=5 (low shots)
    d5_rounds = sorted(int(k) for k in d5.keys())
    d5_log = [d5[str(r)]["surface"]["logical_zero_rate_dijkstra_mwpm"]
              for r in d5_rounds]
    d5_bare = [d5[str(r)]["bare"]["physical_zero_rate"] for r in d5_rounds]
    ax.plot(d5_rounds, d5_log, "o-", color="tab:blue", lw=2, label="d=5 logical (MWPM)")
    ax.plot(d5_rounds, d5_bare, "s--", color="tab:blue", lw=1, alpha=0.6,
             label="d=5 bare")
    # d=7 (high shots)
    d7_rounds = []
    d7_log = []
    d7_bare = []
    for key, v in d7.items():
        if not key.startswith("d7_"):
            continue
        nr = v["surface"]["n_rounds"]
        d7_rounds.append(nr)
        d7_log.append(v["surface"]["logical_zero_rate_dijkstra_mwpm"])
        d7_bare.append(v["bare"]["physical_zero_rate"])
    order = np.argsort(d7_rounds)
    d7_rounds = [d7_rounds[i] for i in order]
    d7_log = [d7_log[i] for i in order]
    d7_bare = [d7_bare[i] for i in order]
    ax.plot(d7_rounds, d7_log, "^-", color="tab:red", lw=2, label="d=7 logical (MWPM)")
    ax.plot(d7_rounds, d7_bare, "v--", color="tab:red", lw=1, alpha=0.6,
             label="d=7 bare")
    ax.axhline(0.5, color="gray", linestyle=":", label="random")
    ax.set_xlabel("n_rounds")
    ax.set_ylabel("P(L = 0)")
    ax.set_title("B: d=5 and d=7 surface code -- sustained break-even")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.4, 1.0)


def panel_c(ax):
    """d=7 long rounds + sat point."""
    try:
        d7short = json.loads(Path(__file__).parent.parent.joinpath(
            "experiments/results/surface_code_d7_and_d5_high_shots_dijkstra_analysis.json"
        ).read_text())["paired_by_d_n"]
        d7long = json.loads(Path(__file__).parent.parent.joinpath(
            "experiments/results/surface_code_d7_long_rounds_analysis.json"
        ).read_text())
    except Exception as e:
        ax.set_title(f"C: d=7 long-n (data unavailable: {e})")
        return

    short_rounds, short_log, short_bare = [], [], []
    for key, v in d7short.items():
        if not key.startswith("d7_"):
            continue
        short_rounds.append(v["surface"]["n_rounds"])
        short_log.append(v["surface"]["logical_zero_rate_dijkstra_mwpm"])
        short_bare.append(v["bare"]["physical_zero_rate"])
    order = np.argsort(short_rounds)
    short_rounds = [short_rounds[i] for i in order]
    short_log = [short_log[i] for i in order]
    short_bare = [short_bare[i] for i in order]

    long_rounds = sorted(int(k) for k in d7long.keys())
    long_log = [d7long[str(r)]["dijk"] for r in long_rounds]
    long_bare = [d7long[str(r)]["bare"] for r in long_rounds]

    all_r = short_rounds + long_rounds
    all_log = short_log + long_log
    all_bare = short_bare + long_bare

    ax.plot(all_r, all_log, "o-", color="tab:purple", lw=2, label="d=7 logical")
    ax.plot(all_r, all_bare, "s-", color="tab:orange", lw=2, label="bare baseline")
    ax.axvline(4, color="green", linestyle=":", alpha=0.5)
    ax.text(4, 0.97, "break-even\nregime", color="green", fontsize=8,
            ha="center")
    ax.axvline(8, color="red", linestyle=":", alpha=0.5)
    ax.text(8, 0.97, "bare-baseline\nartifacts", color="red", fontsize=8,
            ha="center")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("n_rounds")
    ax.set_ylabel("P(L = 0)")
    ax.set_title("C: d=7 across n_rounds in {1, 2, 4, 8, 16}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.0)


def panel_d(ax):
    """Logical X_L transversal gate."""
    # Hardcode from the previous decode output
    data = {
        1: {"no_xl": 0.8257, "with_xl_p1": 0.8102},
        4: {"no_xl": 0.6157, "with_xl_p1": 0.6436},
    }
    nrs = sorted(data.keys())
    no_xl = [data[n]["no_xl"] for n in nrs]
    xl = [data[n]["with_xl_p1"] for n in nrs]
    width = 0.35
    x = np.arange(len(nrs))
    ax.bar(x - width/2, no_xl, width, color="tab:blue", label="No X_L: P(L=0)")
    ax.bar(x + width/2, xl, width, color="tab:red", label="X_L applied: P(L=1)")
    ax.axhline(0.5, color="gray", linestyle=":", label="random")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nrs])
    ax.set_xlabel("n_rounds")
    ax.set_ylabel("Logical-state recovery fidelity")
    ax.set_title("D: Transversal X_L preserves codespace (symmetric)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.4, 1.0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.suptitle("Systrophe Heron-r2 QEC results on ibm_kingston",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_dir = Path(__file__).parent.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qec_heron_r2_summary.png", dpi=160)
    fig.savefig(out_dir / "qec_heron_r2_summary.pdf")
    print(f"Wrote {out_dir / 'qec_heron_r2_summary.png'}")
    print(f"Wrote {out_dir / 'qec_heron_r2_summary.pdf'}")


if __name__ == "__main__":
    main()
