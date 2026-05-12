"""Extract per-round logical error rate vs distance from the Heron-r2
QEC results. Fit the threshold-theorem prediction

    p_L(d, n) = 1 - exp(-Λ_d * n)
    Λ_d ~ A * (p_phys / p_th)^((d + 1) / 2)

to obtain Λ_d for each distance, then plot log Λ_d vs d. A linear fit
in (d+1)/2 gives an estimated threshold p_th if p_phys is known.

Uses existing data:
  d=3 Steane:    paper/steane_round_sweep_analysis.json
  d=5 Kingston:  surface_code_d5_dijkstra_mwpm_analysis.json
  d=5 Marrakesh: surface_code_d5_marrakesh_analysis.json
  d=7 Kingston:  surface_code_d7_and_d5_high_shots_dijkstra_analysis.json
  d=9 Kingston:  surface_code_d9_kingston_analysis.json (when available)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def decay_model(n, p_L_per_round, p0):
    """P(L=0; n) = 0.5 + (p0 - 0.5) * (1 - 2 * p_L_per_round)^n.

    Alternative: P(L=0; n) ~ p0 * exp(-Λ * n) for small p_L.
    Simpler form for fitting: exponential decay toward 0.5.
    """
    return 0.5 + (p0 - 0.5) * np.exp(-p_L_per_round * n)


def fit_curve(rounds, rates, shots=8192):
    n = np.array(rounds, dtype=float)
    r = np.array(rates, dtype=float)
    sigma = np.maximum(np.sqrt(r * (1 - r) / shots), 0.003)
    try:
        popt, _ = curve_fit(decay_model, n, r, p0=[0.05, 0.9],
                             sigma=sigma, absolute_sigma=True, maxfev=5000,
                             bounds=([0.0, 0.5], [1.0, 1.0]))
        return float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"  fit failed: {e}")
        return float("nan"), float("nan")


def main() -> None:
    here = Path(__file__).parent.parent / "experiments" / "results"

    series = {}

    # d=3 Steane (Kingston) — already has logical_zero_rate per n_rounds
    try:
        st = json.loads((here / "steane_round_sweep_analysis.json").read_text())
        paired = st["paired_by_n_rounds"]
        rounds = sorted(int(k) for k in paired.keys())
        log = [paired[str(n)]["steane"]["logical_zero_rate"] for n in rounds]
        series[("d=3 Steane", "Kingston")] = (rounds, log, 4096)
    except Exception as e:
        print(f"d=3 Steane data: {e}")

    # d=5 Kingston (4096-shot Dijkstra-MWPM)
    try:
        d5k = json.loads((here / "surface_code_d5_dijkstra_mwpm_analysis.json").read_text())
        paired = d5k["paired_by_n_rounds"]
        rounds = sorted(int(k) for k in paired.keys())
        log = [paired[str(n)]["surface"]["logical_zero_rate_dijkstra_mwpm"]
               for n in rounds]
        series[("d=5 surface", "Kingston")] = (rounds, log, 4096)
    except Exception as e:
        print(f"d=5 Kingston data: {e}")

    # d=5 Marrakesh
    try:
        d5m = json.loads((here / "surface_code_d5_marrakesh_analysis.json").read_text())
        rounds = sorted(int(k) for k in d5m.keys())
        log = [d5m[str(n)]["logical"] for n in rounds]
        series[("d=5 surface", "Marrakesh")] = (rounds, log, 8192)
    except Exception as e:
        print(f"d=5 Marrakesh data: {e}")

    # d=7 Kingston (high-shots Dijkstra)
    try:
        d7k = json.loads((here / "surface_code_d7_and_d5_high_shots_dijkstra_analysis.json").read_text())
        paired = d7k["paired_by_d_n"]
        rounds, log = [], []
        for k, v in paired.items():
            if k.startswith("d7_"):
                rounds.append(v["surface"]["n_rounds"])
                log.append(v["surface"]["logical_zero_rate_dijkstra_mwpm"])
        order = np.argsort(rounds)
        rounds = [rounds[i] for i in order]
        log = [log[i] for i in order]
        series[("d=7 surface", "Kingston")] = (rounds, log, 16384)
    except Exception as e:
        print(f"d=7 Kingston data: {e}")

    # d=9 Kingston (if available)
    d9_path = here / "surface_code_d9_kingston_analysis.json"
    if d9_path.exists():
        try:
            d9k = json.loads(d9_path.read_text())
            rounds = sorted(int(k) for k in d9k.keys())
            log = [d9k[str(n)]["logical"] for n in rounds]
            series[("d=9 surface", "Kingston")] = (rounds, log, 8192)
        except Exception as e:
            print(f"d=9 Kingston data: {e}")
    else:
        print("d=9 Kingston data not yet available")

    print()
    print("Per-round logical error rate fits (exponential decay model)")
    print("=" * 70)
    print(f"{'Code':>20}  {'Chip':>10}  {'p_L/round':>12}  {'p0':>10}  {'rounds':>10}")
    fits = {}
    for (code, chip), (rounds, rates, shots) in sorted(series.items()):
        if not rounds or len(rounds) < 2:
            continue
        p_L, p0 = fit_curve(rounds, rates, shots=shots)
        fits[(code, chip)] = (p_L, p0, rounds, rates)
        print(f"{code:>20}  {chip:>10}  {p_L:>12.5f}  {p0:>10.5f}  "
              f"{rounds}")

    # Plot logical vs rounds across distances
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10),
                                          gridspec_kw={"height_ratios": [3, 2]})
    colors = {"d=3 Steane": "tab:gray", "d=5 surface": "tab:blue",
              "d=7 surface": "tab:red", "d=9 surface": "tab:green"}
    markers = {"Kingston": "o", "Marrakesh": "s"}
    for (code, chip), (rounds, rates, shots) in sorted(series.items()):
        if not rounds:
            continue
        col = colors.get(code, "black")
        mk = markers.get(chip, "o")
        sigma = np.sqrt(np.array(rates)*(1-np.array(rates))/shots)
        ax_top.errorbar(rounds, rates, yerr=sigma, fmt=f"{mk}-",
                        color=col, lw=2, capsize=3,
                        label=f"{code} ({chip})")
    ax_top.axhline(0.5, color="black", linestyle=":", alpha=0.5,
                    label="random")
    ax_top.set_xlabel("n_rounds")
    ax_top.set_ylabel("Logical P(L = 0)")
    ax_top.set_title("Heron-r2 QEC: logical decay across distance + chip")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="lower left", fontsize=9)
    ax_top.set_ylim(0.4, 1.0)

    # Bottom panel: per-round logical error rate vs d
    d_vals = []
    pL_vals = []
    labels = []
    for (code, chip), (p_L, p0, _, _) in fits.items():
        if "d=3" in code:
            d = 3
        elif "d=5" in code:
            d = 5
        elif "d=7" in code:
            d = 7
        elif "d=9" in code:
            d = 9
        else:
            continue
        d_vals.append(d)
        pL_vals.append(p_L)
        labels.append(f"{code[:5]} {chip[0]}")
    if d_vals:
        ax_bot.scatter(d_vals, pL_vals, s=100, marker="o",
                       color=[colors.get(f"d={d} {('surface' if d>=5 else 'Steane')}",
                                          "black") for d in d_vals])
        for d, pL, lab in zip(d_vals, pL_vals, labels):
            ax_bot.annotate(lab, (d, pL), textcoords="offset points",
                             xytext=(7, 5), fontsize=8)
        ax_bot.set_yscale("log")
        ax_bot.set_xlabel("Code distance d")
        ax_bot.set_ylabel("Per-round logical error rate p_L")
        ax_bot.set_title("Threshold-theorem view: p_L vs d (Heron-r2)")
        ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    out_dir = Path(__file__).parent.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qec_threshold_analysis.png", dpi=160)
    fig.savefig(out_dir / "qec_threshold_analysis.pdf")
    print(f"\nWrote {out_dir / 'qec_threshold_analysis.png'}")
    print(f"Wrote {out_dir / 'qec_threshold_analysis.pdf'}")

    out_json = Path(__file__).parent / "results" / "qec_threshold_analysis.json"
    out_json.write_text(json.dumps({
        "fits": {f"{c}|{chip}": {"p_L_per_round": p_L, "p_0": p0}
                  for (c, chip), (p_L, p0, _, _) in fits.items()},
        "data": {f"{c}|{chip}": {"rounds": r, "rates": rates, "shots": shots}
                  for (c, chip), (r, rates, shots) in series.items()},
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
