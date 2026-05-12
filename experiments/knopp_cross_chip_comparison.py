"""Cross-chip comparison: Knopp Drive band-gating on Kingston vs Marrakesh.

Both ibm_kingston and ibm_marrakesh ran the IDENTICAL 16-point r-sweep
circuit (batch 7, 8192 shots/point, dynamical decoupling). Two different
Heron-r2 chips with different physical qubits, calibration, and noise
profiles.

This script:
  1. Loads both per-chip JSON files.
  2. Computes the absolute per-r difference and the chip-averaged signal.
  3. Plots overlay of both chips with shot-noise error bars.
  4. Fits the chip-AVERAGED data to the smoothed-step band-gating model
     for a tighter combined band-edge estimate.
  5. Reports cross-chip RMS difference vs shot-noise budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def band_model(r, p_idle, p_active, r_in, w_in, r_out, w_out):
    sig_in = 1.0 / (1.0 + np.exp(-(r - r_in) / max(w_in, 1e-3)))
    sig_out = 1.0 / (1.0 + np.exp(-(r_out - r) / max(w_out, 1e-3)))
    return p_idle + (p_active - p_idle) * sig_in * sig_out


def main() -> None:
    here = Path(__file__).parent
    kingston = json.loads((here / "results" / "marrakesh_batch7_kingston_hw_analysis.json").read_text())
    marrakesh = json.loads((here / "results" / "marrakesh_batch7_marrakesh_hw_analysis.json").read_text())

    rs = np.array([p["r"] for p in kingston["per_circuit"]])
    p_kg = np.array([p["P_data1_observed"] for p in kingston["per_circuit"]])
    p_mr = np.array([p["P_data1_observed"] for p in marrakesh["per_circuit"]])

    shots = 8192
    sigma_kg = np.sqrt(p_kg * (1.0 - p_kg) / shots)
    sigma_mr = np.sqrt(p_mr * (1.0 - p_mr) / shots)

    diff = p_kg - p_mr
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    pooled_shot_noise = float(np.sqrt(np.mean((sigma_kg ** 2 + sigma_mr ** 2))))
    print("Cross-chip agreement on the 16-point r-sweep:")
    print(f"  RMS(Kingston - Marrakesh) = {rms_diff:.4f}")
    print(f"  Pooled shot-noise sigma    = {pooled_shot_noise:.4f}")
    print(f"  RMS / shot-noise sigma     = {rms_diff / pooled_shot_noise:.2f}")
    print()

    # Combined-chip fit
    p_avg = (p_kg + p_mr) / 2
    sigma_avg = np.sqrt(sigma_kg ** 2 + sigma_mr ** 2) / 2
    sigma_avg = np.maximum(sigma_avg, 0.003)
    p0 = [0.07, 0.65, 2.6, 0.1, 5.5, 0.3]
    bounds = ([0, 0, 1, 0.01, 3, 0.05], [0.5, 1, 4, 1, 6.5, 2])
    popt, pcov = curve_fit(band_model, rs, p_avg, p0=p0, sigma=sigma_avg,
                            absolute_sigma=True, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    print("Combined-chip smoothed-step fit:")
    print(f"  r_edge_in   = {popt[2]:.4f} +- {perr[2]:.4f}")
    print(f"  r_edge_out  = {popt[4]:.4f} +- {perr[4]:.4f}")
    print(f"  P_idle      = {popt[0]:.4f} +- {perr[0]:.4f}")
    print(f"  P_active    = {popt[1]:.4f} +- {perr[1]:.4f}")
    print()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8),
                                          sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    # Top: overlay of both chips + combined fit
    r_dense = np.linspace(rs.min() - 0.1, rs.max() + 0.1, 800)
    ax_top.errorbar(rs, p_kg, yerr=sigma_kg, fmt="o", color="tab:blue",
                    markersize=6, capsize=2, label="Kingston (8192 shots)")
    ax_top.errorbar(rs, p_mr, yerr=sigma_mr, fmt="s", color="tab:red",
                    markersize=5, capsize=2, label="Marrakesh (8192 shots)")
    ax_top.plot(r_dense, band_model(r_dense, *popt), "k-", lw=2,
                label=f"Combined-chip fit (r_in={popt[2]:.3f}, r_out={popt[4]:.3f})")
    ax_top.set_ylabel("P(data = 1)")
    ax_top.set_title("Knopp Drive batch 7: Kingston vs Marrakesh (two Heron-r2 chips)")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(alpha=0.3)
    ax_top.set_ylim(0, 0.8)

    # Bottom: residuals (Kingston - Marrakesh) in units of pooled sigma
    pooled = np.sqrt(sigma_kg ** 2 + sigma_mr ** 2)
    ax_bot.errorbar(rs, (p_kg - p_mr) / pooled, yerr=1.0,
                    fmt="o", color="tab:purple", capsize=2)
    ax_bot.axhline(0, color="k", linestyle="-", alpha=0.5)
    ax_bot.axhline(1, color="gray", linestyle=":")
    ax_bot.axhline(-1, color="gray", linestyle=":")
    ax_bot.set_xlabel("Cylinder radius r")
    ax_bot.set_ylabel("(K - M) / sigma_pool")
    ax_bot.set_title(f"Cross-chip residuals (RMS = {rms_diff/pooled_shot_noise:.2f} sigma)")
    ax_bot.grid(alpha=0.3)
    ax_bot.set_ylim(-3, 3)

    fig.tight_layout()
    out_dir = here.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "knopp_cross_chip.png", dpi=160)
    fig.savefig(out_dir / "knopp_cross_chip.pdf")
    print(f"Wrote {out_dir / 'knopp_cross_chip.png'}")
    print(f"Wrote {out_dir / 'knopp_cross_chip.pdf'}")

    json_out = here / "results" / "knopp_cross_chip_comparison.json"
    json_out.write_text(json.dumps({
        "r_grid": rs.tolist(),
        "P_data1_kingston":  p_kg.tolist(),
        "P_data1_marrakesh": p_mr.tolist(),
        "shot_noise_sigma_kg":  sigma_kg.tolist(),
        "shot_noise_sigma_mr":  sigma_mr.tolist(),
        "rms_cross_chip_diff": rms_diff,
        "pooled_shot_noise":  pooled_shot_noise,
        "rms_in_sigma_units": rms_diff / pooled_shot_noise,
        "combined_fit": {
            "P_idle":     {"value": float(popt[0]), "sigma": float(perr[0])},
            "P_active":   {"value": float(popt[1]), "sigma": float(perr[1])},
            "r_edge_in":  {"value": float(popt[2]), "sigma": float(perr[2])},
            "width_in":   {"value": float(popt[3]), "sigma": float(perr[3])},
            "r_edge_out": {"value": float(popt[4]), "sigma": float(perr[4])},
            "width_out":  {"value": float(popt[5]), "sigma": float(perr[5])},
        },
    }, indent=2))
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
