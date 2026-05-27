"""Plot Knopp Drive Kingston batch 7 band-edge + 2-resonance substructure fit.

Generates `paper/figures/knopp_band_edge_kingston.{png,pdf}` for the
whitepaper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smoothed_step(r, p_idle, p_active, r_in, w_in, r_out, w_out):
    s_in = 1.0 / (1.0 + np.exp(-(r - r_in) / max(w_in, 1e-3)))
    s_out = 1.0 / (1.0 + np.exp(-(r_out - r) / max(w_out, 1e-3)))
    return p_idle + (p_active - p_idle) * s_in * s_out


def envelope_plus_peaks(r, env_params, peaks):
    base = smoothed_step(r, *env_params)
    multiplier = np.ones_like(r)
    for pk in peaks:
        a = pk["amplitude"]["value"]
        r_k = pk["centre"]["value"]
        w_k = pk["width"]["value"]
        multiplier += a * (w_k ** 2) / ((r - r_k) ** 2 + w_k ** 2)
    return base * multiplier


def main() -> None:
    here = Path(__file__).parent
    hw = json.loads((here / "results" / "marrakesh_batch7_kingston_hw_analysis.json").read_text())
    edge = json.loads((here / "results" / "knopp_band_edge_fit_kingston_batch7.json").read_text())
    sub = json.loads((here / "results" / "knopp_substructure_fit_kingston_batch7.json").read_text())

    rs = np.array([p["r"] for p in hw["per_circuit"]])
    p_d1 = np.array([p["P_data1_observed"] for p in hw["per_circuit"]])
    sigma = np.sqrt(p_d1 * (1.0 - p_d1) / 8192)

    r_dense = np.linspace(rs.min() - 0.1, rs.max() + 0.1, 800)
    env_params = [
        edge["fit_parameters"]["P_idle"]["value"],
        edge["fit_parameters"]["P_active"]["value"],
        edge["fit_parameters"]["r_edge_in"]["value"],
        edge["fit_parameters"]["width_in"]["value"],
        edge["fit_parameters"]["r_edge_out"]["value"],
        edge["fit_parameters"]["width_out"]["value"],
    ]
    env_curve = smoothed_step(r_dense, *env_params)

    sub2 = sub["n_peaks_2"]
    sub2_env = [
        sub2["envelope"]["P_idle"]["value"],
        sub2["envelope"]["P_active"]["value"],
        sub2["envelope"]["r_edge_in"]["value"],
        sub2["envelope"]["width_in"]["value"],
        sub2["envelope"]["r_edge_out"]["value"],
        sub2["envelope"]["width_out"]["value"],
    ]
    full_curve = envelope_plus_peaks(r_dense, sub2_env, sub2["peaks"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(
        rs, p_d1, yerr=sigma, fmt="o", color="black",
        markersize=6, capsize=3, label="Kingston HW (8192 shots/point)",
    )
    ax.plot(r_dense, env_curve, "--", color="tab:blue", lw=2,
            label=f"Smoothed-step envelope ($\\chi^2/\\mathrm{{dof}}={edge['goodness']['reduced_chi2']:.1f}$)")
    ax.plot(r_dense, full_curve, "-", color="tab:red", lw=2,
            label=f"Envelope + 2 resonances ($\\chi^2/\\mathrm{{dof}}={sub2['goodness']['reduced_chi2']:.2f}$)")

    r_in = edge["fit_parameters"]["r_edge_in"]["value"]
    r_out = edge["fit_parameters"]["r_edge_out"]["value"]
    ax.axvline(r_in,  color="tab:green", linestyle=":", alpha=0.7,
               label=f"$r_\\mathrm{{in}}={r_in:.2f}$")
    ax.axvline(r_out, color="tab:orange", linestyle=":", alpha=0.7,
               label=f"$r_\\mathrm{{out}}={r_out:.2f}$")
    for k, pk in enumerate(sub2["peaks"]):
        r_k = pk["centre"]["value"]
        ax.axvline(r_k, color="tab:purple", linestyle="-.", alpha=0.4)

    ax.set_xlabel("Cylinder radius $r$ (units of $a=1$ Tipler cylinder)")
    ax.set_ylabel("$P(\\mathrm{data}=1)$  on ibm_kingston")
    ax.set_title("Knopp Drive band-edge fit (Kingston, 16-point r-sweep, batch 7)")
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    out_dir = here.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "knopp_band_edge_kingston.png", dpi=180)
    fig.savefig(out_dir / "knopp_band_edge_kingston.pdf")
    print(f"Wrote {out_dir / 'knopp_band_edge_kingston.png'}")
    print(f"Wrote {out_dir / 'knopp_band_edge_kingston.pdf'}")


if __name__ == "__main__":
    main()
