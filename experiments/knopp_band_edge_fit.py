"""Quantitative band-edge fit on Knopp Drive batch 7 (Kingston) data.

Fits a smoothed-step model

    P(d=1; r) = P_idle + (P_active - P_idle) * Pi(r)

where Pi(r) is the Knopp Drive band-gating profile

    Pi(r) = sigmoid((r - r_edge_in) / w_in)
          * sigmoid((r_edge_out - r) / w_out)

and (P_idle, P_active, r_edge_in, w_in, r_edge_out, w_out) are fit by
least-squares to the 16 Kingston r-points.

The fit gives the FOUR observables that matter for engineering:

  - r_edge_in (band entry radius), and its 1-sigma uncertainty
  - r_edge_out (band exit radius), and its 1-sigma uncertainty
  - active-band width = r_edge_out - r_edge_in
  - signal contrast = P_active - P_idle

These are the quantities a propulsion engineer would need to design a
Knopp Drive r-sweep schedule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def band_gating_model(r, p_idle, p_active, r_in, w_in, r_out, w_out):
    sig_in = 1.0 / (1.0 + np.exp(-(r - r_in) / max(w_in, 1e-3)))
    sig_out = 1.0 / (1.0 + np.exp(-(r_out - r) / max(w_out, 1e-3)))
    return p_idle + (p_active - p_idle) * sig_in * sig_out


def fit_batch7(json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    per = data["per_circuit"]
    rs = np.array([p["r"] for p in per])
    p_d1 = np.array([p["P_data1_observed"] for p in per])
    # Shot-noise sigma per circuit (8192 shots) — binomial std
    shots = 8192
    sigma = np.sqrt(p_d1 * (1.0 - p_d1) / shots)
    sigma = np.maximum(sigma, 0.005)  # floor for stability

    # Initial guess from inspection
    p0 = [0.07, 0.65, 2.6, 0.15, 5.0, 0.5]
    bounds = (
        [0.0, 0.0, 1.0, 0.01, 3.0, 0.05],
        [0.5, 1.0, 4.0, 1.0, 6.5, 2.0],
    )
    popt, pcov = curve_fit(
        band_gating_model, rs, p_d1, p0=p0, sigma=sigma,
        absolute_sigma=True, bounds=bounds, maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    p_idle, p_active, r_in, w_in, r_out, w_out = popt
    s_idle, s_active, s_r_in, s_w_in, s_r_out, s_w_out = perr

    # Goodness of fit
    p_pred = band_gating_model(rs, *popt)
    residuals = (p_d1 - p_pred) / sigma
    chi2 = float(np.sum(residuals ** 2))
    dof = len(rs) - len(popt)
    reduced_chi2 = chi2 / max(dof, 1)

    # Contrast and SNR
    contrast = p_active - p_idle
    contrast_unc = float(np.sqrt(s_idle ** 2 + s_active ** 2))
    snr = contrast / contrast_unc if contrast_unc > 0 else float("inf")
    band_width = r_out - r_in
    band_width_unc = float(np.sqrt(s_r_in ** 2 + s_r_out ** 2))

    return {
        "fit_parameters": {
            "P_idle":   {"value": float(p_idle),   "sigma": float(s_idle)},
            "P_active": {"value": float(p_active), "sigma": float(s_active)},
            "r_edge_in":  {"value": float(r_in),    "sigma": float(s_r_in)},
            "width_in":   {"value": float(w_in),    "sigma": float(s_w_in)},
            "r_edge_out": {"value": float(r_out),   "sigma": float(s_r_out)},
            "width_out":  {"value": float(w_out),   "sigma": float(s_w_out)},
        },
        "derived": {
            "contrast":         float(contrast),
            "contrast_sigma":   float(contrast_unc),
            "snr":              float(snr),
            "band_width":       float(band_width),
            "band_width_sigma": float(band_width_unc),
            "r_edge_in_central":  float(r_in),
            "r_edge_out_central": float(r_out),
        },
        "goodness": {
            "chi2": chi2,
            "dof": dof,
            "reduced_chi2": float(reduced_chi2),
            "n_data_points": len(rs),
        },
        "predicted_p_d1": p_pred.tolist(),
        "residuals_sigma_units": residuals.tolist(),
        "r_grid": rs.tolist(),
        "observed_p_d1": p_d1.tolist(),
    }


def main() -> None:
    path = Path(__file__).parent / "results" / "marrakesh_batch7_kingston_hw_analysis.json"
    out = fit_batch7(path)

    print("=" * 70)
    print("Knopp Drive band-edge fit (Kingston batch 7, 16 r-points)")
    print("=" * 70)
    print()
    p = out["fit_parameters"]
    d = out["derived"]
    g = out["goodness"]
    print(f"  P_idle      = {p['P_idle']['value']:.4f}  +- {p['P_idle']['sigma']:.4f}")
    print(f"  P_active    = {p['P_active']['value']:.4f}  +- {p['P_active']['sigma']:.4f}")
    print(f"  r_edge_in   = {p['r_edge_in']['value']:.3f}  +- {p['r_edge_in']['sigma']:.3f}")
    print(f"  width_in    = {p['width_in']['value']:.3f}  +- {p['width_in']['sigma']:.3f}")
    print(f"  r_edge_out  = {p['r_edge_out']['value']:.3f}  +- {p['r_edge_out']['sigma']:.3f}")
    print(f"  width_out   = {p['width_out']['value']:.3f}  +- {p['width_out']['sigma']:.3f}")
    print()
    print(f"  contrast    = {d['contrast']:.4f}  +- {d['contrast_sigma']:.4f}   "
          f"(SNR = {d['snr']:.1f})")
    print(f"  band_width  = {d['band_width']:.3f}  +- {d['band_width_sigma']:.3f}")
    print(f"  chi^2 / dof = {g['chi2']:.2f} / {g['dof']} = "
          f"{g['reduced_chi2']:.3f}")
    print()

    out_path = path.parent / "knopp_band_edge_fit_kingston_batch7.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
