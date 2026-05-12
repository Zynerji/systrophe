"""Multi-resonance substructure fit on Knopp Drive batch 7 active-band.

The single-step band-gating model in `knopp_band_edge_fit.py` gives
chi^2/dof = 16.2, meaning the active-band plateau is NOT flat -- there
is real substructure beyond shot noise.

This fitter superimposes K Lorentzian peaks onto the smoothed-step
envelope. Each peak is parameterised by (centre, width, amplitude),
modelling internal Knopp Drive resonances within the CTC band.

Output: per-peak centres + widths, total chi^2 reduction vs the single
sigmoid model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def model(r, *params):
    """Envelope x (1 + sum_k a_k * Lorentz(r; r_k, w_k))

    params: p_idle, p_active, r_in, w_in, r_out, w_out,
            then K triplets (a_k, r_k, w_k)
    """
    p_idle, p_active, r_in, w_in, r_out, w_out = params[:6]
    rest = params[6:]
    n_peaks = len(rest) // 3
    sig_in = 1.0 / (1.0 + np.exp(-(r - r_in) / max(w_in, 1e-3)))
    sig_out = 1.0 / (1.0 + np.exp(-(r_out - r) / max(w_out, 1e-3)))
    env = p_idle + (p_active - p_idle) * sig_in * sig_out
    multiplier = np.ones_like(r)
    for k in range(n_peaks):
        a, r_k, w_k = rest[3 * k:3 * k + 3]
        multiplier += a * (w_k ** 2) / ((r - r_k) ** 2 + w_k ** 2)
    return env * multiplier


def fit_substructure(json_path: Path, n_peaks: int = 2) -> dict:
    data = json.loads(json_path.read_text())
    per = data["per_circuit"]
    rs = np.array([p["r"] for p in per])
    p_d1 = np.array([p["P_data1_observed"] for p in per])
    shots = 8192
    sigma = np.sqrt(p_d1 * (1.0 - p_d1) / shots)
    sigma = np.maximum(sigma, 0.005)

    # Initial guess: envelope + n_peaks placed inside the active band
    p0_env = [0.06, 0.65, 2.6, 0.06, 5.5, 0.2]
    centres0 = np.linspace(3.0, 4.5, n_peaks)
    peak_init = []
    for r_k in centres0:
        peak_init.extend([0.05, float(r_k), 0.3])
    p0 = p0_env + peak_init

    # Bounds
    lb_env = [0.0, 0.0, 1.0, 0.01, 3.0, 0.05]
    ub_env = [0.5, 1.0, 4.0, 1.0, 6.5, 2.0]
    lb_peak = [-0.5, 2.5, 0.05] * n_peaks
    ub_peak = [+0.5, 5.5, 1.5] * n_peaks
    bounds = (lb_env + lb_peak, ub_env + ub_peak)

    popt, pcov = curve_fit(
        model, rs, p_d1, p0=p0, sigma=sigma, absolute_sigma=True,
        bounds=bounds, maxfev=40000,
    )
    perr = np.sqrt(np.diag(pcov))
    p_pred = model(rs, *popt)
    residuals = (p_d1 - p_pred) / sigma
    chi2 = float(np.sum(residuals ** 2))
    dof = len(rs) - len(popt)
    reduced_chi2 = chi2 / max(dof, 1)

    env = {
        "P_idle":     {"value": float(popt[0]), "sigma": float(perr[0])},
        "P_active":   {"value": float(popt[1]), "sigma": float(perr[1])},
        "r_edge_in":  {"value": float(popt[2]), "sigma": float(perr[2])},
        "width_in":   {"value": float(popt[3]), "sigma": float(perr[3])},
        "r_edge_out": {"value": float(popt[4]), "sigma": float(perr[4])},
        "width_out":  {"value": float(popt[5]), "sigma": float(perr[5])},
    }
    peaks = []
    for k in range(n_peaks):
        a = popt[6 + 3 * k]; r_k = popt[7 + 3 * k]; w_k = popt[8 + 3 * k]
        sa = perr[6 + 3 * k]; sr = perr[7 + 3 * k]; sw = perr[8 + 3 * k]
        peaks.append({
            "amplitude":   {"value": float(a),   "sigma": float(sa)},
            "centre":      {"value": float(r_k), "sigma": float(sr)},
            "width":       {"value": float(w_k), "sigma": float(sw)},
        })

    return {
        "n_peaks": n_peaks,
        "envelope": env,
        "peaks": peaks,
        "goodness": {
            "chi2": chi2,
            "dof": dof,
            "reduced_chi2": float(reduced_chi2),
            "n_data_points": len(rs),
        },
        "r_grid": rs.tolist(),
        "observed_p_d1": p_d1.tolist(),
        "predicted_p_d1": p_pred.tolist(),
    }


def main() -> None:
    path = Path(__file__).parent / "results" / "marrakesh_batch7_kingston_hw_analysis.json"

    print("=" * 70)
    print("Knopp Drive substructure fit (Kingston batch 7)")
    print("=" * 70)
    print()

    results = {}
    for n_peaks in (1, 2, 3):
        try:
            out = fit_substructure(path, n_peaks=n_peaks)
        except Exception as e:
            print(f"  n_peaks={n_peaks}: fit failed -- {e}")
            continue
        results[f"n_peaks_{n_peaks}"] = out
        print(f"--- {n_peaks} internal resonance peak{'s' if n_peaks != 1 else ''} ---")
        print(f"  chi^2/dof = {out['goodness']['chi2']:.2f} / "
              f"{out['goodness']['dof']} = "
              f"{out['goodness']['reduced_chi2']:.3f}")
        for k, pk in enumerate(out["peaks"]):
            print(f"  peak {k + 1}: r = {pk['centre']['value']:.3f} +- "
                  f"{pk['centre']['sigma']:.3f}  "
                  f"a = {pk['amplitude']['value']:+.3f}  "
                  f"w = {pk['width']['value']:.3f}")
        print()

    out_path = path.parent / "knopp_substructure_fit_kingston_batch7.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
