"""Phase 2b: 4D Hadamard biparametrix + mode-sum on the supercritical Tipler.

Computes the local 4D Hadamard coefficients V_0(x), V_1(x), the
Hadamard-subtracted WKB partial mode-sum <phi^2>(r), the trace anomaly
recovered through V_1, and the 4D chronology-protection scan (does V_1
diverge at the Cauchy horizons?).

Result preview:
  V_0 = 0 (vacuum + massless + conformal)
  V_1 = K/720 (kept as point check)
  2 V_1 / (8 pi^2) matches point_splitting K/(2880 pi^2) at 1e-10 rel
  V_1 stays BOUNDED at every Cauchy horizon (power ~ 0).
  -> the chronology-protection signal is state-dependent (Phase 2a)
     and NOT in the 4D local biparametrix.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from systrophe import (
    cauchy_horizon_estimate,
    hadamard_chronology_report,
    hadamard_modesum_novelty_scan,
    hadamard_subtraction_residual,
    hadamard_V_0,
    hadamard_V_1,
    kretschmann_scalar,
    phi_squared_wkb_partial_sum,
    trace_anomaly_4d_exact,
    trace_anomaly_via_V_1,
)
from systrophe.vanstockum import VanStockumInterior


def _fit_to_dict(fit) -> dict:
    return {
        "r_horizon": fit.r_horizon,
        "quantity": fit.quantity,
        "power": fit.power,
        "amplitude_log": fit.amplitude_log,
        "n_samples_fit": fit.n_samples_fit,
        "fit_residual_rms": fit.fit_residual_rms,
        "diverges": fit.diverges,
    }


def main() -> dict:
    t_start = time.time()
    vs = VanStockumInterior(omega=2.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs).tolist()
    r_grid = [0.5 * (vs.R + horizons[0]), 1.20, 1.30]

    # 1. Local Hadamard coefficients across regular radii
    local_table = []
    for r in r_grid:
        V_0 = hadamard_V_0(vs, float(r), mass=0.0, xi=1.0 / 6.0)
        V_1 = hadamard_V_1(vs, float(r))
        K = kretschmann_scalar(vs, float(r))
        anom_V1 = trace_anomaly_via_V_1(vs, float(r))
        anom_ps = trace_anomaly_4d_exact(vs, float(r))
        local_table.append({
            "r": float(r),
            "V_0": float(V_0),
            "V_1": float(V_1),
            "K_kretschmann": float(K),
            "trace_anomaly_via_V_1": float(anom_V1),
            "trace_anomaly_point_splitting": float(anom_ps),
            "rel_diff_anomaly": float(abs(anom_V1 - anom_ps) / max(abs(anom_ps), 1e-30)),
        })

    # 2. WKB partial mode-sum + Hadamard subtraction at a regular radius
    wkb_summary = []
    r_test = 1.20
    for omega_max in (5.0, 10.0, 20.0, 40.0):
        psum = phi_squared_wkb_partial_sum(
            vs, r_test, omega_max=omega_max, n_omega=80, n_phi_max=2,
        )
        sub = hadamard_subtraction_residual(
            vs, r_test, omega_max=omega_max, n_omega=80, n_phi_max=2,
        )
        wkb_summary.append({
            "omega_max": float(omega_max),
            "partial_sum": psum["phi_squared_partial"],
            "uv_subtract": psum["leading_UV_scaling"],
            "hadamard_subtracted": float(sub),
        })

    # 3. 4D chronology-protection scan
    rep = hadamard_chronology_report(
        vs, n_horizons=3, n_samples=14, eps_min=1e-3, eps_max=3e-2,
    )

    # 4. Novelty catcher (always-on)
    novelty = hadamard_modesum_novelty_scan(vs, n_radii=32)

    elapsed = time.time() - t_start
    out = {
        "phase": "2b",
        "title": "4D Hadamard biparametrix + WKB mode-sum on supercritical Tipler",
        "spacetime": {"omega": float(vs.omega), "R": float(vs.R), "a": float(vs.a),
                       "alpha": float(vs.alpha)},
        "horizons": horizons,
        "local_hadamard_table": local_table,
        "wkb_modesum_convergence": wkb_summary,
        "chronology_report": {
            "n_horizons_scanned": rep.n_horizons_scanned,
            "horizons": list(rep.horizons),
            "V_1_fits": [_fit_to_dict(f) for f in rep.V_1_fits],
            "kretschmann_fits": [_fit_to_dict(f) for f in rep.kretschmann_fits],
            "summary": rep.summary,
        },
        "novelty_scan": {
            "verdict": novelty["verdict"],
            "n_sharp_features": novelty["n_sharp_features"],
            "sharp_features": novelty["sharp_features"],
            "lambda_2_at_radius": {str(k): v for k, v in novelty["lambda_2_at_radius"].items()},
            "n_radii": novelty["n_radii"],
        },
        "elapsed_seconds": elapsed,
    }

    out_path = pathlib.Path(__file__).with_name("phase_2b_hadamard_modesum_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 72)
    print("Phase 2b: 4D Hadamard biparametrix + WKB mode-sum")
    print("=" * 72)
    print("Local Hadamard coefficients (vacuum, massless, conformal):")
    print("  r       V_0        V_1        K_4d       anom (V_1)  anom (PS)  reldiff")
    for row in local_table:
        print(f"  {row['r']:.3f}   {row['V_0']:+.3e}  {row['V_1']:+.3e}  "
              f"{row['K_kretschmann']:+.3e}  {row['trace_anomaly_via_V_1']:+.3e}  "
              f"{row['trace_anomaly_point_splitting']:+.3e}  {row['rel_diff_anomaly']:.2e}")
    print()
    print("WKB mode-sum convergence at r = 1.20:")
    print("  omega_max  partial_sum    uv_subtract    hadamard_subtracted")
    for w in wkb_summary:
        print(f"  {w['omega_max']:6.1f}    {w['partial_sum']:+.4e}   "
              f"{w['uv_subtract']:+.4e}   {w['hadamard_subtracted']:+.4e}")
    print()
    print("4D chronology-protection scan (V_1 across Cauchy horizons):")
    for f in rep.V_1_fits:
        print(f"  r_H = {f.r_horizon:.4f}   power = {f.power:+.4f}   "
              f"rms = {f.fit_residual_rms:.2e}   diverges = {f.diverges}")
    print()
    print(rep.summary)
    print()
    print(f"Novelty catcher verdict: {novelty['verdict']} "
          f"({novelty['n_sharp_features']} sharp)")
    print()
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
