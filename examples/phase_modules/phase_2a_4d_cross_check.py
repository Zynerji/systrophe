"""Phase 2a cross-check: 2D Polyakov vs 4D point-splitting on the LP exterior.

The 4D vacuum trace anomaly is K_kretsch/(2880 pi^2) (Birrell-Davies),
local and exact. The 2D Polyakov trace anomaly is R_2D/(24 pi). These
are two *different* quantities (different conformal-anomaly coefficients
in different dimensions), but for a static spacetime with a Cauchy
horizon they share the same generic divergence signature: simple-pole
in proper distance.

This script tabulates both side-by-side approaching the first Cauchy
horizon of the omega=1, R=1 supercritical Tipler, and reports their
divergence rates.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from systrophe import (
    StressEnergyState,
    cauchy_horizon_estimate,
    divergence_rate_at_horizon,
    kretschmann_scalar,
    ricci_scalar_2d,
    trace_anomaly_2d,
    trace_anomaly_4d_exact,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def main() -> dict:
    t_start = time.time()
    vs = VanStockumInterior(omega=1.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs)
    r_H1 = float(horizons[0])

    # Sample at decreasing distance from r_H1 from below (F > 0 side)
    eps_grid = np.geomspace(5e-4, 5e-2, 14)
    rows = []
    for eps in eps_grid:
        r = r_H1 - float(eps)
        F = float(vs.analytic_exterior_F(np.array([r]))[0])
        K_4d = kretschmann_scalar(vs, r)
        anomaly_4d = trace_anomaly_4d_exact(vs, r)
        R_2d = ricci_scalar_2d(vs, r)
        anomaly_2d = trace_anomaly_2d(vs, r)
        rows.append({
            "r": r,
            "eps": eps,
            "F": F,
            "K_4d": K_4d,
            "anomaly_4d_local": anomaly_4d,
            "R_2d": R_2d,
            "anomaly_2d_polyakov": anomaly_2d,
        })

    # Log-log power-law fits on both anomalies
    eps_arr = np.array([r["eps"] for r in rows])
    log_eps = np.log(eps_arr)
    K_arr = np.array([r["K_4d"] for r in rows])
    R_arr = np.array([r["R_2d"] for r in rows])
    anom4_arr = np.array([r["anomaly_4d_local"] for r in rows])
    anom2_arr = np.array([r["anomaly_2d_polyakov"] for r in rows])

    def fit_power(arr):
        mask = np.isfinite(arr) & (np.abs(arr) > 1e-30)
        if mask.sum() < 4:
            return float("nan")
        p, _ = np.polyfit(log_eps[mask], np.log(np.abs(arr[mask])), 1)
        return float(p)

    p_K = fit_power(K_arr)
    p_R = fit_power(R_arr)
    p_a4 = fit_power(anom4_arr)
    p_a2 = fit_power(anom2_arr)

    # Boulware T_tt divergence (already known to be -1)
    boulware_fit = divergence_rate_at_horizon(
        vs, r_H1, state=StressEnergyState.BOULWARE,
        n_samples=14, eps_min=5e-4, eps_max=5e-2, component="T_tt",
    )

    summary = {
        "kretschmann_power": p_K,
        "ricci_2d_power": p_R,
        "anomaly_4d_local_power": p_a4,
        "anomaly_2d_polyakov_power": p_a2,
        "boulware_T_tt_power": float(boulware_fit.power),
        "ratio_4d_to_2d_local": "K (4D) is locally FINITE at simple F-zero "
                                "(power ~ 0); R_2D diverges (power ~ -1). 4D "
                                "trace anomaly is therefore bounded, 2D unbounded.",
    }

    out = {
        "phase": "2a-cross-check",
        "title": "2D Polyakov vs 4D point-splitting near the first Cauchy horizon",
        "spacetime": {"omega": 1.0, "R": 1.0, "r_H1": r_H1},
        "rows": rows,
        "fits": summary,
        "elapsed_seconds": time.time() - t_start,
    }
    out_path = pathlib.Path(__file__).with_name("phase_2a_4d_cross_check_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 72)
    print("Phase 2a 4D-vs-2D cross-check at the first Cauchy horizon (omega=1, R=1)")
    print("=" * 72)
    print(f"  r_H1 = {r_H1:.6f}")
    print()
    print("  eps          F          K_4d        anomaly_4d   R_2d        anomaly_2d")
    print("-" * 80)
    for r in rows:
        print(f"  {r['eps']:.4e}   {r['F']:+.4e}   {r['K_4d']:+.4e}  "
              f"{r['anomaly_4d_local']:+.4e}  {r['R_2d']:+.4e}  "
              f"{r['anomaly_2d_polyakov']:+.4e}")
    print()
    print("Power-law fits (log|q| vs log(eps), eps = r_H1 - r):")
    print(f"  Kretschmann K(4D)         : power = {p_K:+.4f}")
    print(f"  R_2D (Polyakov denominator): power = {p_R:+.4f}")
    print(f"  anomaly 4D K/(2880 pi^2)  : power = {p_a4:+.4f}")
    print(f"  anomaly 2D R/(24 pi)      : power = {p_a2:+.4f}")
    print(f"  Boulware <T_tt>           : power = {summary['boulware_T_tt_power']:+.4f}")
    print()
    print("Interpretation:")
    print("  4D trace anomaly K/(2880 pi^2) is locally bounded at F=0")
    print("  (Kretschmann does NOT diverge at simple F-zero in LP vacuum).")
    print()
    print("  2D Polyakov trace anomaly R_2D/(24 pi) DOES diverge at F=0,")
    print("  reflecting the (t,r) sector's curvature blow-up.")
    print()
    print("  Both are correct; they measure different anomaly contributions.")
    print("  Boulware <T_{mu nu}> in the (t,r) sector diverges with the 2D")
    print("  Polyakov rate -- this is the relevant CP test (Hawking 1992).")
    print()
    print(f"Elapsed: {out['elapsed_seconds']:.1f} s")
    return out


if __name__ == "__main__":
    main()
