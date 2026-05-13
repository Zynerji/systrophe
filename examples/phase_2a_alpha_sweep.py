"""Phase 2a follow-up: alpha-universality check of the Polyakov simple-pole.

Sweeps the rotation parameter ``a = omega * R`` across the supercritical
range and confirms that the Boulware-stress-tensor power-law exponent at
the first Cauchy horizon is alpha-independent (= -1 for <T_tt>, = -2 for
<T_rr>). This is the analytic universality prediction of the 2D Polyakov
result: the divergence rate at a Killing horizon depends ONLY on F'_H,
not on the specific log-frequency alpha = sqrt(4 a^2 - 1).
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
)
from systrophe.vanstockum import VanStockumInterior


def main() -> dict:
    t_start = time.time()
    a_values = np.array([0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    R_value = 1.0
    rows = []
    for a in a_values:
        omega = float(a / R_value)
        vs = VanStockumInterior(omega=omega, R=R_value)
        horizons = cauchy_horizon_estimate(vs)
        if len(horizons) == 0:
            rows.append({
                "a": float(a),
                "alpha": float(vs.alpha),
                "n_horizons_in_10R": 0,
                "skipped": True,
            })
            continue
        r_H = float(horizons[0])
        tt = divergence_rate_at_horizon(
            vs, r_H, state=StressEnergyState.BOULWARE,
            n_samples=20, eps_min=5e-4, eps_max=2e-2, component="T_tt",
        )
        rr = divergence_rate_at_horizon(
            vs, r_H, state=StressEnergyState.BOULWARE,
            n_samples=20, eps_min=5e-4, eps_max=2e-2, component="T_rr",
        )
        rows.append({
            "a": float(a),
            "alpha": float(vs.alpha),
            "n_horizons_in_10R": int(len(horizons)),
            "r_H1": float(r_H),
            "T_tt_power": float(tt.power),
            "T_tt_fit_rms": float(tt.fit_residual_rms),
            "T_rr_power": float(rr.power),
            "T_rr_fit_rms": float(rr.fit_residual_rms),
            "diverges": bool(tt.diverges and rr.diverges),
        })

    # Universality check
    finite_rows = [r for r in rows if not r.get("skipped")]
    tt_powers = np.array([r["T_tt_power"] for r in finite_rows])
    rr_powers = np.array([r["T_rr_power"] for r in finite_rows])
    tt_spread = float(np.std(tt_powers))
    rr_spread = float(np.std(rr_powers))
    tt_mean = float(np.mean(tt_powers))
    rr_mean = float(np.mean(rr_powers))

    summary = {
        "tt_power_mean": tt_mean,
        "tt_power_std": tt_spread,
        "rr_power_mean": rr_mean,
        "rr_power_std": rr_spread,
        "tt_close_to_minus_one": float(abs(tt_mean - (-1.0))),
        "rr_close_to_minus_two": float(abs(rr_mean - (-2.0))),
        "universality_consistent": bool(
            abs(tt_mean - (-1.0)) < 0.1 and abs(rr_mean - (-2.0)) < 0.1
            and tt_spread < 0.1 and rr_spread < 0.1
        ),
    }

    out = {
        "phase": "2a-follow",
        "title": "Polyakov simple-pole universality across alpha",
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.time() - t_start,
    }
    out_path = pathlib.Path(__file__).with_name("phase_2a_alpha_sweep_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 64)
    print("Phase 2a follow-up: alpha-universality check")
    print("=" * 64)
    print("  a    alpha   r_H1      T_tt p   T_rr p   diverges")
    print("-" * 64)
    for r in rows:
        if r.get("skipped"):
            print(f"  {r['a']:.2f}  {r['alpha']:.4f}  --        --       --       (no horizon in 10R)")
        else:
            print(f"  {r['a']:.2f}  {r['alpha']:.4f}  {r['r_H1']:.4f}    "
                  f"{r['T_tt_power']:+.4f}  {r['T_rr_power']:+.4f}  {r['diverges']}")
    print()
    print(f"  mean T_tt power = {summary['tt_power_mean']:+.4f} +- {summary['tt_power_std']:.4f}  (theory: -1)")
    print(f"  mean T_rr power = {summary['rr_power_mean']:+.4f} +- {summary['rr_power_std']:.4f}  (theory: -2)")
    print(f"  universality_consistent = {summary['universality_consistent']}")
    print()
    print(f"Elapsed: {out['elapsed_seconds']:.1f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
