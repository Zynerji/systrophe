"""Phase 2a: end-to-end quantitative chronology-protection report.

Implements Hawking's chronology-protection prediction as a measurable
power-law fit on the supercritical Tipler exterior.

Run: ``python examples/phase_2a_chronology_protection.py``

Outputs:
    examples/phase_2a_chronology_protection_results.json  — full report
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from systrophe import (
    StressEnergyState,
    boulware_stress_tensor,
    cauchy_horizon_estimate,
    chronology_protection_novelty_scan,
    chronology_protection_report,
    divergence_rate_at_horizon,
    hartle_hawking_stress_tensor,
    stress_tensor,
    surface_gravity_at_horizon,
    unruh_stress_tensor,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def _fit_to_dict(fit) -> dict:
    return {
        "r_horizon": fit.r_horizon,
        "component": fit.component,
        "power": fit.power,
        "amplitude_log": fit.amplitude_log,
        "n_samples_fit": fit.n_samples_fit,
        "state": fit.state,
        "fit_residual_rms": fit.fit_residual_rms,
        "diverges": fit.diverges,
        "sample_r": list(fit.sample_r),
        "sample_T": list(fit.sample_T),
    }


def main() -> dict:
    t_start = time.time()

    # Reference cylinder: a = 2.0, R = 1.0 (alpha ~ 3.87; three horizons in [R, 10R])
    vs = VanStockumInterior(omega=2.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs).tolist()

    # 1. Full multi-horizon, multi-state chronology-protection report
    rep = chronology_protection_report(
        vs, n_horizons=3, n_samples=20, eps_min=5e-4, eps_max=3e-2,
    )

    # 2. Per-component fits at the first horizon (T_tt vs T_rr)
    r_H1 = float(horizons[0])
    tt_fit = divergence_rate_at_horizon(
        vs, r_H1, state=StressEnergyState.BOULWARE, component="T_tt",
        n_samples=24, eps_min=5e-4, eps_max=3e-2,
    )
    rr_fit = divergence_rate_at_horizon(
        vs, r_H1, state=StressEnergyState.BOULWARE, component="T_rr",
        n_samples=24, eps_min=5e-4, eps_max=3e-2,
    )

    # 3. Surface gravities + Hawking temperatures at the three horizons
    horizon_data = []
    for rh in horizons[:3]:
        kappa = surface_gravity_at_horizon(vs, float(rh))
        T_H = kappa / (2.0 * np.pi)
        horizon_data.append({
            "r_horizon": float(rh),
            "surface_gravity": float(kappa),
            "hawking_temperature": float(T_H),
            "ratio_to_first": float(rh) / horizons[0],
        })

    # 4. Sample <T_mu_nu> in all three states at one regular radius
    r_sample = 0.5 * (vs.R + horizons[0])  # midway between source and 1st horizon
    T_states = {
        "boulware": boulware_stress_tensor(vs, r_sample),
        "hartle_hawking": hartle_hawking_stress_tensor(vs, r_sample, r_horizon=r_H1),
        "unruh": unruh_stress_tensor(vs, r_sample, r_horizon=r_H1),
    }
    # Strip non-serialisable: convert numpy types
    for st, T in T_states.items():
        T_states[st] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in T.items()
        }

    # 5. Novelty catcher (always-on)
    novelty = chronology_protection_novelty_scan(vs, n_radii=48)

    elapsed = time.time() - t_start

    out = {
        "phase": "2a",
        "title": "Renormalised stress-energy on the supercritical Tipler exterior",
        "spacetime": {
            "type": "VanStockum exterior",
            "omega": float(vs.omega),
            "R": float(vs.R),
            "a": float(vs.a),
            "alpha": float(vs.alpha),
            "regime": vs.regime,
        },
        "cauchy_horizons": horizon_data,
        "boulware_T_tt_at_first_horizon": _fit_to_dict(tt_fit),
        "boulware_T_rr_at_first_horizon": _fit_to_dict(rr_fit),
        "stress_tensor_at_midpoint": {
            "r": float(r_sample),
            "states": T_states,
        },
        "report": {
            "n_horizons_scanned": rep.n_horizons_scanned,
            "horizons": list(rep.horizons),
            "boulware_fits": [_fit_to_dict(f) for f in rep.boulware_fits],
            "hartle_hawking_fits": [_fit_to_dict(f) for f in rep.hartle_hawking_fits],
            "unruh_fits": [_fit_to_dict(f) for f in rep.unruh_fits],
            "trace_anomaly_max_residual": rep.trace_anomaly_max_residual,
            "verdict": rep.verdict,
            "summary": rep.summary,
        },
        "novelty_scan": {
            "verdict": novelty["verdict"],
            "n_sharp_features": novelty["n_sharp_features"],
            "sharp_features": novelty["sharp_features"],
            "lambda_2_at_radius": {str(k): v for k, v in novelty["lambda_2_at_radius"].items()},
            "n_radii": novelty["n_radii"],
            "state": novelty["state"],
        },
        "elapsed_seconds": elapsed,
    }

    out_path = pathlib.Path(__file__).with_suffix("").with_name(
        "phase_2a_chronology_protection_results"
    ).with_suffix(".json")
    out_path.write_text(json.dumps(out, indent=2))

    # Console summary
    print("=" * 64)
    print(f"PHASE 2a: {out['title']}")
    print("=" * 64)
    print(f"Source: omega={vs.omega}, R={vs.R}, a={vs.a:.3f}, alpha={vs.alpha:.4f}")
    print()
    print("Cauchy horizons (closed form):")
    for h in horizon_data:
        print(f"  r_H = {h['r_horizon']:.4f}   kappa = {h['surface_gravity']:.4f}"
              f"   T_H = {h['hawking_temperature']:.4f}")
    print()
    print("Boulware <T_tt>_B divergence at each horizon (n_samples=20):")
    for f in rep.boulware_fits:
        print(f"  r_H = {f.r_horizon:.4f}  ->  power = {f.power:+.3f}  "
              f"(rms = {f.fit_residual_rms:.2e})  diverges = {f.diverges}")
    print()
    print(f"Trace-anomaly residual (Polyakov identity): {rep.trace_anomaly_max_residual:.2e}")
    print()
    print(f"Verdict: {rep.verdict}")
    print(rep.summary)
    print()
    print(f"Novelty catcher verdict: {novelty['verdict']} "
          f"({novelty['n_sharp_features']} sharp features)")
    print()
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Results written to: {out_path}")
    return out


if __name__ == "__main__":
    main()
