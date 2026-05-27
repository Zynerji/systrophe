"""End-to-end characterisation of the canonical omega=2, R=1 Tipler exterior.

Drives every Phase 1-3 method via the LPAnalyser/PairAnalyser public
API and prints + JSON-dumps the headline numbers. Reproduces the
Phase 2a/2b/3a/3b verdicts from `Systrophe/CHANGELOG.md` v0.20.0 +
v0.21.0 using only the lp-analyser tool.

Runtime: ~30 seconds.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lp_analyser import LPAnalyser, PairAnalyser
from systrophe.geometry.vanstockum import VanStockumInterior


def main():
    print("=" * 72)
    print("LPAnalyser: omega=2, R=1 supercritical Tipler exterior")
    print("=" * 72)

    a = LPAnalyser(omega=2.0, R=1.0)
    t0 = time.time()
    s = a.summary()
    t_summary = time.time() - t0

    print(f"  regime:                {s.regime}")
    print(f"  interior_regime:       {s.interior_regime}")
    print(f"  a = omega R:           {s.a:.3f}")
    print(f"  alpha = sqrt(4a^2-1):  {s.alpha:.4f}")
    print(f"  mass / length:         {s.mass_per_unit_length:.4e}")
    print(f"  ang. mom. / length:    {s.angular_momentum_per_unit_length:.4e}")
    print(f"  EC all satisfied:      {s.energy_conditions_all_satisfied}")
    print(f"  n Cauchy horizons:     {s.n_cauchy_horizons_in_10R}")
    print(f"  first horizon r:       {s.first_horizon_r:.4f}")
    print(f"  surface gravity kappa: {s.first_horizon_surface_gravity:.4f}")
    print(f"  Hawking T_H:           {s.first_horizon_hawking_temperature:.4f}")
    print(f"  Phase 2a Boulware T_tt power: {s.boulware_T_tt_simple_pole_power:+.4f}")
    print(f"  Phase 2a verdict:      {s.chronology_protection_verdict}")
    print(f"  Phase 2b V_1 (mid):    {s.hadamard_V_1_at_midpoint:+.4e}")
    print(f"  Kretschmann (mid):     {s.kretschmann_at_midpoint:+.4e}")
    print(f"  summary() elapsed:     {t_summary:.2f} s")
    print()

    # Detailed Phase 2a: divergence rate at each Cauchy horizon
    print("Phase 2a per-horizon Boulware <T_tt> divergence fit:")
    t0 = time.time()
    rep_2a = a.chronology_protection_scan(n_horizons=3, n_samples=18)
    print(f"  verdict: {rep_2a.verdict}")
    for f in rep_2a.boulware_fits:
        print(f"    r_H={f.r_horizon:.4f}  power={f.power:+.4f}  "
              f"rms={f.fit_residual_rms:.2e}  diverges={f.diverges}")
    print(f"  trace-anomaly residual: {rep_2a.trace_anomaly_max_residual:.2e}")
    print(f"  Phase 2a elapsed:       {time.time() - t0:.2f} s")
    print()

    # Detailed Phase 2b: V_1 stays bounded at each Cauchy horizon
    print("Phase 2b 4D Hadamard V_1 at each Cauchy horizon (should be bounded):")
    t0 = time.time()
    rep_2b = a.hadamard_chronology_scan(n_horizons=3)
    for f in rep_2b.V_1_fits:
        print(f"    r_H={f.r_horizon:.4f}  power={f.power:+.4f}  "
              f"rms={f.fit_residual_rms:.2e}  diverges={f.diverges}")
    print(f"  Phase 2b elapsed:       {time.time() - t0:.2f} s")
    print()

    # Trace-anomaly cross-check at midpoint
    r_mid = 0.5 * (a.R + s.first_horizon_r)
    check = a.trace_anomaly_check(r_mid)
    print(f"Phase 2b cross-check at r_mid = {r_mid:.3f}:")
    print(f"  2 V_1 / (8 pi^2):  {check['trace_anomaly_via_V_1']:+.6e}")
    print(f"  K / (2880 pi^2):   {check['trace_anomaly_via_point_splitting']:+.6e}")
    print(f"  rel diff:          {check['relative_difference']:.2e}")
    print()

    # Phase 3a: N-cylinder extinction
    print("=" * 72)
    print("PairAnalyser Phase 3a: N-fold extinction sweep (omega=1, R=1)")
    print("=" * 72)
    c = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.geometry.array import SystropheArray
    print(f"{'N':>4s}  {'max|A_factor|':>15s}  extinguished?")
    ext_rows = []
    for N in (2, 3, 4, 5, 6, 7, 8):
        # uniform_phase_comb method on the SystropheArray class
        comb = SystropheArray.uniform_phase_comb(c, N=N)
        # Wrap via PairAnalyser for API consistency
        from systrophe.geometry.sinusoid import TiplerSinusoid
        ext = comb.extinction_check(r_max=20.0)
        print(f"  {N}    {ext['max_array_factor']:.3e}    {ext['is_extinguished']}")
        ext_rows.append({"N": N, **ext})
    print()

    # Phase 3a: beam-steer to multiple radii
    print("Phase 3a: beam-steer L=0 to specific r_target")
    beam_rows = []
    for r_t in (2.0, 3.0, 5.0, 7.0):
        p = PairAnalyser.beam_steer(r_target=float(r_t), cylinder=c, N=2)
        L_target = float(p.L(np.array([r_t]))[0])
        print(f"  r_target={r_t:.2f}  L(r_target)={L_target:+.3e}")
        beam_rows.append({"r_target": float(r_t), "L_at_target": L_target})
    print()

    # Phase 3b: off-axis topology
    print("=" * 72)
    print("PairAnalyser Phase 3b: off-axis pair topology (separation=3.0)")
    print("=" * 72)
    c2 = VanStockumInterior(omega=1.0, R=1.0)
    p_off = PairAnalyser([c, c2], separation=3.0)
    topo_rows = []
    for nx, ny in ((41, 21), (81, 41)):
        t = p_off.ctc_region_topology(-3.0, 6.0, -3.0, 3.0, nx=nx, ny=ny)
        print(f"  grid {nx} x {ny}: n_comp={t['n_components']}  "
              f"n_holes={t['n_holes']}  ctc_frac={t['ctc_fraction']:.4f}  "
              f"topology={t['topology_summary']}")
        topo_rows.append({"nx": nx, "ny": ny, **{k: v for k, v in t.items()
                                                     if k != "component_areas"}})

    # JSON dump for reproducibility
    out = {
        "tool": "lp-analyser",
        "spacetime": {
            "omega": float(a.omega),
            "R": float(a.R),
            "a": float(a.a),
            "alpha": float(a.alpha),
        },
        "summary": {
            "regime": s.regime,
            "interior_regime": s.interior_regime,
            "n_cauchy_horizons_in_10R": s.n_cauchy_horizons_in_10R,
            "first_horizon_r": s.first_horizon_r,
            "first_horizon_surface_gravity": s.first_horizon_surface_gravity,
            "first_horizon_hawking_temperature": s.first_horizon_hawking_temperature,
            "boulware_T_tt_simple_pole_power": s.boulware_T_tt_simple_pole_power,
            "chronology_protection_verdict": s.chronology_protection_verdict,
            "hadamard_V_1_at_midpoint": s.hadamard_V_1_at_midpoint,
            "kretschmann_at_midpoint": s.kretschmann_at_midpoint,
        },
        "phase_2a_horizons": [
            {"r_horizon": f.r_horizon, "power": f.power,
              "fit_residual_rms": f.fit_residual_rms, "diverges": f.diverges}
            for f in rep_2a.boulware_fits
        ],
        "phase_2a_trace_anomaly_residual": rep_2a.trace_anomaly_max_residual,
        "phase_2b_horizons": [
            {"r_horizon": f.r_horizon, "power": f.power,
              "fit_residual_rms": f.fit_residual_rms}
            for f in rep_2b.V_1_fits
        ],
        "phase_2b_trace_anomaly_check": check,
        "phase_3a_extinction": ext_rows,
        "phase_3a_beam_steer": beam_rows,
        "phase_3b_topology": topo_rows,
    }
    out_path = pathlib.Path(__file__).with_name("characterize_supercritical_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
