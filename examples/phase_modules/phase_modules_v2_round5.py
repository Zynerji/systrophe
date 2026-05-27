"""Round 5 v2-catcher sweep: 10 more modules, several deliberately
targeting the LP r in [1.46, 2.55] cluster regime so we can
characterize the density of catcher hits across observables there.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.catchers.catcher_v2 import (
    scan_novelty_consensus, scan_novelty_coherence_anomaly,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def run_v2(parameter_axis, fn, label):
    cons = scan_novelty_consensus(parameter_axis, fn,
                                  z_threshold=2.5, consensus_fraction=0.30)
    anom = scan_novelty_coherence_anomaly(parameter_axis, fn,
                                          sharp_jump_threshold=0.25,
                                          min_components=2)
    return {
        "module": label,
        "consensus": cons.verdict,
        "consensus_max": float(cons.max_consensus),
        "consensus_param": float(cons.max_consensus_parameter),
        "consensus_sharps": [
            {"p": float(s["parameter_value"]),
             "z": float(s.get("max_z", 0)),
             "frac": float(s["consensus_fraction"])}
            for s in cons.sharp_features[:3]
        ],
        "anomaly": anom.verdict,
        "anomaly_max": float(anom.max_delta_coherence),
        "anomaly_param": float(anom.max_delta_coherence_parameter),
        "anomaly_sharps": [
            {"p": float(s["parameter_value"]),
             "delta": float(s["delta_coherence"]),
             "before": float(s["coherence_before"]),
             "after": float(s["coherence_after"])}
            for s in anom.sharp_features[:3]
        ],
    }


def m_acoustic_metric():
    from systrophe.analogs.acoustic_metric import (
        acoustic_line_element_at_radius,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            d = acoustic_line_element_at_radius(vs, float(r))
            keys = sorted(k for k, v in d.items()
                          if isinstance(v, (int, float)))
            return np.array([float(d[k]) for k in keys])
        except Exception:
            return np.array([0.0])
    return run_v2(r_grid, fn, "acoustic_metric")


def m_frame_dragging():
    from systrophe.geometry.frame_dragging import (
        lense_thirring_frequency, gyroscope_precession_angle,
        total_drag_per_revolution, compare_to_kerr_LT,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            lt = lense_thirring_frequency(vs, float(r))
            gp = gyroscope_precession_angle(vs, float(r), time=1.0)
            dr = total_drag_per_revolution(vs, float(r))
            kerr = compare_to_kerr_LT(vs, float(r))
            kerr_lt = float(kerr.get("kerr_lt", 0.0))
            return np.array([lt, gp, dr, kerr_lt])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "frame_dragging")


def m_geodesic_completeness():
    from systrophe.geometry.geodesic_completeness import (
        timelike_omega_bounds, is_orbit_timelike,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            F = vs.analytic_exterior_F(np.array([float(r)]))[0]
            K = vs.analytic_exterior_K(np.array([float(r)]))[0]
            L = vs.analytic_exterior_L(np.array([float(r)]))[0]
            lo, hi = timelike_omega_bounds(F, K, L, float(r))
            tl = float(is_orbit_timelike(vs, float(r)))
            return np.array([lo, hi, hi - lo, tl])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "geodesic_completeness")


def m_higher_spin_fields():
    from systrophe.qftcs.higher_spin_fields import (
        solve_vector_mode, mode_energy,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    omegas = np.linspace(0.3, 3.0, 18)
    def fn(omega):
        try:
            modes_amps = []
            for m in (0, 1, 2):
                vm = solve_vector_mode(vs, float(omega), m=m, k=0.0,
                                       r_min=1.05, r_max=5.0, n_grid=101,
                                       initial_amplitude=1e-3)
                E = mode_energy(vm, vs)
                modes_amps.append(float(E))
            return np.array(modes_amps)
        except Exception:
            return np.zeros(3)
    return run_v2(omegas, fn, "higher_spin_fields")


def m_lensing_image():
    from systrophe.geometry.lensing_image import (
        impact_parameter_bare, null_circular_omega,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            b_pro = impact_parameter_bare(vs, float(r), branch="prograde")
            b_ret = impact_parameter_bare(vs, float(r), branch="retrograde")
            F = vs.analytic_exterior_F(np.array([float(r)]))[0]
            K = vs.analytic_exterior_K(np.array([float(r)]))[0]
            L = vs.analytic_exterior_L(np.array([float(r)]))[0]
            om_p, om_m = null_circular_omega(F, K, L, float(r))
            return np.array([
                float(b_pro) if np.isfinite(b_pro) else 0.0,
                float(b_ret) if np.isfinite(b_ret) else 0.0,
                float(om_p), float(om_m),
            ])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "lensing_image")


def m_quantum_diagnostics():
    from systrophe.qftcs.quantum_diagnostics import (
        tolman_blueshift_factor, ricci_scalar, conformal_anomaly_2d_proxy,
        chronology_protection_indicator,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            F = float(vs.analytic_exterior_F(np.array([float(r)]))[0])
            tol = float(tolman_blueshift_factor(np.array([F]))[0])
            ric = float(ricci_scalar(vs, float(r)))
            conf = float(conformal_anomaly_2d_proxy(vs, float(r)))
            chron = float(chronology_protection_indicator(np.array([F]))[0])
            return np.array([tol, ric, conf, chron])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "quantum_diagnostics")


def m_exotic_matter_accounting():
    from systrophe.qftcs.exotic_matter_accounting import (
        morris_thorne_exotic_density, morris_thorne_exotic_budget,
        required_plate_separation, casimir_cavity_energy,
    )
    r_throats = np.linspace(0.3, 5.0, 25)
    def fn(rt):
        try:
            ed = morris_thorne_exotic_density(float(rt), db_dr_at_throat=0.5)
            bud = morris_thorne_exotic_budget(float(rt), cavity_length=1.0,
                                              db_dr_at_throat=0.5)
            sep = required_plate_separation(float(rt), db_dr_at_throat=0.5)
            cas = casimir_cavity_energy(float(rt), plate_separation=0.1)
            return np.array([
                math.log10(max(abs(ed), 1e-60)),
                math.log10(max(abs(bud), 1e-60)),
                math.log10(max(abs(sep), 1e-60)),
                math.log10(max(abs(cas), 1e-60)),
            ])
        except Exception:
            return np.zeros(4)
    return run_v2(r_throats, fn, "exotic_matter_accounting")


def m_krasnikov_tube():
    from systrophe.geometry.krasnikov_tube import (
        krasnikov_energy_density, krasnikov_NEC_radial,
        krasnikov_metric_components,
    )
    rho_grid = np.linspace(0.05, 5.0, 30)
    def fn(rho):
        try:
            ed = krasnikov_energy_density(float(rho), rho_0=1.0, epsilon=0.1)
            nec = krasnikov_NEC_radial(float(rho), rho_0=1.0, epsilon=0.1)
            mc = krasnikov_metric_components(float(rho), rho_0=1.0, epsilon=0.1)
            keys = sorted(k for k, v in mc.items() if isinstance(v, (int, float)))
            metric_vals = [float(mc[k]) for k in keys[:3]]
            return np.array([float(ed), float(nec), *metric_vals])
        except Exception:
            return np.zeros(5)
    return run_v2(rho_grid, fn, "krasnikov_tube")


def m_lentz_soliton():
    from systrophe.geometry.lentz_soliton import (
        lentz_energy_density, lentz_NEC_radial, lentz_shift_vector,
    )
    rho_grid = np.linspace(0.05, 5.0, 30)
    def fn(rho):
        try:
            ed = lentz_energy_density(float(rho), rho_0=1.0)
            nec = lentz_NEC_radial(float(rho), rho_0=1.0)
            sv = lentz_shift_vector(float(rho), rho_0=1.0)
            if hasattr(sv, "__len__"):
                arr = np.array(sv[:3])
                if len(arr) < 3:
                    arr = np.pad(arr, (0, 3 - len(arr)))
                shifts = arr
            else:
                shifts = np.array([float(sv), 0.0, 0.0])
            return np.array([float(ed), float(nec), *shifts])
        except Exception:
            return np.zeros(5)
    return run_v2(rho_grid, fn, "lentz_soliton")


def m_multi_cylinder_dynamics():
    from systrophe.geometry.multi_cylinder_dynamics import (
        TiplerSinusoid, beat_frequency_pair, beat_log_period,
        phasor_extinction_check, n_cylinder_extinction_phases,
    )
    Ns = np.arange(2, 14).astype(float)
    def fn(N_float):
        N = int(round(N_float))
        try:
            phases = n_cylinder_extinction_phases(N)
            extinguished = float(phasor_extinction_check(phases))
            s1 = TiplerSinusoid(alpha=1.5, R=1.0)
            s2 = TiplerSinusoid(alpha=1.5 + 1.0/N, R=1.0)
            beat = beat_frequency_pair(s1, s2)
            blog = beat_log_period(beat)
            return np.array([
                float(N),
                extinguished,
                float(beat),
                float(blog) if np.isfinite(blog) else 0.0,
            ])
        except Exception:
            return np.zeros(4)
    return run_v2(Ns, fn, "multi_cylinder_dynamics")


def main():
    runners = [
        m_acoustic_metric, m_frame_dragging, m_geodesic_completeness,
        m_higher_spin_fields, m_lensing_image, m_quantum_diagnostics,
        m_exotic_matter_accounting, m_krasnikov_tube, m_lentz_soliton,
        m_multi_cylinder_dynamics,
    ]
    results = []
    print("Round 5 v2-catcher sweep across 10 more phase modules")
    print("=" * 70)
    for r in runners:
        try:
            res = r()
        except Exception as e:
            res = {"module": r.__name__, "error": str(e)}
        if res is None:
            print(f"\n{r.__name__}: SKIPPED")
            continue
        results.append(res)
        if "error" in res:
            print(f"\n{res['module']}: ERROR - {res['error']}")
            continue
        print(f"\n{res['module']}:")
        print(f"  consensus: {res['consensus']} (max={res['consensus_max']:.3f}"
              f"{' @ p=' + str(round(res['consensus_param'], 3)) if res['consensus_max'] > 0 else ''})")
        for s in res.get("consensus_sharps", [])[:2]:
            print(f"    sharp @ p={s['p']:.3f}: z={s['z']:.2f}, frac={s['frac']:.2f}")
        print(f"  anomaly:   {res['anomaly']} (max_delta={res['anomaly_max']:.3f}"
              f"{' @ p=' + str(round(res['anomaly_param'], 3)) if res['anomaly_max'] > 0 else ''})")
        for s in res.get("anomaly_sharps", [])[:2]:
            print(f"    sharp @ p={s['p']:.3f}: delta={s['delta']:.3f} "
                  f"({s['before']:.2f}->{s['after']:.2f})")

    out_path = Path(__file__).parent / "phase_modules_v2_round5_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print()
    new = []
    for r in results:
        if "error" in r:
            continue
        if r["anomaly"] == "novel_structure":
            new.append(f"{r['module']} (anomaly, p={r['anomaly_param']:.3f}, "
                       f"delta={r['anomaly_max']:.2f})")
    print(f"NEW ANOMALY EMERGENTS: {len(new)}")
    for e in new:
        print(f"  - {e}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
