"""Round 7 v2-catcher sweep: 10 more modules. Multiple sweep over LP
radial r to keep probing the [1.46, 2.55] cluster band."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.catcher_v2 import (
    scan_novelty_consensus, scan_novelty_coherence_anomaly,
)
from systrophe.vanstockum import VanStockumInterior


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


def m_anomaly_inflow():
    from systrophe.anomaly_inflow import (
        z3_branch_etas, z3_branch_twists, z3_total_eta,
        z3_anomaly_inflow_balance,
    )
    gammas = np.linspace(0.0, 2.0, 25)
    def fn(gamma):
        try:
            etas = z3_branch_etas(float(gamma))
            twists = z3_branch_twists(float(gamma))
            tot = z3_total_eta(float(gamma))
            bal = z3_anomaly_inflow_balance(float(gamma), B_z=1.0, area=1.0)
            bal_v = float(next(v for v in bal.values()
                               if isinstance(v, (int, float))))
            return np.array([*etas[:2], *twists[:2], float(tot), bal_v])
        except Exception:
            return np.zeros(6)
    return run_v2(gammas, fn, "anomaly_inflow")


def m_dynamical_cylinder():
    from systrophe.dynamical_cylinder import (
        instantaneous_ctc_measure, adiabatic_back_reaction_estimate,
    )
    omegas = np.linspace(0.3, 2.0, 25)
    def fn(omega):
        try:
            ctc = instantaneous_ctc_measure(float(omega), R=1.0,
                                            r_min=1.05, r_max=10.0,
                                            n_grid=1001)
            br = adiabatic_back_reaction_estimate(float(omega), R=1.0)
            vals = []
            for d in (ctc, br):
                for v in d.values():
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
            while len(vals) < 4: vals.append(0.0)
            return np.array(vals[:4])
        except Exception:
            return np.zeros(4)
    return run_v2(omegas, fn, "dynamical_cylinder")


def m_floquet_engineering():
    from systrophe.floquet_engineering import (
        analyze_floquet_mobius, ctc_stability_gap,
    )
    branch_energies = np.array([-1.0, 0.0, 1.0])
    omega_drive = 2.0
    drive_amps = np.linspace(0.0, 2.5, 25)
    def fn(amp):
        try:
            res = analyze_floquet_mobius(branch_energies, hopping=0.3,
                                          drive_amp=float(amp),
                                          omega_drive=omega_drive, n_steps=100)
            qe = res.quasi_energies if hasattr(res, "quasi_energies") else []
            gap = ctc_stability_gap(np.array(qe), omega_drive)
            qe_arr = np.array(qe[:3]) if len(qe) >= 3 else np.zeros(3)
            return np.array([float(gap), *qe_arr])
        except Exception:
            return np.zeros(4)
    return run_v2(drive_amps, fn, "floquet_engineering")


def m_g_renormalization():
    from systrophe.g_renormalization import (
        g_eff_at_r, curvature_scale_at_r,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            data = g_eff_at_r(vs, float(r))
            kscale = curvature_scale_at_r(vs, float(r))
            vals = [float(getattr(data, k))
                    for k in dir(data)
                    if not k.startswith('_')
                    and isinstance(getattr(data, k, None), (int, float))][:3]
            while len(vals) < 3: vals.append(0.0)
            return np.array([*vals, math.log10(max(abs(kscale), 1e-30))])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "g_renormalization")


def m_kg_scattering():
    from systrophe.kg_scattering import (
        effective_potential, transmission_coefficient,
        reflection_coefficient,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    omega = 1.0
    def fn(r):
        try:
            V_at_r = effective_potential(vs, float(r), omega=omega,
                                         n_phi=0, mass=0.0)
            # Transmission/reflection between (r_left, r_right):
            T = transmission_coefficient(vs, max(float(r) - 0.3, 1.05),
                                          float(r) + 0.3, omega=omega,
                                          n_phi=0, mass=0.0, n_steps=200)
            R = reflection_coefficient(vs, max(float(r) - 0.3, 1.05),
                                        float(r) + 0.3, omega=omega,
                                        n_phi=0, mass=0.0)
            return np.array([float(V_at_r), float(T), float(R)])
        except Exception:
            return np.zeros(3)
    return run_v2(r_grid, fn, "kg_scattering")


def m_gw_emission():
    from systrophe.gw_emission import (
        moment_of_inertia, cylindrical_resonant_frequencies,
        radial_oscillation_strain, spin_up_strain,
    )
    omegas = np.linspace(0.6, 2.5, 25)
    def fn(omega):
        try:
            vs = VanStockumInterior(omega=float(omega), R=1.0)
            I = moment_of_inertia(vs, cylinder_length=1.0)
            ros = radial_oscillation_strain(vs, dR_amplitude=0.01,
                                             omega_R=1.0,
                                             observer_distance=1.0,
                                             cylinder_length=1.0)
            sus = spin_up_strain(vs, alpha_spin=1.0,
                                  observer_distance=1.0,
                                  cylinder_length=1.0)
            ros_v = float(next(v for v in ros.values()
                               if isinstance(v, (int, float))))
            sus_v = float(next(v for v in sus.values()
                               if isinstance(v, (int, float))))
            return np.array([float(I), ros_v, sus_v])
        except Exception:
            return np.zeros(3)
    return run_v2(omegas, fn, "gw_emission")


def m_qi_channel():
    from systrophe.qi_channel import (
        redshift_factor, channel_capacity_holevo,
        amplitude_damping_p,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    r_A = 1.05
    def fn(r):
        try:
            rs = redshift_factor(vs, r_A, float(r))
            p = amplitude_damping_p(float(rs))
            cap = channel_capacity_holevo(float(p))
            return np.array([float(rs), float(p), float(cap)])
        except Exception:
            return np.zeros(3)
    return run_v2(r_grid, fn, "qi_channel")


def m_optical_fiber_analog():
    from systrophe.optical_fiber_analog import (
        fiber_analog_horizon, gaussian_pump_profile,
        pulse_collision_radiation,
    )
    amps = np.linspace(0.3, 0.99, 25)
    def fn(amp):
        try:
            prof = gaussian_pump_profile(amplitude=float(amp), sigma=1.0, x0=5.0)
            h = fiber_analog_horizon(prof)
            vals = [float(v) for v in h.values()
                    if isinstance(v, (int, float))][:2]
            while len(vals) < 2: vals.append(0.0)
            pc = pulse_collision_radiation(float(amp), 0.5)
            pc_v = float(next(v for v in pc.values()
                              if isinstance(v, (int, float))))
            return np.array([*vals, pc_v])
        except Exception:
            return np.zeros(3)
    return run_v2(amps, fn, "optical_fiber_analog")


def m_hawking_budget():
    from systrophe.hawking_budget import (
        bekenstein_hawking_entropy_at_CH, evaporation_time_estimate,
        hawking_temperature_at_CH, horizon_area_at_CH,
        surface_gravity_at_CH,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            S = bekenstein_hawking_entropy_at_CH(vs, float(r))
            t = evaporation_time_estimate(vs, float(r))
            T = hawking_temperature_at_CH(vs, float(r))
            A = horizon_area_at_CH(vs, float(r))
            kap = surface_gravity_at_CH(vs, float(r))
            return np.array([
                math.log10(max(abs(S), 1e-30)),
                math.log10(max(abs(t), 1e-30)),
                float(T), math.log10(max(abs(A), 1e-30)), float(kap),
            ])
        except Exception:
            return np.zeros(5)
    return run_v2(r_grid, fn, "hawking_budget")


def m_holographic():
    from systrophe.holographic import (
        brown_york_stress_tensor, dual_operator_dimensions,
        boundary_two_point_correlator,
    )
    omegas = np.linspace(0.6, 2.0, 20)
    def fn(omega):
        try:
            vs = VanStockumInterior(omega=float(omega), R=1.0)
            T = brown_york_stress_tensor(vs, r_boundary=20.0)
            t_vals = [float(v) for v in T.values()
                      if isinstance(v, (int, float))][:3]
            while len(t_vals) < 3: t_vals.append(0.0)
            corr = boundary_two_point_correlator(vs, separation=1.0)
            corr_re = float(np.real(corr))
            return np.array([*t_vals, corr_re])
        except Exception:
            return np.zeros(4)
    return run_v2(omegas, fn, "holographic")


def main():
    runners = [
        m_anomaly_inflow, m_dynamical_cylinder, m_floquet_engineering,
        m_g_renormalization, m_kg_scattering, m_gw_emission,
        m_qi_channel, m_optical_fiber_analog, m_hawking_budget,
        m_holographic,
    ]
    results = []
    print("Round 7 v2-catcher sweep across 10 more phase modules")
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

    out_path = Path(__file__).parent / "phase_modules_v2_round7_results.json"
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
