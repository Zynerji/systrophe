"""Round 8 v2-catcher sweep: 10 more modules. Mix of LP-radial,
fractal/DSI, warp-shell, and vacuum-state observables."""

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


def m_adm_export():
    from systrophe.adm_export import (
        adm_decompose_lp, adm_summary, hamiltonian_constraint_residual,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    omegas = np.linspace(0.5, 2.0, 20)
    def fn(omega):
        try:
            vs_i = VanStockumInterior(omega=float(omega), R=1.0)
            r_grid = np.linspace(1.05, 5.0, 50)
            slice_data = adm_decompose_lp(vs_i, r_grid)
            summary = adm_summary(slice_data)
            ham_res = hamiltonian_constraint_residual(slice_data)
            vals = [float(v) for v in summary.values()
                    if isinstance(v, (int, float))][:3]
            while len(vals) < 3: vals.append(0.0)
            return np.array([*vals, math.log10(max(abs(np.mean(np.abs(ham_res))), 1e-30))])
        except Exception:
            return np.zeros(4)
    return run_v2(omegas, fn, "adm_export")


def m_feedback_amplified_shell():
    from systrophe.feedback_amplified_shell import (
        parametric_gain_to_steady, saturation_field_energy,
        required_drive_power, pfenning_ford_check, cavity_lifetime,
    )
    Qs = np.geomspace(1.0, 1000.0, 25)
    sigma = 8.0
    v_s, R = 1.0, 1.0
    def fn(Q):
        try:
            g = parametric_gain_to_steady(v_s, R, sigma, float(Q))
            sat = saturation_field_energy(v_s, R, sigma, float(Q))
            P = required_drive_power(v_s, R, sigma, float(Q))
            chk = pfenning_ford_check(v_s, R, sigma, float(Q))
            chk_v = float(next(v for v in chk.values()
                               if isinstance(v, (int, float))))
            tau = cavity_lifetime(float(Q), sigma)
            return np.array([
                math.log10(max(abs(g), 1e-30)),
                math.log10(max(abs(sat), 1e-30)),
                math.log10(max(abs(P), 1e-30)),
                chk_v,
                math.log10(max(abs(tau), 1e-30)),
            ])
        except Exception:
            return np.zeros(5)
    return run_v2(Qs, fn, "feedback_amplified_shell")


def m_horn_toroidal_warp():
    from systrophe.horn_toroidal_warp import (
        horn_steering_magnitude, horn_radius,
        horn_T_tt_profile, bm_NEC_radial, bm_shape,
    )
    eps_grid = np.linspace(0.0, 0.99, 25)
    def fn(eps):
        try:
            sm = horn_steering_magnitude(R=1.0, epsilon=float(eps),
                                          theta_0=0.0, m_ADM=-1.0, sigma=4.0)
            hr = horn_radius(theta=np.pi/4, R=1.0, epsilon=float(eps),
                              theta_0=0.0)
            tt = horn_T_tt_profile(theta=np.pi/4, R=1.0, epsilon=float(eps),
                                    theta_0=0.0, m_ADM=-1.0, sigma=4.0)
            nec = bm_NEC_radial(x=0.5, rho=1.0, m_ADM=1.0, R=1.0, sigma=4.0)
            shp = bm_shape(x=0.5, rho=1.0, R=1.0, sigma=4.0)
            return np.array([float(sm), float(hr), float(tt), float(nec), float(shp)])
        except Exception:
            return np.zeros(5)
    return run_v2(eps_grid, fn, "horn_toroidal_warp")


def m_krasnikov_pair():
    from systrophe.krasnikov_pair import (
        krasnikov_pair_NEC_radial, krasnikov_pair_total_negative_energy,
    )
    delta_grid = np.linspace(0.0, 2 * np.pi, 25)
    def fn(delta):
        try:
            nec_at = [krasnikov_pair_NEC_radial(x=float(x), t=1.0,
                                                delta=float(delta), alpha=4.0)
                      for x in (-1.0, 0.0, 1.0)]
            tne = krasnikov_pair_total_negative_energy(delta=float(delta))
            return np.array([*nec_at, float(tne)])
        except Exception:
            return np.zeros(4)
    return run_v2(delta_grid, fn, "krasnikov_pair")


def m_krasnikov_ring():
    from systrophe.krasnikov_ring import (
        krasnikov_ring_NEC_radial, krasnikov_ring_total_negative_energy,
    )
    eps_grid = np.linspace(0.0, np.pi/2, 25)
    def fn(eps):
        try:
            nec_at = [krasnikov_ring_NEC_radial(x=float(x), t=1.0,
                                                 epsilon=float(eps),
                                                 N=3, alpha=4.0)
                      for x in (-1.0, 0.0, 1.0)]
            tne = krasnikov_ring_total_negative_energy(N=3, epsilon=float(eps))
            return np.array([*nec_at, float(tne)])
        except Exception:
            return np.zeros(4)
    return run_v2(eps_grid, fn, "krasnikov_ring")


def m_nr_initial_data():
    from systrophe.nr_initial_data import (
        adm_profile_1d, bssn_decomposition,
        hamiltonian_constraint_residual, momentum_constraint_residual,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    omegas = np.linspace(0.5, 2.0, 20)
    def fn(omega):
        try:
            vs_i = VanStockumInterior(omega=float(omega), R=1.0)
            r_grid = np.linspace(1.05, 5.0, 40)
            adm = adm_profile_1d(vs_i, r_grid)
            bssn = bssn_decomposition(vs_i, r_grid)
            ham = hamiltonian_constraint_residual(vs_i, r_grid)
            mom = momentum_constraint_residual(vs_i, r_grid)
            vals = []
            for d in (adm, bssn, ham, mom):
                for v in d.values():
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                        break
                    elif isinstance(v, np.ndarray):
                        vals.append(float(np.mean(np.abs(v))))
                        break
            while len(vals) < 4: vals.append(0.0)
            return np.array(vals[:4])
        except Exception:
            return np.zeros(4)
    return run_v2(omegas, fn, "nr_initial_data")


def m_solitons_on_lp():
    from systrophe.solitons_on_lp import (
        kink_mass_on_lp, pair_extinction_soliton_decay,
        bound_state_spectrum_in_band,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r0_grid = np.linspace(1.2, 5.0, 25)
    def fn(r0):
        try:
            km = kink_mass_on_lp(vs, r_0=float(r0), width=0.5)
            pe = pair_extinction_soliton_decay(vs, r_0=float(r0), delta=1.0)
            pe_v = float(next(v for v in pe.values()
                              if isinstance(v, (int, float))))
            spec = bound_state_spectrum_in_band(vs,
                                                r_inner=float(r0) - 0.2,
                                                r_outer=float(r0) + 0.2,
                                                n_modes=3)
            spec_arr = np.array(spec[:3])
            if len(spec_arr) < 3:
                spec_arr = np.pad(spec_arr, (0, 3 - len(spec_arr)))
            return np.array([float(km), pe_v, *spec_arr])
        except Exception:
            return np.zeros(5)
    return run_v2(r0_grid, fn, "solitons_on_lp")


def m_tipler_fractal():
    from systrophe.tipler_fractal import (
        zero_geometric_ratio, dsi_rescaling_factor,
        zero_set, verify_geometric_progression,
    )
    alphas = np.linspace(1.05, 3.0, 25)
    def fn(alpha):
        try:
            zr = zero_geometric_ratio(float(alpha))
            ds = dsi_rescaling_factor(float(alpha))
            zs = zero_set(R=1.0, alpha=float(alpha), delta=0.1,
                           r_min=1.05, r_max=5.0)
            zs_first = float(zs[0]) if len(zs) > 0 else 0.0
            ver = verify_geometric_progression(zs)
            ver_v = float(next(v for v in ver.values()
                               if isinstance(v, (int, float))))
            return np.array([float(zr), float(ds), zs_first, float(len(zs)), ver_v])
        except Exception:
            return np.zeros(5)
    return run_v2(alphas, fn, "tipler_fractal")


def m_vacuum_states():
    from systrophe.vacuum_states import (
        boulware_energy_density_proxy, vacuum_selection_summary,
        compare_vacua_at_radius,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            bed = boulware_energy_density_proxy(vs, float(r))
            sel = vacuum_selection_summary(vs, float(r))
            cmp_v = compare_vacua_at_radius(vs, float(r))
            sel_vals = [float(getattr(sel, k))
                        for k in dir(sel)
                        if not k.startswith('_')
                        and isinstance(getattr(sel, k, None), (int, float))][:2]
            while len(sel_vals) < 2: sel_vals.append(0.0)
            cmp_v_v = float(next(v for v in cmp_v.values()
                                 if isinstance(v, (int, float))))
            return np.array([float(bed), *sel_vals, cmp_v_v])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "vacuum_states")


def m_floquet_mobius():
    from systrophe.floquet_mobius import (
        analyze_floquet_mobius, joint_floquet_spectrum,
        joint_static_hamiltonian, z3_symmetry_check,
    )
    drive_amps = np.linspace(0.0, 2.5, 25)
    branch_energies = np.array([-1.0, 0.0, 1.0])
    omega_drive = 2.0
    def fn(amp):
        try:
            H = joint_static_hamiltonian(branch_energies, hopping=0.3)
            spec = joint_floquet_spectrum(H, drive_amp=float(amp),
                                           omega_drive=omega_drive,
                                           n_steps=100)
            spec_arr = np.array(spec[:3])
            if len(spec_arr) < 3:
                spec_arr = np.pad(spec_arr, (0, 3 - len(spec_arr)))
            z3_chk = z3_symmetry_check(branch_energies, hopping=0.3,
                                        drive_amp=float(amp),
                                        omega_drive=omega_drive, n_steps=100)
            z3_v = float(next(v for v in z3_chk.values()
                              if isinstance(v, (int, float))))
            return np.array([*spec_arr, z3_v])
        except Exception:
            return np.zeros(4)
    return run_v2(drive_amps, fn, "floquet_mobius")


def main():
    runners = [
        m_adm_export, m_feedback_amplified_shell, m_horn_toroidal_warp,
        m_krasnikov_pair, m_krasnikov_ring, m_nr_initial_data,
        m_solitons_on_lp, m_tipler_fractal, m_vacuum_states,
        m_floquet_mobius,
    ]
    results = []
    print("Round 8 v2-catcher sweep across 10 more phase modules")
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

    out_path = Path(__file__).parent / "phase_modules_v2_round8_results.json"
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
