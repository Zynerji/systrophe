"""Round 6 v2-catcher sweep: 10 more modules. Several deliberately
scan over LP r in [1.05, 5.0] to probe the dense cluster band."""

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


def m_casimir_throat():
    from systrophe.casimir_throat import (
        brown_maclay_at_lp_point, kretschmann_scalar,
        topological_throat_coefficient,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    d_eff = 0.1
    def fn(r):
        try:
            d_bm = brown_maclay_at_lp_point(vs, float(r), d_eff)
            keys = sorted(k for k, v in d_bm.items()
                          if isinstance(v, (int, float)))
            vals = [float(d_bm[k]) for k in keys[:3]]
            kret = kretschmann_scalar(vs, float(r))
            return np.array([*vals, math.log10(max(abs(kret), 1e-30))])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "casimir_throat")


def m_casimir_polder():
    from systrophe.casimir_polder import (
        casimir_polder_force_static, casimir_polder_with_rotation,
        frame_drag_correction,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            static_F = casimir_polder_force_static(float(r), polarizability=1.0)
            rot = casimir_polder_with_rotation(vs, float(r), polarizability=1.0)
            keys = sorted(k for k, v in rot.items()
                          if isinstance(v, (int, float)))
            rot_vals = [float(rot[k]) for k in keys[:3]]
            drag = frame_drag_correction(vs, float(r))
            return np.array([float(static_F), *rot_vals, float(drag)])
        except Exception:
            return np.zeros(5)
    return run_v2(r_grid, fn, "casimir_polder")


def m_dirac_sea():
    from systrophe.dirac_sea import (
        local_energy, dirac_sea_pressure_proxy, density_of_states_radial,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    E_infty = 1.0
    def fn(r):
        try:
            F = float(vs.analytic_exterior_F(np.array([float(r)]))[0])
            E_local = float(local_energy(np.array([F]), E_infty)[0])
            P = float(dirac_sea_pressure_proxy(vs, float(r)))
            dos = float(density_of_states_radial(F, m=0, k=1.0, mass=0.5,
                                                  E_infty=E_infty))
            return np.array([E_local, P, dos])
        except Exception:
            return np.zeros(3)
    return run_v2(r_grid, fn, "dirac_sea")


def m_dm_scalar_coupling():
    from systrophe.dm_scalar_coupling import (
        DM_drag_on_orbits, DM_induced_CTC_lifetime,
        DM_superradiance_growth_rate,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            drag = DM_drag_on_orbits(vs, float(r))
            ctc = DM_induced_CTC_lifetime(vs)
            sr = DM_superradiance_growth_rate(vs)
            vals = []
            for d in (drag, ctc, sr):
                for v in d.values():
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                        if len(vals) >= 4: break
                if len(vals) >= 4: break
            while len(vals) < 4: vals.append(0.0)
            return np.array(vals[:4])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "dm_scalar_coupling")


def m_hadamard_offtrace():
    from systrophe.hadamard_offtrace import (
        energy_density_in_static_frame, kretschmann_scalar,
        trace_decomposition,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            ed = energy_density_in_static_frame(vs, float(r))
            kret = kretschmann_scalar(vs, float(r))
            td = trace_decomposition(vs, float(r))
            keys = sorted(k for k, v in td.items() if isinstance(v, (int, float)))
            td_vals = [float(td[k]) for k in keys[:2]]
            return np.array([
                math.log10(max(abs(ed), 1e-30)),
                math.log10(max(abs(kret), 1e-30)),
                *td_vals,
            ])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "hadamard_offtrace")


def m_point_splitting():
    from systrophe.point_splitting import (
        phi_squared_largemass_expansion, dewitt_a2_coefficient,
        trace_anomaly_4d_exact, vacuum_residual,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            phi2 = phi_squared_largemass_expansion(vs, float(r), mass=1.0)
            a2 = dewitt_a2_coefficient(vs, float(r))
            ta = trace_anomaly_4d_exact(vs, float(r))
            vr = vacuum_residual(vs, float(r))
            return np.array([float(phi2), float(a2), float(ta), float(vr)])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "point_splitting")


def m_spinor_monodromy():
    from systrophe.spinor_monodromy import (
        expected_monodromy_phase_per_revolution,
        monodromy_period_in_revolutions,
        monodromy_eigenvalues,
        spin_connection_phi,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            phase = expected_monodromy_phase_per_revolution(vs, float(r))
            period = monodromy_period_in_revolutions(vs, float(r))
            eigs = monodromy_eigenvalues(vs, float(r))
            spin = spin_connection_phi(vs, float(r))
            spin_vals = [float(v) for v in spin.values()
                         if isinstance(v, (int, float))][:1]
            if not spin_vals: spin_vals = [0.0]
            eig0 = float(np.real(eigs[0])) if len(eigs) > 0 else 0.0
            return np.array([float(phase), float(period), eig0, *spin_vals])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "spinor_monodromy")


def m_stochastic_lp():
    from systrophe.stochastic_lp import (
        diffusion_coefficient_estimate,
        escape_probability_in_finite_time,
        mean_first_passage_to_CH,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 25)
    def fn(r):
        try:
            D = diffusion_coefficient_estimate(vs, float(r))
            esc = escape_probability_in_finite_time(vs, float(r), T_window=1.0)
            mfp = mean_first_passage_to_CH(vs, float(r))
            mfp_vals = [float(getattr(mfp, k))
                        for k in dir(mfp)
                        if not k.startswith('_')
                        and isinstance(getattr(mfp, k, None), (int, float))][:2]
            while len(mfp_vals) < 2: mfp_vals.append(0.0)
            return np.array([float(D), float(esc), *mfp_vals])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "stochastic_lp")


def m_qftcs_backreaction():
    from systrophe.qftcs_backreaction import (
        conformal_anomaly_trace, radial_temporal_ricci_scalar,
        vacuum_energy_density_proxy,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            ca = float(conformal_anomaly_trace(vs, float(r)))
            ric = float(radial_temporal_ricci_scalar(vs, float(r)))
            ved = float(vacuum_energy_density_proxy(vs, float(r)))
            return np.array([ca, ric, ved])
        except Exception:
            return np.zeros(3)
    return run_v2(r_grid, fn, "qftcs_backreaction")


def m_synchrotron_analog():
    from systrophe.synchrotron_analog import (
        orbital_frequency, synchrotron_critical_frequency,
        emitted_power, effective_gamma_factor, back_reaction_drag,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            of = orbital_frequency(vs, float(r))
            sc = synchrotron_critical_frequency(vs, float(r))
            ep = emitted_power(vs, float(r))
            gf = effective_gamma_factor(vs, float(r))
            dr = back_reaction_drag(vs, float(r))
            return np.array([float(of), float(sc),
                             math.log10(max(abs(ep), 1e-30)),
                             float(gf), float(dr)])
        except Exception:
            return np.zeros(5)
    return run_v2(r_grid, fn, "synchrotron_analog")


def main():
    runners = [
        m_casimir_throat, m_casimir_polder, m_dirac_sea, m_dm_scalar_coupling,
        m_hadamard_offtrace, m_point_splitting, m_spinor_monodromy,
        m_stochastic_lp, m_qftcs_backreaction, m_synchrotron_analog,
    ]
    results = []
    print("Round 6 v2-catcher sweep across 10 more phase modules")
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

    out_path = Path(__file__).parent / "phase_modules_v2_round6_results.json"
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
