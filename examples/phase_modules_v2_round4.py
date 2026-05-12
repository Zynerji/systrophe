"""Round 4 v2-catcher sweep: 10 more phase modules with diverse
physics (D-CTC, tunneling, acoustic, floquet, etc.)."""

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
        "consensus": cons.verdict, "consensus_max": float(cons.max_consensus),
        "consensus_param": float(cons.max_consensus_parameter),
        "consensus_sharps": [
            {"p": float(s["parameter_value"]),
              "z": float(s.get("max_z", 0)),
              "frac": float(s["consensus_fraction"])}
            for s in cons.sharp_features[:3]
        ],
        "anomaly": anom.verdict, "anomaly_max": float(anom.max_delta_coherence),
        "anomaly_param": float(anom.max_delta_coherence_parameter),
        "anomaly_sharps": [
            {"p": float(s["parameter_value"]),
              "delta": float(s["delta_coherence"]),
              "before": float(s["coherence_before"]),
              "after": float(s["coherence_after"])}
            for s in anom.sharp_features[:3]
        ],
    }


def m_chronology_protection():
    try:
        from systrophe.chronology_protection import chronology_protection_study
    except ImportError:
        return None
    a_grid = np.linspace(0.3, 2.0, 25)
    def fn(a):
        try:
            vs = VanStockumInterior(omega=float(a), R=1.0)
            study = chronology_protection_study(vs)
            # Extract numeric fields
            keys = sorted([k for k, v in study.__dict__.items()
                            if isinstance(v, (int, float))])
            return np.array([float(getattr(study, k)) for k in keys])
        except Exception:
            return np.array([0.0])
    return run_v2(a_grid, fn, "chronology_protection")


def m_ctc_tunneling():
    from systrophe.ctc_tunneling import (
        tunneling_action, attempt_frequency, tunneling_rate,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            S = tunneling_action(vs, float(r))
            nu = attempt_frequency(vs, float(r))
            rate = tunneling_rate(vs, float(r))
            return np.array([
                math.log10(max(abs(S), 1e-30)),
                math.log10(max(nu, 1e-30)),
                math.log10(max(rate, 1e-30)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "ctc_tunneling")


def m_particle_creation():
    from systrophe.particle_creation import (
        bose_einstein, fermi_dirac,
    )
    T = 1.0
    omegas = np.geomspace(0.1, 100.0, 25)
    def fn(omega):
        try:
            return np.array([
                float(bose_einstein(np.array([float(omega)]), T)[0]),
                float(fermi_dirac(np.array([float(omega)]), T)[0]),
                float(omega) / T,
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(omegas, fn, "particle_creation")


def m_topology_change():
    from systrophe.topology_change import (
        ctc_band_pinch_off_action, ctc_band_merger_action,
        topology_change_probability,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 4.0, 25)
    def fn(r):
        try:
            S_pinch = ctc_band_pinch_off_action(vs, float(r))
            S_merge = ctc_band_merger_action(vs, float(r))
            P_pinch = topology_change_probability(S_pinch)
            P_merge = topology_change_probability(S_merge)
            return np.array([S_pinch, S_merge, P_pinch, P_merge])
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "topology_change")


def m_acoustic_hawking_spectrum():
    from systrophe.acoustic_hawking_spectrum import (
        hawking_planck_spectrum, total_emission_power,
    )
    Ts = np.geomspace(0.01, 10.0, 25)
    def fn(T):
        omegas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        try:
            return hawking_planck_spectrum(omegas, float(T))
        except Exception:
            return np.zeros(5)
    return run_v2(Ts, fn, "acoustic_hawking_spectrum")


def m_d_ctc():
    try:
        from systrophe.d_ctc import (
            fixed_point_iter, kraus_operators_from_unitary,
        )
    except ImportError:
        return None
    # Use random Kraus channels at different dim_CR sizes; deterministic
    dim_cr = 4
    dim_ctc_grid = np.arange(2, 12).astype(float)
    rng_seed = 42
    def fn(dim_ctc_float):
        dim_ctc = int(round(dim_ctc_float))
        try:
            rng = np.random.default_rng(rng_seed)
            U = rng.normal(0, 1, (dim_cr * dim_ctc, dim_cr * dim_ctc))
            U, _ = np.linalg.qr(U)
            ks = kraus_operators_from_unitary(U, dim_cr, dim_ctc)
            rho_init = np.eye(dim_ctc) / dim_ctc
            rho_fixed, iters = fixed_point_iter(ks, rho_init, max_iter=200, tol=1e-9)
            return np.array([
                iters,
                float(np.real(np.trace(rho_fixed @ rho_fixed))),
                float(np.linalg.norm(rho_fixed - np.eye(dim_ctc) / dim_ctc)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(dim_ctc_grid, fn, "d_ctc")


def m_floquet():
    try:
        from systrophe.floquet import (
            stroboscopic_floquet_spectrum,
        )
    except ImportError:
        return None
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    drives = np.linspace(0.0, 2.0, 25)
    def fn(drive):
        try:
            spec = stroboscopic_floquet_spectrum(vs, drive_amplitude=float(drive))
            if hasattr(spec, "__len__"):
                arr = np.array(spec[:5])
                if len(arr) < 5:
                    arr = np.pad(arr, (0, 5 - len(arr)))
                return arr
            return np.array([float(spec), 0, 0, 0, 0])
        except Exception:
            return np.zeros(5)
    return run_v2(drives, fn, "floquet")


def m_back_reaction():
    try:
        from systrophe.back_reaction import (
            backreaction_correction_to_F,
        )
    except ImportError:
        return None
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 4.0, 25)
    def fn(r):
        try:
            dF = backreaction_correction_to_F(vs, float(r))
            F = vs.analytic_exterior_F(np.array([float(r)]))[0]
            return np.array([dF, F, dF / F if abs(F) > 1e-12 else 0.0])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "back_reaction")


def m_cauchy_stability():
    try:
        from systrophe.cauchy_stability import (
            cauchy_perturbation_growth,
        )
    except ImportError:
        return None
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 4.0, 30)
    def fn(r):
        try:
            growth = cauchy_perturbation_growth(vs, float(r))
            if hasattr(growth, "__len__"):
                arr = np.array(growth[:3])
                if len(arr) < 3:
                    arr = np.pad(arr, (0, 3 - len(arr)))
                return arr
            return np.array([float(growth), 0, 0])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "cauchy_stability")


def m_photon_orbits():
    from systrophe.photon_orbits import vanstockum_photon_omega
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            omega_plus, omega_minus = vanstockum_photon_omega(vs, float(r))
            return np.array([float(omega_plus), float(omega_minus),
                             float(omega_plus + omega_minus),
                             float(omega_plus - omega_minus)])
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "photon_orbits")


def main():
    runners = [
        m_chronology_protection, m_ctc_tunneling, m_particle_creation,
        m_topology_change, m_acoustic_hawking_spectrum, m_d_ctc,
        m_floquet, m_back_reaction, m_cauchy_stability, m_photon_orbits,
    ]
    results = []
    print("Round 4 v2-catcher sweep across 10 more phase modules")
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
        print(f"  consensus: {res['consensus']} (max={res['consensus_max']:.3f})")
        for s in res.get("consensus_sharps", [])[:2]:
            print(f"    sharp @ p={s['p']:.3f}: z={s['z']:.2f}, frac={s['frac']:.2f}")
        print(f"  anomaly:   {res['anomaly']} (max_delta={res['anomaly_max']:.3f})")
        for s in res.get("anomaly_sharps", [])[:2]:
            print(f"    sharp @ p={s['p']:.3f}: delta={s['delta']:.3f} "
                  f"({s['before']:.2f}->{s['after']:.2f})")

    out_path = Path(__file__).parent / "phase_modules_v2_round4_results.json"
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
