"""Wide v2-catcher sweep across 14 Systrophe phase modules with
multi-component novelty_scan APIs.

For each module, extract the function fn(parameter) -> vector output,
run the parameter sweep, and apply the v2 detectors:
  - coherence (saturated for highly-correlated physics)
  - coherence-anomaly (sharp jumps in the coherence trajectory)
  - consensus (per-component z-score above baseline)

Report only modules that produce NOVEL_STRUCTURE in coherence-anomaly
or consensus (since base coherence saturates trivially).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.catcher_v2 import (
    scan_novelty_coherent,
    scan_novelty_consensus,
    scan_novelty_coherence_anomaly,
)


def run_v2(parameter_axis, fn, label):
    coh = scan_novelty_coherent(parameter_axis, fn,
                                  sharp_threshold=0.20, min_components=2)
    cons = scan_novelty_consensus(parameter_axis, fn,
                                    z_threshold=2.5, consensus_fraction=0.30)
    anom = scan_novelty_coherence_anomaly(parameter_axis, fn,
                                            sharp_jump_threshold=0.25,
                                            min_components=2)
    return {
        "module": label,
        "coherence": coh.verdict,
        "coherence_max": float(coh.max_coherence),
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


# === Module-specific data extractors ===

def m_aharonov_bohm():
    from systrophe.aharonov_bohm_ctc import aharonov_bohm_phase, enclosed_flux
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        phase = aharonov_bohm_phase(vs, float(r))
        flux = enclosed_flux(vs, float(r))
        return np.array([math.cos(phase), math.sin(phase), flux])
    return run_v2(r_grid, fn, "aharonov_bohm")


def m_anyonic_ctc():
    from systrophe.anyonic_ctc import braid_phase_at_band
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    alpha = vs.alpha
    bands = np.arange(1, 30).astype(float)
    def fn(b):
        theta = braid_phase_at_band(int(round(b)), alpha)
        return np.array([math.cos(theta), math.sin(theta), theta])
    return run_v2(bands, fn, "anyonic_ctc")


def m_berry_phase():
    from systrophe.berry_phase_lp import (
        berry_phase_per_revolution, berry_connection, berry_curvature,
    )
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 8.0, 35)
    def fn(r):
        try:
            gamma = berry_phase_per_revolution(vs, float(r))
            A = berry_connection(vs, float(r))
            F = berry_curvature(vs, float(r))
            return np.array([
                gamma if math.isfinite(gamma) else 1e6,
                A if math.isfinite(A) else 1e6,
                F if math.isfinite(F) else 1e6,
            ])
        except Exception:
            return np.array([1e6, 1e6, 1e6])
    return run_v2(r_grid, fn, "berry_phase_lp")


def m_bh_pair_production():
    from systrophe.bh_pair_production import schwinger_analog_rate
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 10.0, 30)
    def fn(r):
        try:
            rate = schwinger_analog_rate(vs, float(r))
            return np.array([
                math.log10(max(rate.production_rate, 1e-300)),
                math.log10(max(rate.field_strength, 1e-30)),
                rate.bh_mass_threshold,
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "bh_pair_production")


def m_er_epr_pair():
    from systrophe.er_epr_pair import (
        bridge_strength, bell_pair_fidelity_at_delta,
        mutual_information_proxy, entanglement_entropy_pair,
    )
    deltas = np.linspace(0.0, math.pi, 30)
    def fn(d):
        try:
            return np.array([
                bridge_strength(float(d)),
                bell_pair_fidelity_at_delta(float(d)),
                mutual_information_proxy(float(d)),
                entanglement_entropy_pair(float(d)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0])
    return run_v2(deltas, fn, "er_epr_pair")


def m_holographic_complexity():
    from systrophe.holographic_complexity import volume_proxy_C, action_proxy_C
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_base = 1.5
    widths = np.linspace(0.1, 3.0, 25)
    def fn(w):
        try:
            CV = volume_proxy_C(vs, r_base, r_base + float(w))
            CA = action_proxy_C(vs, r_base, r_base + float(w))
            return np.array([
                CV if math.isfinite(CV) else 1e6,
                CA if math.isfinite(CA) else 1e6,
            ])
        except Exception:
            return np.array([1e6, 1e6])
    return run_v2(widths, fn, "holographic_complexity")


def m_lp_dualities():
    from systrophe.lp_dualities import s_dual_a
    a_grid = np.linspace(0.3, 2.5, 30)
    def fn(a):
        a = float(a)
        try:
            a_S = s_dual_a(a)
            alpha = math.sqrt(4 * a ** 2 - 1) if a > 0.5 else 0.0
            return np.array([a, a_S, alpha])
        except Exception:
            return np.array([a, 0, 0])
    return run_v2(a_grid, fn, "lp_dualities")


def m_monopole_on_cylinder():
    from systrophe.monopole_on_cylinder import dirac_quantization_condition
    from systrophe.vanstockum import VanStockumInterior
    a_grid = np.linspace(0.3, 2.0, 25)
    def fn(a):
        try:
            vs = VanStockumInterior(omega=float(a), R=1.0)
            charges = dirac_quantization_condition(vs, n_max=5)
            return np.array(charges)
        except Exception:
            return np.zeros(5)
    return run_v2(a_grid, fn, "monopole_on_cylinder")


def m_one_loop_backreaction():
    from systrophe.one_loop_backreaction import (
        one_loop_F_correction, corrected_F, trace_anomaly_at_r,
    )
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            return np.array([
                one_loop_F_correction(vs, float(r)),
                corrected_F(vs, float(r)),
                trace_anomaly_at_r(vs, float(r)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "one_loop_backreaction")


def m_twistor_lp():
    from systrophe.twistor_lp import alpha_plane_at_r, twistor_inner_product
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_ref = 1.5
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            Z = alpha_plane_at_r(vs, float(r))
            ip = twistor_inner_product(vs, r_ref, float(r))
            return np.array([
                Z.norm_squared,
                float(abs(Z.spinor_omega) ** 2),
                float(abs(Z.spinor_pi) ** 2),
                float(abs(ip)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "twistor_lp")


def m_unruh_effect():
    from systrophe.unruh_effect import combined_unruh_hawking_T
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 5.0, 30)
    def fn(r):
        try:
            res = combined_unruh_hawking_T(vs, float(r))
            return np.array([
                min(res["T_Unruh"], 1e6),
                res["T_Hawking"],
                min(res["T_combined"], 1e6),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "unruh_effect")


def m_horn_toroidal():
    try:
        from systrophe.horn_toroidal_warp import (
            steering_dipole_moment, horn_curvature_extrinsic,
            adm_dipole_asymmetry,
        )
    except ImportError:
        return None
    epsilons = np.linspace(0.0, 0.95, 25)
    def fn(eps):
        try:
            return np.array([
                steering_dipole_moment(epsilon=float(eps)),
                horn_curvature_extrinsic(epsilon=float(eps)),
                adm_dipole_asymmetry(epsilon=float(eps)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(epsilons, fn, "horn_toroidal_warp")


def m_alcubierre():
    try:
        from systrophe.alcubierre import (
            warp_factor, exotic_matter_density, drive_power,
        )
    except ImportError:
        return None
    v_grid = np.linspace(0.1, 5.0, 25)
    def fn(v):
        try:
            return np.array([
                warp_factor(float(v), R=1.0, sigma=1.0),
                exotic_matter_density(float(v), R=1.0, sigma=1.0),
                drive_power(float(v), R=1.0, sigma=1.0),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(v_grid, fn, "alcubierre")


def m_lentz_soliton():
    try:
        from systrophe.lentz_soliton import (
            lentz_soliton_energy, lentz_soliton_velocity,
        )
    except ImportError:
        return None
    R_grid = np.linspace(0.5, 5.0, 25)
    def fn(R):
        try:
            E = lentz_soliton_energy(R=float(R))
            v = lentz_soliton_velocity(R=float(R))
            return np.array([
                E if math.isfinite(E) else 1e6,
                v if math.isfinite(v) else 0.0,
            ])
        except Exception:
            return np.array([1e6, 0.0])
    return run_v2(R_grid, fn, "lentz_soliton")


def main():
    runners = [
        m_aharonov_bohm, m_anyonic_ctc, m_berry_phase, m_bh_pair_production,
        m_er_epr_pair, m_holographic_complexity, m_lp_dualities,
        m_monopole_on_cylinder, m_one_loop_backreaction, m_twistor_lp,
        m_unruh_effect, m_horn_toroidal, m_alcubierre, m_lentz_soliton,
    ]
    results = []
    print("Wide v2-catcher sweep across 14 phase modules")
    print("=" * 70)
    for r in runners:
        try:
            res = r()
        except Exception as e:
            res = {"module": r.__name__, "error": str(e)}
        if res is None:
            print(f"\n{r.__name__}: SKIPPED (subcritical/unavailable)")
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

    out_path = Path(__file__).parent / "phase_modules_v2_wide_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    # Summary: new emergents
    new_emergents = []
    for r in results:
        if "error" in r:
            continue
        if r["consensus"] == "novel_structure":
            new_emergents.append(f"{r['module']} (consensus, p={r['consensus_param']:.3f})")
        if r["anomaly"] == "novel_structure":
            new_emergents.append(f"{r['module']} (anomaly, p={r['anomaly_param']:.3f}, "
                                   f"delta={r['anomaly_max']:.2f})")
    print()
    print(f"NEW EMERGENTS: {len(new_emergents)} of {len(results)} modules tested")
    for e in new_emergents:
        print(f"  - {e}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
