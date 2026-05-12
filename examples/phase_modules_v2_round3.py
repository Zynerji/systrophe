"""Round 3 v2-catcher sweep: 10 more phase modules without
novelty_scan() but with multi-component physics outputs."""

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


def m_tidal_forces():
    from systrophe.tidal_forces import (
        riemann_scalar_radial, tidal_stretch_at_radius,
        tidal_squeeze_at_radius, spaghettification_proxy,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 6.0, 30)
    def fn(r):
        try:
            return np.array([
                riemann_scalar_radial(vs, float(r)),
                tidal_stretch_at_radius(vs, float(r)),
                tidal_squeeze_at_radius(vs, float(r)),
                spaghettification_proxy(vs, float(r)),
            ])
        except Exception:
            return np.zeros(4)
    return run_v2(r_grid, fn, "tidal_forces")


def m_kk_embedding():
    from systrophe.kk_embedding import kk_tower_masses
    L_grid = np.linspace(0.3, 5.0, 25)
    def fn(L):
        try:
            return kk_tower_masses(float(L), n_max=10)
        except Exception:
            return np.zeros(10)
    return run_v2(L_grid, fn, "kk_embedding")


def m_anec_bound():
    from systrophe.anec_bound import (
        anec_integral_radial, anec_quantum_inequality_bound,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 6.0, 25)
    def fn(r):
        try:
            anec = anec_integral_radial(vs, float(r))
            qi = anec_quantum_inequality_bound(vs, float(r))
            return np.array([anec, qi, anec - qi])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "anec_bound")


def m_dynamical_casimir():
    from systrophe.dynamical_casimir import (
        cavity_mode_frequency, on_resonance_photon_number,
    )
    d_0 = 1.0
    modes = np.arange(1, 20).astype(float)
    def fn(n):
        try:
            omega = cavity_mode_frequency(int(n), d_0)
            n_photons = on_resonance_photon_number(
                omega_n=omega, eta=0.1, gamma=0.01, t=1.0
            )
            return np.array([omega, math.log10(max(n_photons, 1e-30)), float(n) / omega])
        except Exception:
            return np.zeros(3)
    return run_v2(modes, fn, "dynamical_casimir")


def m_cosmological_embedding():
    from systrophe.cosmological_embedding import (
        hubble_parameter, cosmological_corrections_to_alpha,
        horizon_chirp_rate,
    )
    a_grid = np.linspace(0.1, 5.0, 25)
    def fn(a):
        try:
            H = hubble_parameter(float(a))
            alpha_corr = cosmological_corrections_to_alpha(
                VanStockumInterior(omega=1.0, R=1.0), float(a)
            )
            chirp = horizon_chirp_rate(float(a))
            return np.array([H, alpha_corr, chirp])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(a_grid, fn, "cosmological_embedding")


def m_lqg_discretization():
    from systrophe.lqg_discretization import (
        lqg_area_spectrum_at_j, lqg_volume_spectrum_at_j,
    )
    js = np.linspace(0.5, 20.0, 25)
    def fn(j):
        try:
            return np.array([
                lqg_area_spectrum_at_j(float(j)),
                lqg_volume_spectrum_at_j(float(j), n_edges=3),
                lqg_volume_spectrum_at_j(float(j), n_edges=5),
            ])
        except Exception:
            return np.zeros(3)
    return run_v2(js, fn, "lqg_discretization")


def m_bdg_triple_vortex():
    try:
        from systrophe.bdg_triple_vortex import (
            triple_vortex_velocity, sonic_horizon_locus,
            hawking_temperature_at_horizon_2d,
        )
    except ImportError:
        return None
    r_grid = np.linspace(0.5, 5.0, 25)
    def fn(r):
        try:
            v = triple_vortex_velocity(float(r), Gamma=1.0)
            # Just return magnitudes of v components (likely 2D vector)
            if hasattr(v, "__len__"):
                return np.array([float(x) for x in v])
            return np.array([float(v)])
        except Exception:
            return np.array([0.0])
    return run_v2(r_grid, fn, "bdg_triple_vortex")


def m_penrose_extraction():
    from systrophe.penrose_extraction import penrose_efficiency_at
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 6.0, 25)
    def fn(r):
        try:
            d = penrose_efficiency_at(vs, float(r))
            keys = sorted(k for k, val in d.items()
                            if isinstance(val, (int, float)))
            return np.array([float(d[k]) for k in keys])
        except Exception:
            return np.array([0.0])
    return run_v2(r_grid, fn, "penrose_extraction")


def m_page_curve_ctc():
    from systrophe.page_curve_ctc import (
        enclosed_region_volume, entanglement_entropy_proxy,
        bekenstein_bound_estimate,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 4.0, 25)
    def fn(r):
        try:
            V = enclosed_region_volume(vs, vs.R, float(r))
            S = entanglement_entropy_proxy(vs, vs.R, float(r))
            B = bekenstein_bound_estimate(vs, vs.R, float(r))
            return np.array([
                math.log10(max(V, 1e-30)),
                S,
                math.log10(max(B, 1e-30)),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "page_curve_ctc")


def m_wormhole_throat():
    from systrophe.wormhole_throat import (
        effective_shape_function, effective_redshift_function,
    )
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return None
    r_grid = np.linspace(1.05, 6.0, 30)
    def fn(r):
        try:
            b = effective_shape_function(vs, float(r))
            phi = effective_redshift_function(vs, float(r))
            return np.array([b, phi, b - phi])
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    return run_v2(r_grid, fn, "wormhole_throat")


def main():
    runners = [
        m_tidal_forces, m_kk_embedding, m_anec_bound, m_dynamical_casimir,
        m_cosmological_embedding, m_lqg_discretization, m_bdg_triple_vortex,
        m_penrose_extraction, m_page_curve_ctc, m_wormhole_throat,
    ]
    results = []
    print("Round 3 v2-catcher sweep across 10 more phase modules")
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

    out_path = Path(__file__).parent / "phase_modules_v2_round3_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    # Summary, filtering edge artifacts (consensus at first grid point)
    print()
    real_emergents = []
    for r in results:
        if "error" in r:
            continue
        if r["anomaly"] == "novel_structure":
            real_emergents.append(f"{r['module']} (anomaly, p={r['anomaly_param']:.3f}, "
                                    f"delta={r['anomaly_max']:.2f})")
        if r["consensus"] == "novel_structure" and r["consensus_param"] > 0:
            real_emergents.append(f"{r['module']} (consensus, p={r['consensus_param']:.3f})")
    print(f"FLAGS: {len(real_emergents)}")
    for e in real_emergents:
        print(f"  - {e}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
