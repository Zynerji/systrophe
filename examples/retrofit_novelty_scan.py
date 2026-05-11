"""Retrofit pass: run the novelty catcher on every Systrophe phase
that doesn't yet have its own novelty_scan() entry-point.

Per the always-on rule, every module's natural parameter scan must
be inspected for sharp features. This script does that ad-hoc for
phases 1-34 (the modules that predate the catcher).
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import scan_novelty
from systrophe.vanstockum import VanStockumInterior

RESULTS_DIR = Path(__file__).parent
OUT_PATH = RESULTS_DIR / "retrofit_novelty_results.json"

vs = VanStockumInterior(omega=1.0, R=1.0)
R_GRID = np.linspace(1.05 * vs.R, 10 * vs.R, 30)


def safe_scan(name: str, fn, params=None) -> dict:
    """Run a novelty scan, capture errors."""
    if params is None:
        params = R_GRID
    try:
        result = scan_novelty(params, fn, n_bits=32)
        return {
            "module": name,
            "verdict": result.verdict,
            "n_sharp": len(result.sharp_features),
            "sharp": [
                {k: (v if not isinstance(v, np.integer) else int(v))
                 for k, v in s.items()}
                for s in result.sharp_features
            ],
        }
    except Exception as e:
        return {
            "module": name,
            "error": f"{type(e).__name__}: {e}",
        }


# --- Phase 11: Cauchy stability ---
def scan_cauchy_stability():
    from systrophe.cauchy_stability import lyapunov_exponent_at_horizon
    def fn(r):
        return np.array([float(lyapunov_exponent_at_horizon(vs, float(r)))])
    return safe_scan("cauchy_stability", fn)


# --- Phase 12: Frame dragging ---
def scan_frame_dragging():
    from systrophe.frame_dragging import lense_thirring_frequency
    def fn(r):
        return np.array([float(lense_thirring_frequency(vs, float(r)))])
    return safe_scan("frame_dragging", fn)


# --- Phase 13: Optical fiber analog ---
def scan_optical_fiber_analog():
    from systrophe.optical_fiber_analog import fiber_analog_horizon
    def fn(v_pump):
        return np.array([float(fiber_analog_horizon(c_probe=1.0, v_pump=v_pump))])
    return safe_scan("optical_fiber_analog", fn, params=np.linspace(0.5, 1.5, 30))


# --- Phase 14: Energy condition survey ---
def scan_energy_condition_survey():
    from systrophe.energy_condition_survey import systematic_energy_survey
    def fn(omega):
        res = systematic_energy_survey(omega=float(omega), R=1.0,
                                          n_samples=10)
        return np.array([
            float(res.get("WEC_violations", 0)),
            float(res.get("NEC_violations", 0)),
            float(res.get("DEC_violations", 0)),
            float(res.get("SEC_violations", 0)),
        ])
    return safe_scan("energy_condition_survey", fn,
                       params=np.linspace(0.3, 2.0, 30))


# --- Phase 16: Holographic boundary correlator ---
def scan_holographic():
    from systrophe.holographic import boundary_two_point_correlator
    def fn(s):
        c = boundary_two_point_correlator(vs, separation=float(s))
        return np.array([float(c.real), float(c.imag), float(abs(c))])
    return safe_scan("holographic", fn, params=np.geomspace(1.0, 1000.0, 30))


# --- Phase 17: QI channel ---
def scan_qi_channel():
    from systrophe.qi_channel import channel_capacity_holevo
    def fn(p):
        return np.array([float(channel_capacity_holevo(float(p)))])
    return safe_scan("qi_channel", fn, params=np.linspace(0.0, 1.0, 30))


# --- Phase 19: spinor monodromy ---
def scan_spinor_monodromy():
    from systrophe.spinor_monodromy import expected_monodromy_phase_per_revolution
    def fn(r):
        return np.array([float(expected_monodromy_phase_per_revolution(vs, float(r)))])
    return safe_scan("spinor_monodromy", fn)


# --- Phase 20: synchrotron analog ---
def scan_synchrotron_analog():
    from systrophe.synchrotron_analog import (
        orbital_frequency, effective_gamma_factor,
        synchrotron_critical_frequency,
    )
    def fn(r):
        Om = orbital_frequency(vs, float(r))
        g = effective_gamma_factor(vs, float(r))
        om_c = synchrotron_critical_frequency(vs, float(r))
        return np.array([
            Om if math.isfinite(Om) else 0.0,
            g if math.isfinite(g) else 0.0,
            om_c if math.isfinite(om_c) else 0.0,
        ])
    return safe_scan("synchrotron_analog", fn)


# --- Phase 21: tidal forces ---
def scan_tidal_forces():
    from systrophe.tidal_forces import riemann_scalar_radial
    def fn(r):
        return np.array([float(riemann_scalar_radial(vs, float(r)))])
    return safe_scan("tidal_forces", fn)


# --- Phase 22: KG scattering ---
def scan_kg_scattering():
    from systrophe.kg_scattering import effective_potential
    def fn(r):
        V = effective_potential(vs, float(r), omega=1.0)
        return np.array([float(V) if math.isfinite(V) else 0.0])
    return safe_scan("kg_scattering", fn)


# --- Phase 24: DM scalar coupling ---
def scan_dm_scalar_coupling():
    from systrophe.dm_scalar_coupling import DM_density_profile_around_cylinder
    def fn(r):
        prof = DM_density_profile_around_cylinder(vs, np.array([float(r)]))
        return np.array([float(prof["rho_DM"][0])])
    return safe_scan("dm_scalar_coupling", fn,
                       params=np.linspace(0.01, 100.0, 30))


# --- Phase 26: GW emission ---
def scan_gw_emission():
    from systrophe.gw_emission import radial_oscillation_strain
    def fn(omega_R):
        res = radial_oscillation_strain(vs, omega_R=float(omega_R))
        return np.array([float(res["h_peak"]), float(res["f_GW"])])
    return safe_scan("gw_emission", fn, params=np.linspace(0.1, 10.0, 30))


# --- Phase 27: ANEC bound ---
def scan_anec_bound():
    from systrophe.anec_bound import T_kk_radial
    def fn(r):
        v = T_kk_radial(vs, float(r))
        return np.array([float(v) if math.isfinite(v) else 0.0])
    return safe_scan("anec_bound", fn)


# --- Phase 28: Page curve CTC ---
def scan_page_curve_ctc():
    from systrophe.page_curve_ctc import entanglement_entropy_proxy
    def fn(t):
        return np.array([float(entanglement_entropy_proxy(vs, 1.83, 11.23, float(t)))])
    return safe_scan("page_curve_ctc", fn, params=np.linspace(0.0, 200.0, 30))


# --- Phase 29: LQG discretization ---
def scan_lqg_discretization():
    from systrophe.lqg_discretization import discretization_error_relative
    def fn(r):
        q = discretization_error_relative(vs, float(r))
        return np.array([float(q.relative_error)])
    return safe_scan("lqg_discretization", fn)


# --- Phase 30: CTC tunneling ---
def scan_ctc_tunneling():
    from systrophe.ctc_tunneling import tunneling_action
    def fn(r_outer):
        return np.array([float(tunneling_action(vs, r_inner=1.83, r_outer=float(r_outer)))])
    return safe_scan("ctc_tunneling", fn, params=np.linspace(2.0, 11.0, 30))


def main():
    all_scans = [
        scan_cauchy_stability,
        scan_frame_dragging,
        scan_optical_fiber_analog,
        scan_energy_condition_survey,
        scan_holographic,
        scan_qi_channel,
        scan_spinor_monodromy,
        scan_synchrotron_analog,
        scan_tidal_forces,
        scan_kg_scattering,
        scan_dm_scalar_coupling,
        scan_gw_emission,
        scan_anec_bound,
        scan_page_curve_ctc,
        scan_lqg_discretization,
        scan_ctc_tunneling,
    ]
    results = []
    for scanfn in all_scans:
        result = scanfn()
        results.append(result)
        name = result["module"]
        if "error" in result:
            print(f"  {name:<35s} ERROR: {result['error'][:50]}")
        else:
            print(f"  {name:<35s} {result['verdict']:<20s} sharp={result['n_sharp']}")

    # Summary
    print()
    print("=" * 70)
    print("Retrofit novelty scan summary")
    print("=" * 70)
    novel = [r for r in results if r.get("verdict") == "novel_structure"]
    print(f"Modules with NOVEL_STRUCTURE: {len(novel)}/{len(results)}")
    for r in novel:
        print(f"  - {r['module']} ({r['n_sharp']} sharp features)")
        for sf in r["sharp"][:3]:
            print(f"    p={sf.get('parameter_value', '?')}  step={sf.get('hamming_step', '?')}  median={sf.get('median_step', '?')}")

    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
