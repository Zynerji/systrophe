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

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior

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
    from systrophe.ctc.cauchy_stability import lyapunov_exponent_at_horizon
    def fn(r):
        return np.array([float(lyapunov_exponent_at_horizon(vs, float(r)))])
    return safe_scan("cauchy_stability", fn)


# --- Phase 12: Frame dragging ---
def scan_frame_dragging():
    from systrophe.geometry.frame_dragging import lense_thirring_frequency
    def fn(r):
        return np.array([float(lense_thirring_frequency(vs, float(r)))])
    return safe_scan("frame_dragging", fn)


# --- Phase 13: Optical fiber analog ---
def scan_optical_fiber_analog():
    from systrophe.analogs.optical_fiber_analog import (
        fiber_analog_horizon, linear_pump_profile,
    )
    def fn(v_end):
        prof = linear_pump_profile(v_start=0.4, v_end=float(v_end),
                                     x_start=0.0, x_end=10.0)
        res = fiber_analog_horizon(prof, probe_index=1.5, c=1.0)
        horizons = res.get("horizons", [])
        return np.array([
            float(len(horizons)),
            float(horizons[0] if horizons else -1.0),
        ])
    return safe_scan("optical_fiber_analog", fn,
                       params=np.linspace(0.5, 1.2, 30))


# --- Phase 14: Energy condition survey ---
def scan_energy_condition_survey():
    from systrophe.qftcs.energy_condition_survey import (
        systematic_energy_survey, summary_statistics,
    )
    def fn(omega_center):
        # Narrow window around omega_center
        oc = float(omega_center)
        res = systematic_energy_survey(
            omega_range=(max(oc - 0.1, 0.05), oc + 0.1),
            R_range=(0.5, 2.0), n_omega=3, n_R=3, r_test=0.5,
        )
        stats = summary_statistics(res)
        return np.array([
            float(stats.get("WEC_violation_fraction", 0.0)),
            float(stats.get("NEC_violation_fraction", 0.0)),
            float(stats.get("DEC_violation_fraction", 0.0)),
            float(stats.get("SEC_violation_fraction", 0.0)),
        ])
    return safe_scan("energy_condition_survey", fn,
                       params=np.linspace(0.3, 2.0, 30))


# --- Phase 16: Holographic boundary correlator ---
def scan_holographic():
    from systrophe.quantum_info.holographic import boundary_two_point_correlator
    def fn(s):
        c = boundary_two_point_correlator(vs, separation=float(s))
        return np.array([float(c.real), float(c.imag), float(abs(c))])
    return safe_scan("holographic", fn, params=np.geomspace(1.0, 1000.0, 30))


# --- Phase 17: QI channel ---
def scan_qi_channel():
    from systrophe.quantum_info.qi_channel import channel_capacity_holevo
    def fn(p):
        return np.array([float(channel_capacity_holevo(float(p)))])
    return safe_scan("qi_channel", fn, params=np.linspace(0.0, 1.0, 30))


# --- Phase 19: spinor monodromy ---
def scan_spinor_monodromy():
    from systrophe.quantum_info.spinor_monodromy import expected_monodromy_phase_per_revolution
    def fn(r):
        return np.array([float(expected_monodromy_phase_per_revolution(vs, float(r)))])
    return safe_scan("spinor_monodromy", fn)


# --- Phase 20: synchrotron analog ---
def scan_synchrotron_analog():
    from systrophe.analogs.synchrotron_analog import (
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
    from systrophe.geometry.tidal_forces import riemann_scalar_radial
    def fn(r):
        return np.array([float(riemann_scalar_radial(vs, float(r)))])
    return safe_scan("tidal_forces", fn)


# --- Phase 22: KG scattering ---
def scan_kg_scattering():
    from systrophe.qftcs.kg_scattering import effective_potential
    def fn(r):
        V = effective_potential(vs, float(r), omega=1.0)
        return np.array([float(V) if math.isfinite(V) else 0.0])
    return safe_scan("kg_scattering", fn)


# --- Phase 24: DM scalar coupling ---
def scan_dm_scalar_coupling():
    from systrophe.qftcs.dm_scalar_coupling import DM_density_profile_around_cylinder
    def fn(r):
        prof = DM_density_profile_around_cylinder(vs, np.array([float(r)]))
        return np.array([float(prof["rho_DM"][0])])
    return safe_scan("dm_scalar_coupling", fn,
                       params=np.linspace(0.01, 100.0, 30))


# --- Phase 26: GW emission ---
def scan_gw_emission():
    from systrophe.qftcs.gw_emission import radial_oscillation_strain
    def fn(omega_R):
        res = radial_oscillation_strain(vs, omega_R=float(omega_R))
        return np.array([float(res["h_peak"]), float(res["f_GW"])])
    return safe_scan("gw_emission", fn, params=np.linspace(0.1, 10.0, 30))


# --- Phase 27: ANEC bound ---
def scan_anec_bound():
    from systrophe.qftcs.anec_bound import T_kk_radial
    def fn(r):
        v = T_kk_radial(vs, float(r))
        return np.array([float(v) if math.isfinite(v) else 0.0])
    return safe_scan("anec_bound", fn)


# --- Phase 28: Page curve CTC ---
def scan_page_curve_ctc():
    from systrophe.ctc.page_curve_ctc import entanglement_entropy_proxy
    def fn(t):
        return np.array([float(entanglement_entropy_proxy(vs, 1.83, 11.23, float(t)))])
    return safe_scan("page_curve_ctc", fn, params=np.linspace(0.0, 200.0, 30))


# --- Phase 29: LQG discretization ---
def scan_lqg_discretization():
    from systrophe.quantum_info.lqg_discretization import discretization_error_relative
    def fn(r):
        q = discretization_error_relative(vs, float(r))
        return np.array([float(q.relative_error)])
    return safe_scan("lqg_discretization", fn)


# --- Phase 30: CTC tunneling ---
def scan_ctc_tunneling():
    from systrophe.ctc.ctc_tunneling import tunneling_action
    def fn(r_outer):
        return np.array([float(tunneling_action(vs, r_inner=1.83, r_outer=float(r_outer)))])
    return safe_scan("ctc_tunneling", fn, params=np.linspace(2.0, 11.0, 30))


# --- Phase 18 (new): photon sphere structure ---
def scan_photon_sphere():
    from systrophe.geometry.photon_sphere import impact_parameter_bare
    def fn(r):
        try:
            b = impact_parameter_bare(vs, float(r), branch="prograde")
        except Exception:
            b = 0.0
        return np.array([float(b) if math.isfinite(b) else 0.0])
    return safe_scan("photon_sphere", fn, params=np.linspace(1.05, 10.0, 30))


# --- Phase 25 (new): chronology protection budget ---
def scan_chronology_protection():
    from systrophe.ctc.chronology_protection import (
        chronology_protection_study, chronology_protection_verdict,
    )
    from systrophe.geometry.sinusoid import TiplerSinusoid
    def fn(delta):
        s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
        s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=float(delta))
        r_samples = np.linspace(1.05, 8.0, 20)
        study = chronology_protection_study(s1=s1, s2_base=s2, r_samples=r_samples)
        verdict = chronology_protection_verdict(study)
        return np.array([
            float(verdict.get("max_residual", 0.0)),
            float(verdict.get("mean_residual", 0.0)),
            float(1.0 if verdict.get("violated", False) else 0.0),
        ])
    return safe_scan("chronology_protection", fn,
                       params=np.linspace(0.0, 2 * math.pi, 30))


# --- Phase 36/38 (new): Aharonov-Bohm CTC phase ---
def scan_aharonov_bohm_ctc():
    from systrophe.foundations.aharonov_bohm_ctc import aharonov_bohm_phase
    def fn(r):
        try:
            phi = aharonov_bohm_phase(vs, float(r))
        except Exception:
            phi = 0.0
        return np.array([float(phi) if math.isfinite(phi) else 0.0])
    return safe_scan("aharonov_bohm_ctc", fn, params=np.linspace(1.05, 12.0, 30))


# --- Phase 49 (new): Berry phase on LP ---
def scan_berry_phase_lp():
    from systrophe.lp.berry_phase_lp import berry_phase_per_revolution
    def fn(r):
        try:
            ph = berry_phase_per_revolution(vs, float(r))
        except Exception:
            ph = 0.0
        return np.array([float(ph) if math.isfinite(ph) else 0.0])
    return safe_scan("berry_phase_lp", fn, params=np.linspace(1.05, 12.0, 30))


# --- Phase 48 (new): twistor on LP ---
def scan_twistor_lp():
    from systrophe.lp.twistor_lp import twistor_norm
    def fn(r):
        try:
            n = twistor_norm(vs, float(r))
        except Exception:
            n = 0.0
        return np.array([float(n) if math.isfinite(n) else 0.0])
    return safe_scan("twistor_lp", fn, params=np.linspace(1.05, 12.0, 30))


# --- Phase 47 (new): vacuum polarization ---
def scan_vacuum_polarization():
    from systrophe.qftcs.vacuum_polarization import vacuum_polarization_at_r
    def fn(r):
        try:
            res = vacuum_polarization_at_r(vs, float(r))
            v = res.get("scalar_one_loop", 0.0) if isinstance(res, dict) else float(res)
        except Exception:
            v = 0.0
        return np.array([float(v) if math.isfinite(v) else 0.0])
    return safe_scan("vacuum_polarization", fn,
                       params=np.linspace(1.05, 12.0, 30))


# --- Phase 50 (new): holographic complexity ---
def scan_holographic_complexity():
    from systrophe.quantum_info.holographic_complexity import complexity_growth_rate
    def fn(r):
        try:
            c = complexity_growth_rate(vs, float(r))
        except Exception:
            c = 0.0
        return np.array([float(c) if math.isfinite(c) else 0.0])
    return safe_scan("holographic_complexity", fn,
                       params=np.linspace(1.05, 12.0, 30))


# --- Phase 40 (new): anyonic CTC braid ---
def scan_anyonic_ctc():
    from systrophe.ctc.anyonic_ctc import braid_phase_at_band
    def fn(band_index):
        try:
            ph = braid_phase_at_band(int(band_index), alpha=float(vs.alpha))
        except Exception:
            ph = 0.0
        return np.array([float(ph) if math.isfinite(ph) else 0.0])
    # band indices 1..30
    return safe_scan("anyonic_ctc", fn,
                       params=np.arange(1, 31, dtype=float))


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
        # Wave 2 (added 2026-05-11):
        scan_photon_sphere,
        scan_chronology_protection,
        scan_aharonov_bohm_ctc,
        scan_berry_phase_lp,
        scan_twistor_lp,
        scan_vacuum_polarization,
        scan_holographic_complexity,
        scan_anyonic_ctc,
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
