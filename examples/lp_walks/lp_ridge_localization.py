"""Localize the LP-cluster ridge by running the same fine-grain
radial grid against every module that fired in rounds 1-8.

The hypothesis: emergents do not just happen to lie in r in [1.5, 2.55];
they pile up at SPECIFIC radii (~1.54, ~1.87, ~2.00, ~2.55) where
multiple independent observables flip simultaneously.

Method:
  1. Fine r grid (dr = 0.01) over [1.40, 2.60] = 121 points.
  2. For each of ~20 LP-radial modules, evaluate its multi-component
     observable vector at every r.
  3. Run the v2 coherence-anomaly detector on each module's trajectory.
  4. Collect every sharp jump across all modules into a single histogram
     binned at dr = 0.02.
  5. Peaks in that histogram = the ridge.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.catchers.catcher_v2 import scan_novelty_coherence_anomaly
from systrophe.geometry.vanstockum import VanStockumInterior


VS = VanStockumInterior(omega=1.0, R=1.0)
assert VS.is_supercritical(), "VS must be supercritical"


# ---------- module observable functions ----------

def f_one_loop_backreaction(r):
    from systrophe.qftcs.one_loop_backreaction import (
        anomaly_radial_pattern,
    )
    try:
        d = anomaly_radial_pattern(VS, r_min=float(r) - 0.005,
                                    r_max=float(r) + 0.005, n_grid=3)
        return np.array([float(v[1]) if isinstance(v, np.ndarray) and len(v) > 1
                          else (float(v) if isinstance(v, (int, float)) else 0.0)
                          for v in list(d.values())[:4]])
    except Exception:
        return np.zeros(4)


def f_tidal_forces(r):
    from systrophe.geometry.tidal_forces import (
        riemann_scalar_radial, tidal_stretch_at_radius,
        tidal_squeeze_at_radius, spaghettification_proxy,
    )
    try:
        return np.array([
            float(riemann_scalar_radial(VS, float(r))),
            float(tidal_stretch_at_radius(VS, float(r))),
            float(tidal_squeeze_at_radius(VS, float(r))),
            float(spaghettification_proxy(VS, float(r))),
        ])
    except Exception:
        return np.zeros(4)


def f_aharonov_bohm(r):
    try:
        from systrophe.foundations.aharonov_bohm_ctc import (
            aharonov_bohm_phase_around_loop,
        )
    except Exception:
        return np.zeros(3)
    try:
        ph = aharonov_bohm_phase_around_loop(VS, float(r))
        if isinstance(ph, dict):
            return np.array([float(v) for v in ph.values()
                              if isinstance(v, (int, float))][:3] + [0.0] * 3)[:3]
        return np.array([float(ph), 0.0, 0.0])
    except Exception:
        return np.zeros(3)


def f_frame_dragging(r):
    from systrophe.geometry.frame_dragging import (
        lense_thirring_frequency, gyroscope_precession_angle,
        total_drag_per_revolution,
    )
    try:
        return np.array([
            float(lense_thirring_frequency(VS, float(r))),
            float(gyroscope_precession_angle(VS, float(r), time=1.0)),
            float(total_drag_per_revolution(VS, float(r))),
        ])
    except Exception:
        return np.zeros(3)


def f_cauchy_stability(r):
    try:
        from systrophe.ctc.cauchy_stability import cauchy_perturbation_growth
    except Exception:
        return np.zeros(3)
    try:
        growth = cauchy_perturbation_growth(VS, float(r))
        if hasattr(growth, "__len__"):
            arr = np.array(growth[:3])
            if len(arr) < 3:
                arr = np.pad(arr, (0, 3 - len(arr)))
            return arr
        return np.array([float(growth), 0, 0])
    except Exception:
        return np.zeros(3)


def f_unruh_effect(r):
    try:
        from systrophe.qftcs.unruh_effect import unruh_temperature_at_radius
        T = unruh_temperature_at_radius(VS, float(r))
        if isinstance(T, dict):
            return np.array([float(v) for v in T.values()
                              if isinstance(v, (int, float))][:3] + [0.0] * 3)[:3]
        return np.array([float(T), 0.0, 0.0])
    except Exception:
        return np.zeros(3)


def f_wormhole_throat(r):
    from systrophe.geometry.wormhole_throat import (
        effective_shape_function, effective_redshift_function,
    )
    try:
        b = effective_shape_function(VS, float(r))
        phi = effective_redshift_function(VS, float(r))
        return np.array([float(b), float(phi), float(b - phi)])
    except Exception:
        return np.zeros(3)


def f_kg_scattering(r):
    from systrophe.qftcs.kg_scattering import (
        effective_potential, transmission_coefficient,
        reflection_coefficient,
    )
    omega = 1.0
    try:
        V = effective_potential(VS, float(r), omega=omega, n_phi=0, mass=0.0)
        T = transmission_coefficient(VS, max(float(r) - 0.3, 1.05),
                                      float(r) + 0.3, omega=omega,
                                      n_phi=0, mass=0.0, n_steps=200)
        R = reflection_coefficient(VS, max(float(r) - 0.3, 1.05),
                                    float(r) + 0.3, omega=omega,
                                    n_phi=0, mass=0.0)
        return np.array([float(V), float(T), float(R)])
    except Exception:
        return np.zeros(3)


def f_lensing_image(r):
    from systrophe.geometry.lensing_image import (
        impact_parameter_bare, null_circular_omega,
    )
    try:
        b_pro = impact_parameter_bare(VS, float(r), branch="prograde")
        b_ret = impact_parameter_bare(VS, float(r), branch="retrograde")
        F = VS.analytic_exterior_F(np.array([float(r)]))[0]
        K = VS.analytic_exterior_K(np.array([float(r)]))[0]
        L = VS.analytic_exterior_L(np.array([float(r)]))[0]
        om_p, om_m = null_circular_omega(F, K, L, float(r))
        return np.array([
            float(b_pro) if np.isfinite(b_pro) else 0.0,
            float(b_ret) if np.isfinite(b_ret) else 0.0,
            float(om_p), float(om_m),
        ])
    except Exception:
        return np.zeros(4)


def f_acoustic_metric(r):
    from systrophe.analogs.acoustic_metric import acoustic_line_element_at_radius
    try:
        d = acoustic_line_element_at_radius(VS, float(r))
        keys = sorted(k for k, v in d.items() if isinstance(v, (int, float)))
        return np.array([float(d[k]) for k in keys])
    except Exception:
        return np.array([0.0])


def f_quantum_diagnostics(r):
    from systrophe.qftcs.quantum_diagnostics import (
        tolman_blueshift_factor, ricci_scalar, conformal_anomaly_2d_proxy,
        chronology_protection_indicator,
    )
    try:
        F = float(VS.analytic_exterior_F(np.array([float(r)]))[0])
        tol = float(tolman_blueshift_factor(np.array([F]))[0])
        ric = float(ricci_scalar(VS, float(r)))
        conf = float(conformal_anomaly_2d_proxy(VS, float(r)))
        chron = float(chronology_protection_indicator(np.array([F]))[0])
        return np.array([tol, ric, conf, chron])
    except Exception:
        return np.zeros(4)


def f_berry_phase_lp(r):
    try:
        from systrophe.lp.berry_phase_lp import berry_phase_around_loop_at_radius
        ph = berry_phase_around_loop_at_radius(VS, float(r))
        if isinstance(ph, dict):
            return np.array([float(v) for v in ph.values()
                              if isinstance(v, (int, float))][:3] + [0.0] * 3)[:3]
        return np.array([float(ph), 0.0, 0.0])
    except Exception:
        return np.zeros(3)


def f_photon_orbits(r):
    from systrophe.geometry.photon_orbits import vanstockum_photon_omega
    try:
        op, om = vanstockum_photon_omega(VS, float(r))
        return np.array([float(op), float(om), float(op + om), float(op - om)])
    except Exception:
        return np.zeros(4)


def f_dirac_sea(r):
    from systrophe.qftcs.dirac_sea import (
        local_energy, dirac_sea_pressure_proxy, density_of_states_radial,
    )
    E_infty = 1.0
    try:
        F = float(VS.analytic_exterior_F(np.array([float(r)]))[0])
        E_local = float(local_energy(np.array([F]), E_infty)[0])
        P = float(dirac_sea_pressure_proxy(VS, float(r)))
        dos = float(density_of_states_radial(F, m=0, k=1.0, mass=0.5,
                                              E_infty=E_infty))
        return np.array([E_local, P, dos])
    except Exception:
        return np.zeros(3)


def f_hadamard_offtrace(r):
    from systrophe.qftcs.hadamard_offtrace import (
        energy_density_in_static_frame, kretschmann_scalar,
        trace_decomposition,
    )
    try:
        ed = energy_density_in_static_frame(VS, float(r))
        kret = kretschmann_scalar(VS, float(r))
        td = trace_decomposition(VS, float(r))
        keys = sorted(k for k, v in td.items() if isinstance(v, (int, float)))
        td_vals = [float(td[k]) for k in keys[:2]]
        return np.array([
            math.log10(max(abs(ed), 1e-30)),
            math.log10(max(abs(kret), 1e-30)),
            *td_vals,
        ])
    except Exception:
        return np.zeros(4)


def f_spinor_monodromy(r):
    from systrophe.quantum_info.spinor_monodromy import (
        expected_monodromy_phase_per_revolution,
        monodromy_period_in_revolutions,
        monodromy_eigenvalues, spin_connection_phi,
    )
    try:
        phase = expected_monodromy_phase_per_revolution(VS, float(r))
        period = monodromy_period_in_revolutions(VS, float(r))
        eigs = monodromy_eigenvalues(VS, float(r))
        spin = spin_connection_phi(VS, float(r))
        spin_vals = [float(v) for v in spin.values()
                     if isinstance(v, (int, float))][:1]
        if not spin_vals: spin_vals = [0.0]
        eig0 = float(np.real(eigs[0])) if len(eigs) > 0 else 0.0
        return np.array([float(phase), float(period), eig0, *spin_vals])
    except Exception:
        return np.zeros(4)


def f_stochastic_lp(r):
    from systrophe.lp.stochastic_lp import (
        diffusion_coefficient_estimate,
        escape_probability_in_finite_time,
    )
    try:
        D = diffusion_coefficient_estimate(VS, float(r))
        esc = escape_probability_in_finite_time(VS, float(r), T_window=1.0)
        return np.array([float(D), float(esc), 0.0])
    except Exception:
        return np.zeros(3)


def f_qftcs_backreaction(r):
    from systrophe.qftcs.qftcs_backreaction import (
        conformal_anomaly_trace, radial_temporal_ricci_scalar,
        vacuum_energy_density_proxy,
    )
    try:
        return np.array([
            float(conformal_anomaly_trace(VS, float(r))),
            float(radial_temporal_ricci_scalar(VS, float(r))),
            float(vacuum_energy_density_proxy(VS, float(r))),
        ])
    except Exception:
        return np.zeros(3)


def f_casimir_polder(r):
    from systrophe.qftcs.casimir_polder import (
        casimir_polder_force_static, casimir_polder_with_rotation,
        frame_drag_correction,
    )
    try:
        static_F = casimir_polder_force_static(float(r), polarizability=1.0)
        rot = casimir_polder_with_rotation(VS, float(r), polarizability=1.0)
        keys = sorted(k for k, v in rot.items() if isinstance(v, (int, float)))
        rot_vals = [float(rot[k]) for k in keys[:3]]
        drag = frame_drag_correction(VS, float(r))
        return np.array([float(static_F), *rot_vals, float(drag)])
    except Exception:
        return np.zeros(5)


def f_qi_channel(r):
    from systrophe.quantum_info.qi_channel import (
        redshift_factor, channel_capacity_holevo, amplitude_damping_p,
    )
    r_A = 1.05
    try:
        rs = redshift_factor(VS, r_A, float(r))
        p = amplitude_damping_p(float(rs))
        cap = channel_capacity_holevo(float(p))
        return np.array([float(rs), float(p), float(cap)])
    except Exception:
        return np.zeros(3)


def f_vacuum_states(r):
    from systrophe.qftcs.vacuum_states import (
        boulware_energy_density_proxy, vacuum_selection_summary,
        compare_vacua_at_radius,
    )
    try:
        bed = boulware_energy_density_proxy(VS, float(r))
        sel = vacuum_selection_summary(VS, float(r))
        cmp_v = compare_vacua_at_radius(VS, float(r))
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


def f_solitons_on_lp(r):
    from systrophe.lp.solitons_on_lp import (
        kink_mass_on_lp, bound_state_spectrum_in_band,
    )
    try:
        km = kink_mass_on_lp(VS, r_0=float(r), width=0.5)
        spec = bound_state_spectrum_in_band(VS,
                                            r_inner=float(r) - 0.2,
                                            r_outer=float(r) + 0.2,
                                            n_modes=3)
        spec_arr = np.array(spec[:3])
        if len(spec_arr) < 3:
            spec_arr = np.pad(spec_arr, (0, 3 - len(spec_arr)))
        return np.array([float(km), *spec_arr])
    except Exception:
        return np.zeros(4)


MODULES = {
    "one_loop_backreaction": f_one_loop_backreaction,
    "tidal_forces": f_tidal_forces,
    "aharonov_bohm": f_aharonov_bohm,
    "frame_dragging": f_frame_dragging,
    "cauchy_stability": f_cauchy_stability,
    "unruh_effect": f_unruh_effect,
    "wormhole_throat": f_wormhole_throat,
    "kg_scattering": f_kg_scattering,
    "lensing_image": f_lensing_image,
    "acoustic_metric": f_acoustic_metric,
    "quantum_diagnostics": f_quantum_diagnostics,
    "berry_phase_lp": f_berry_phase_lp,
    "photon_orbits": f_photon_orbits,
    "dirac_sea": f_dirac_sea,
    "hadamard_offtrace": f_hadamard_offtrace,
    "spinor_monodromy": f_spinor_monodromy,
    "stochastic_lp": f_stochastic_lp,
    "qftcs_backreaction": f_qftcs_backreaction,
    "casimir_polder": f_casimir_polder,
    "qi_channel": f_qi_channel,
    "vacuum_states": f_vacuum_states,
    "solitons_on_lp": f_solitons_on_lp,
}


def main():
    r_grid = np.linspace(1.40, 2.60, 121)  # dr = 0.01
    print(f"LP-ridge localization: {len(r_grid)} radii (dr={r_grid[1]-r_grid[0]:.3f}), "
          f"{len(MODULES)} modules")
    print("=" * 70)

    # For each module, run coherence anomaly with low jump threshold
    per_module = {}
    for name, fn in MODULES.items():
        try:
            res = scan_novelty_coherence_anomaly(
                r_grid, fn,
                sharp_jump_threshold=0.15,  # lower threshold: fine grid
                min_components=2,
            )
            per_module[name] = {
                "verdict": res.verdict,
                "max_delta": float(res.max_delta_coherence),
                "max_at": float(res.max_delta_coherence_parameter),
                "sharps": [
                    {"r": float(s["parameter_value"]),
                     "delta": float(s["delta_coherence"])}
                    for s in res.sharp_features
                ],
            }
            n_sharps = len(per_module[name]["sharps"])
            print(f"{name:30s}: {res.verdict:18s} max_delta={res.max_delta_coherence:.3f} "
                  f"@ r={res.max_delta_coherence_parameter:.3f}, n_sharp={n_sharps}")
        except Exception as e:
            per_module[name] = {"error": str(e)}
            print(f"{name:30s}: ERROR - {e}")

    # Aggregate: histogram of sharp-jump r-values across all modules
    print()
    print("Ridge histogram (sharp jumps across all modules)")
    print("-" * 70)
    bin_centers = np.arange(1.40, 2.61, 0.02)
    hist = np.zeros(len(bin_centers))
    modules_per_bin = [[] for _ in bin_centers]
    for name, m in per_module.items():
        if "error" in m: continue
        for s in m["sharps"]:
            r = s["r"]
            bi = int(round((r - 1.40) / 0.02))
            if 0 <= bi < len(bin_centers):
                hist[bi] += 1
                modules_per_bin[bi].append(name)

    # Dedupe by unique modules per bin (kg_scattering and quantum_diagnostics
    # fire repeatedly across components; we care about distinct observables).
    unique_per_bin = [sorted(set(mods)) for mods in modules_per_bin]
    unique_counts = np.array([len(m) for m in unique_per_bin])

    order = np.argsort(-unique_counts)
    print("\nTop 12 ridge peaks (deduped by unique modules, bin r ± 0.01):")
    print(f"{'r':>6}  {'n_obs':>5}  {'evt':>4}  modules")
    for i in order[:12]:
        if unique_counts[i] < 1: continue
        print(f"{bin_centers[i]:6.3f}  {int(unique_counts[i]):>5}  "
              f"{int(hist[i]):>4}  {', '.join(unique_per_bin[i])}")

    out_path = Path(__file__).parent / "lp_ridge_localization_results.json"
    out_path.write_text(json.dumps({
        "r_grid": r_grid.tolist(),
        "per_module": per_module,
        "bin_centers": bin_centers.tolist(),
        "hist": hist.tolist(),
        "modules_per_bin": modules_per_bin,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")

    n_modules_fired = sum(1 for m in per_module.values()
                          if "error" not in m and m["verdict"] == "novel_structure")
    n_total = sum(1 for m in per_module.values() if "error" not in m)
    print(f"\n{n_modules_fired}/{n_total} modules fired (verdict=novel_structure).")
    print(f"Total sharp jumps detected: {int(hist.sum())}")


if __name__ == "__main__":
    main()
