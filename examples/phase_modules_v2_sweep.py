"""Apply catcher v2 (coherence + consensus + multi-run) to 5
representative Systrophe phase modules with multi-component outputs.

Goal: find emergents the v1 catcher missed.

Target modules:
  1. bms_soft_hair: supertranslation mode amplitudes vs n_modes
  2. energy_condition_survey: 4 energy conditions (WEC/NEC/SEC/DEC)
     across (omega, R) parameter grid
  3. dirac_spectrum: bound-state energies + amplitudes vs alpha
  4. modified_dispersion: dispersion at multiple momentum scales
  5. vacuum_polarization: HE shift + alpha running + pair threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.catcher_v2 import (
    scan_novelty_coherent,
    scan_novelty_consensus,
    scan_novelty_coherence_anomaly,
)


def run_v2_on_vector_scan(parameter_axis, output_fn,
                            label: str,
                            coherence_threshold: float = 0.20,
                            consensus_z: float = 2.5,
                            consensus_fraction: float = 0.30) -> dict:
    """Run coherence + consensus on a parametric vector-output scan."""
    coh = scan_novelty_coherent(
        parameter_axis, output_fn,
        sharp_threshold=coherence_threshold,
        min_components=2,
    )
    cons = scan_novelty_consensus(
        parameter_axis, output_fn,
        z_threshold=consensus_z,
        consensus_fraction=consensus_fraction,
    )
    anomaly = scan_novelty_coherence_anomaly(
        parameter_axis, output_fn,
        sharp_jump_threshold=0.30,
        min_components=2,
    )
    return {
        "module": label,
        "coherence": {
            "verdict": coh.verdict,
            "max_coherence": coh.max_coherence,
            "max_coherence_parameter": coh.max_coherence_parameter,
            "n_sharp": len(coh.sharp_features),
        },
        "consensus": {
            "verdict": cons.verdict,
            "max_consensus": cons.max_consensus,
            "max_consensus_parameter": cons.max_consensus_parameter,
            "n_sharp": len(cons.sharp_features),
        },
        "coherence_anomaly": {
            "verdict": anomaly.verdict,
            "max_delta_coherence": anomaly.max_delta_coherence,
            "max_delta_coherence_parameter": anomaly.max_delta_coherence_parameter,
            "n_sharp": len(anomaly.sharp_features),
            "sharps": [
                {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                 for k, v in s.items()}
                for s in anomaly.sharp_features[:5]
            ],
        },
    }


# -------- 1. BMS soft hair --------
def sweep_bms_soft_hair():
    from systrophe.bms_soft_hair import supertranslation_mode_amplitudes
    from systrophe.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return {"module": "bms_soft_hair", "skipped": "subcritical"}
    n_max = 30

    def fn(n_float):
        n = max(int(round(n_float)), 1)
        modes = supertranslation_mode_amplitudes(vs, n)
        amps = np.array([abs(m.amplitude) for m in modes])
        # Pad / truncate to fixed length 30 components for v2 detectors
        if len(amps) < n_max:
            amps = np.pad(amps, (0, n_max - len(amps)))
        return amps[:n_max]

    params = np.linspace(2, n_max, 29).astype(float)
    return run_v2_on_vector_scan(params, fn, "bms_soft_hair")


# -------- 2. Energy condition survey across (omega, R) --------
def sweep_energy_conditions():
    from systrophe.vanstockum import VanStockumInterior
    from systrophe.energy_conditions import energy_condition_report
    # Sweep omega at fixed R=1.0; output: (WEC, NEC, SEC, DEC) as 0/1
    omegas = np.linspace(0.3, 2.0, 20)

    def fn(omega):
        try:
            vs = VanStockumInterior(omega=float(omega), R=1.0)
            r_test = 0.5
            report = energy_condition_report(vs, r=r_test)
            return np.array([
                float(report.weak),
                float(report.null),
                float(report.strong),
                float(report.dominant),
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0, 0.0])

    return run_v2_on_vector_scan(omegas, fn, "energy_conditions_omega_sweep")


# -------- 3. Dirac spectrum --------
def sweep_dirac_spectrum():
    from systrophe.vanstockum import VanStockumInterior
    try:
        from systrophe.dirac_spectrum import find_bound_states
    except ImportError:
        return {"module": "dirac_spectrum", "skipped": "import_failed"}
    # Sweep omega (LP supercritical regime); output: bound-state energies
    omegas = np.linspace(0.6, 1.5, 18)
    n_states = 8

    def fn(omega):
        try:
            vs = VanStockumInterior(omega=float(omega), R=1.0)
            states = find_bound_states(vs, n_target=n_states)
            energies = np.array([s.energy for s in states][:n_states])
            if len(energies) < n_states:
                energies = np.pad(energies, (0, n_states - len(energies)))
            return energies
        except Exception:
            return np.zeros(n_states)

    return run_v2_on_vector_scan(omegas, fn, "dirac_spectrum_omega_sweep")


# -------- 4. Modified dispersion --------
def sweep_modified_dispersion():
    from systrophe.vanstockum import VanStockumInterior
    try:
        from systrophe.modified_dispersion import modified_speed_of_light
    except ImportError:
        return {"module": "modified_dispersion", "skipped": "import_failed"}
    # Sweep radius; output: speed at multiple momenta
    vs = VanStockumInterior(omega=1.2, R=1.0)
    rs = np.linspace(1.05, 5.0, 20)
    momenta = np.geomspace(0.01, 10.0, 10)

    def fn(r):
        try:
            v = np.array([
                modified_speed_of_light(vs, r=float(r), k=float(k))
                for k in momenta
            ])
            return v
        except Exception:
            return np.zeros(len(momenta))

    return run_v2_on_vector_scan(rs, fn, "modified_dispersion_radial_sweep")


# -------- 5. Vacuum polarization --------
def sweep_vacuum_polarization():
    from systrophe.vanstockum import VanStockumInterior
    try:
        from systrophe.vacuum_polarization import (
            heisenberg_euler_shift, effective_fine_structure_running,
            vacuum_polarization_at_r,
        )
    except ImportError:
        return {"module": "vacuum_polarization", "skipped": "import_failed"}
    vs = VanStockumInterior(omega=1.2, R=1.0)
    rs = np.linspace(1.05, 5.0, 25)

    def fn(r):
        try:
            v = vacuum_polarization_at_r(vs, r=float(r))
            # v is dict; extract numeric fields
            keys = sorted([k for k, val in v.items()
                            if isinstance(val, (int, float))])
            return np.array([float(v[k]) for k in keys])
        except Exception:
            return np.array([0.0])

    return run_v2_on_vector_scan(rs, fn, "vacuum_polarization_radial_sweep")


def main():
    sweepers = [
        sweep_bms_soft_hair,
        sweep_energy_conditions,
        sweep_dirac_spectrum,
        sweep_modified_dispersion,
        sweep_vacuum_polarization,
    ]
    out = []
    print("Catcher v2 sweep across 5 Systrophe phase modules")
    print("=" * 70)
    for sweeper in sweepers:
        try:
            r = sweeper()
        except Exception as e:
            r = {"module": sweeper.__name__, "error": str(e)}
        out.append(r)
        if "error" in r:
            print(f"\n{r['module']}: ERROR -- {r['error']}")
            continue
        if "skipped" in r:
            print(f"\n{r['module']}: SKIPPED -- {r['skipped']}")
            continue
        print(f"\n{r['module']}:")
        print(f"  coherence: verdict={r['coherence']['verdict']}, "
              f"max_coh={r['coherence']['max_coherence']:.3f}")
        print(f"  consensus: verdict={r['consensus']['verdict']}, "
              f"max_cons={r['consensus']['max_consensus']:.3f}")
        a = r['coherence_anomaly']
        print(f"  coherence_anomaly: verdict={a['verdict']}, "
              f"max_delta={a['max_delta_coherence']:.3f} at "
              f"p={a['max_delta_coherence_parameter']:.3f}, "
              f"n_sharp={a['n_sharp']}")
        for s in a.get('sharps', [])[:3]:
            print(f"    sharp @ p={s['parameter_value']:.3f}: "
                  f"d_coh={s['delta_coherence']:.3f} "
                  f"({s['coherence_before']:.2f}->{s['coherence_after']:.2f})")

    out_path = Path(__file__).parent / "phase_modules_v2_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Summary: how many new emergents via coherence-ANOMALY (the real
    # detection statistic for deterministic physics modules)?
    new_emergents = []
    for r in out:
        if "error" in r or "skipped" in r:
            continue
        if r["coherence_anomaly"]["verdict"] == "novel_structure":
            p_val = r["coherence_anomaly"]["max_delta_coherence_parameter"]
            d = r["coherence_anomaly"]["max_delta_coherence"]
            new_emergents.append(f"{r['module']} (coherence_anomaly, "
                                   f"p={p_val:.3f}, delta={d:.3f})")
        if r["consensus"]["verdict"] == "novel_structure":
            new_emergents.append(f"{r['module']} (consensus, "
                                   f"p={r['consensus']['max_consensus_parameter']:.3f})")
    print()
    print(f"NEW EMERGENTS via v2 catchers: {len(new_emergents)}")
    for e in new_emergents:
        print(f"  - {e}")


if __name__ == "__main__":
    main()
