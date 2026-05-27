"""Cluster investigation of the r=1.976 structural feature.

Catcher emergents #11 (kg_scattering) and #12 (photon_sphere) both
flag a sharp transition at the SAME radius r = 1.976 in the
supercritical Lewis-Papapetrou exterior (omega=1, R=1). This script
probes many additional diagnostics at a fine grid around that radius
to see how many independent physics measures converge on the same
structural feature.

Hypothesis: r=1.976 is a previously-undescribed structural locus of
the supercritical LP geometry where multiple physical observables
have a coincident sharp transition. Independent confirmation across
N>=4 diagnostics would harden the finding into a real shortcut /
publishable physics.

Probes attempted (best-effort; missing modules are skipped):
    kg_scattering.effective_potential
    photon_sphere.impact_parameter_bare
    spinor_monodromy.expected_monodromy_phase_per_revolution
    aharonov_bohm_ctc.aharonov_bohm_phase
    berry_phase_lp.berry_phase_per_revolution
    tidal_forces.riemann_scalar_radial
    frame_dragging.lense_thirring_frequency
    holographic.boundary_two_point_correlator
    qftcs_backreaction.<curvature scalar>
    quantum_diagnostics.ricci_scalar
    cauchy_stability.lyapunov_exponent_at_horizon
    vacuum_polarization.vacuum_polarization_at_r
    holographic_complexity.complexity_growth_rate
    twistor_lp.twistor_norm
    geodesic.<orbit invariant>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.catchers.novelty_catcher import catch_novelty_in_named_arrays
from systrophe.geometry.vanstockum import VanStockumInterior

vs = VanStockumInterior(omega=1.0, R=1.0)

# Fine grid around r=1.976
R_FINE = np.linspace(1.85, 2.10, 80)


def probe_kg_scattering() -> np.ndarray | None:
    try:
        from systrophe.qftcs.kg_scattering import effective_potential
        return np.array([float(effective_potential(vs, float(r), omega=1.0))
                          for r in R_FINE])
    except Exception:
        return None


def probe_photon_sphere() -> np.ndarray | None:
    try:
        from systrophe.geometry.photon_sphere import impact_parameter_bare
        out = []
        for r in R_FINE:
            try:
                b = float(impact_parameter_bare(vs, float(r), branch="prograde"))
                out.append(b if math.isfinite(b) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_spinor_monodromy() -> np.ndarray | None:
    try:
        from systrophe.quantum_info.spinor_monodromy import expected_monodromy_phase_per_revolution
        return np.array([float(expected_monodromy_phase_per_revolution(vs, float(r)))
                          for r in R_FINE])
    except Exception:
        return None


def probe_aharonov_bohm() -> np.ndarray | None:
    try:
        from systrophe.foundations.aharonov_bohm_ctc import aharonov_bohm_phase
        out = []
        for r in R_FINE:
            try:
                p = float(aharonov_bohm_phase(vs, float(r)))
                out.append(p if math.isfinite(p) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_berry_phase() -> np.ndarray | None:
    try:
        from systrophe.lp.berry_phase_lp import berry_phase_per_revolution
        out = []
        for r in R_FINE:
            try:
                p = float(berry_phase_per_revolution(vs, float(r)))
                out.append(p if math.isfinite(p) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_tidal_forces() -> np.ndarray | None:
    try:
        from systrophe.geometry.tidal_forces import riemann_scalar_radial
        out = []
        for r in R_FINE:
            try:
                v = float(riemann_scalar_radial(vs, float(r)))
                out.append(v if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_frame_dragging() -> np.ndarray | None:
    try:
        from systrophe.geometry.frame_dragging import lense_thirring_frequency
        out = []
        for r in R_FINE:
            try:
                v = float(lense_thirring_frequency(vs, float(r)))
                out.append(v if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_cauchy_stability() -> np.ndarray | None:
    try:
        from systrophe.ctc.cauchy_stability import lyapunov_exponent_at_horizon
        out = []
        for r in R_FINE:
            try:
                v = float(lyapunov_exponent_at_horizon(vs, float(r)))
                out.append(v if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_vacuum_polarization() -> np.ndarray | None:
    try:
        from systrophe.qftcs.vacuum_polarization import vacuum_polarization_at_r
        out = []
        for r in R_FINE:
            try:
                res = vacuum_polarization_at_r(vs, float(r))
                v = res.get("scalar_one_loop", 0.0) if isinstance(res, dict) else float(res)
                out.append(float(v) if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_holographic_complexity() -> np.ndarray | None:
    try:
        from systrophe.quantum_info.holographic_complexity import complexity_growth_rate
        out = []
        for r in R_FINE:
            try:
                v = float(complexity_growth_rate(vs, float(r)))
                out.append(v if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_twistor_lp() -> np.ndarray | None:
    try:
        from systrophe.lp.twistor_lp import twistor_norm
        out = []
        for r in R_FINE:
            try:
                v = float(twistor_norm(vs, float(r)))
                out.append(v if math.isfinite(v) else 0.0)
            except Exception:
                out.append(0.0)
        return np.array(out)
    except Exception:
        return None


def probe_L_metric() -> np.ndarray:
    """Bonnor Case III L metric component itself -- the canonical reference."""
    return np.array([float(vs.analytic_exterior_L(float(r))) for r in R_FINE])


def probe_F_metric() -> np.ndarray:
    return np.array([float(vs.analytic_exterior_F(float(r))) for r in R_FINE])


def find_sharps_in_series(name: str, series: np.ndarray, threshold: float = 5.0) -> list[dict]:
    """Detect adjacent-step outliers in a 1D series at the catcher's
    standard MAD threshold."""
    if series.ndim != 1 or series.size < 3:
        return []
    diffs = np.abs(np.diff(series))
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return []
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(median, 1e-12)
    sharps = []
    for i, d in enumerate(diffs):
        if not math.isfinite(d):
            continue
        # Catcher-style threshold: step > scale * (1+0.3) AND step > median + 2*MAD
        if d > scale * (1.3) and d > median + 2 * (mad if mad > 0 else 1e-12):
            sharps.append({
                "between_indices": [i, i + 1],
                "between_r": [float(R_FINE[i]), float(R_FINE[i + 1])],
                "step": float(d),
                "median": median,
                "mad": mad,
                "ratio_to_median": float(d / max(median, 1e-12)),
            })
    return sharps


def main() -> None:
    print("=" * 70)
    print("Cluster investigation of r = 1.976 structural feature")
    print("=" * 70)
    print(f"Probe grid: {len(R_FINE)} points over r in [{R_FINE[0]:.3f}, "
          f"{R_FINE[-1]:.3f}]")
    print()

    probes = {
        "kg_scattering": probe_kg_scattering(),
        "photon_sphere": probe_photon_sphere(),
        "spinor_monodromy": probe_spinor_monodromy(),
        "aharonov_bohm": probe_aharonov_bohm(),
        "berry_phase": probe_berry_phase(),
        "tidal_forces": probe_tidal_forces(),
        "frame_dragging": probe_frame_dragging(),
        "cauchy_stability": probe_cauchy_stability(),
        "vacuum_polarization": probe_vacuum_polarization(),
        "holographic_complexity": probe_holographic_complexity(),
        "twistor_lp": probe_twistor_lp(),
        "L_metric": probe_L_metric(),
        "F_metric": probe_F_metric(),
    }

    findings = {}
    targets_at_r_1976 = []
    print(f"{'probe':24s} {'available':10s} {'n_sharps':8s} {'sharp r':24s}")
    print("-" * 70)
    for name, series in probes.items():
        if series is None:
            print(f"  {name:22s} skip      -")
            findings[name] = {"available": False}
            continue
        sharps = find_sharps_in_series(name, series)
        # Did any sharp occur within +/- 0.025 of r=1.976?
        near_target = [s for s in sharps
                        if any(abs(rv - 1.976) <= 0.025 for rv in s["between_r"])]
        if near_target:
            targets_at_r_1976.append(name)
        sharp_rs = [f"{(s['between_r'][0]+s['between_r'][1])/2:.4f}" for s in sharps]
        print(f"  {name:22s} ok        {len(sharps):2d}      "
              f"{', '.join(sharp_rs[:5]) if sharp_rs else '-'}")
        findings[name] = {
            "available": True,
            "n_sharps": len(sharps),
            "sharps": sharps,
            "near_r_1976": len(near_target),
        }

    print()
    print("=" * 70)
    print(f"VERDICT: {len(targets_at_r_1976)} probes flag a sharp within +/-0.025 of r=1.976")
    print("=" * 70)
    if targets_at_r_1976:
        for name in targets_at_r_1976:
            print(f"  - {name}")

    # Run catcher across same-quantity comparisons
    quantity_groups = {}
    for name, ser in probes.items():
        if ser is None:
            continue
        # Split by halves of the r grid (low half vs high half)
        mid = len(ser) // 2
        quantity_groups[name] = {
            "lo": ser[:mid],
            "hi": ser[mid:],
        }
    catcher_per_q = {}
    for q, arrs in quantity_groups.items():
        try:
            catcher_per_q[q] = catch_novelty_in_named_arrays(arrs)
        except Exception as e:  # noqa: BLE001
            catcher_per_q[q] = {"error": str(e)}

    findings["catcher_per_quantity"] = {
        q: {"verdict": r.get("verdict"), "n_sharp": len(r.get("sharp_features", []))}
        for q, r in catcher_per_q.items() if "verdict" in r
    }

    out_path = Path(__file__).parent / "investigate_r_1976_cluster_results.json"
    out_path.write_text(json.dumps(findings, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
