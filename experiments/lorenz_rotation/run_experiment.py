"""Orchestrator: induce a Lorenz-class rotation in a Tipler cylinder and audit it.

Runs the full investigation and writes a machine-readable companion JSON plus the
mandatory address-space novelty-catcher verdict (Systrophe project rule, 2026-05-11):
every parameter scan must pass its primary output through the catcher before any
"validated" / "null" claim.

Sections
--------
H3  Rotating-dust Lorenz reduction reproduces the canonical strange attractor
    (Lyapunov spectrum, Kaplan-Yorke dimension, divergence).
H0  Single rigid cylinder geodesics are integrable (conserved p_t, ~0 Lyapunov).
H2  Time-dependent rotation -> conservative chaos (positive Lyapunov) but
    divergence == 0 (Liouville) -> NO attractor.
SCAN  Drive r is scanned; the catcher independently localises the chaos onset.
CTC  The attractor rotation a(t) drives a flickering CTC band structure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; the catcher report contains lambda-2 glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from systrophe.catchers.novelty_catcher import scan_novelty, summarize_novelty_for_report

from rotating_dust_lorenz import (
    RotatingDustLorenz,
    largest_lyapunov_rk4,
    lyapunov_spectrum,
    rotation_parameter_timeseries,
)
from geodesic_rotation_chaos import (
    constant_omega,
    driven_omega,
    finite_time_lyapunov,
    phase_volume_divergence,
    GeodesicRotation,
)
from chaotic_ctc import chaotic_ctc_timeseries, adiabaticity_ratio


def section_h3() -> dict:
    print("\n=== H3: rotating-dust Lorenz reduction vs canonical attractor ===")
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    spec = lyapunov_spectrum(
        m, np.array([1.0, 1.0, 1.0]), t_max=400.0, dt=0.005,
        renorm_every=20, t_transient=20.0,
    )
    out = {
        "sigma": m.sigma, "r": m.r, "b": m.b,
        "cell_aspect": m.cell_aspect,
        "hopf_threshold_r": m.r_critical_hopf,
        "divergence_trJ": m.divergence,
        "lyapunov_exponents": spec["exponents"].tolist(),
        "lyapunov_sum": spec["sum"],
        "kaplan_yorke_dim": spec["kaplan_yorke_dim"],
        "kolmogorov_sinai": spec["kolmogorov_sinai"],
        "canonical_reference": {
            "lyapunov": [0.906, 0.0, -14.572],
            "kaplan_yorke_dim": 2.062,
            "divergence": -13.667,
        },
    }
    print(f"  Lyapunov spectrum = {np.round(spec['exponents'],4)}  (canon [0.906, 0, -14.57])")
    print(f"  Lyapunov sum = {spec['sum']:.4f}  ==  divergence = {m.divergence:.4f}")
    print(f"  Kaplan-Yorke dim = {spec['kaplan_yorke_dim']:.4f}  (canon 2.062)")
    print(f"  Hopf onset r_H = {m.r_critical_hopf:.3f}")
    return out


def section_h0_h2() -> dict:
    print("\n=== H0/H2: geodesic chaos is conservative (no attractor) ===")
    rows = []
    configs = [
        ("rigid (eps=0)", 0.0, 1.0),
        ("weak drive", 0.2, 0.9),
        ("medium drive", 0.4, 0.9),
        ("strong drive", 0.5, 1.3),
    ]
    for label, eps, wd in configs:
        if eps == 0.0:
            g = constant_omega(0.8)
            g = GeodesicRotation(g.omega_fn, g.omega_dot_fn, ell=0.6)
        else:
            g = driven_omega(omega0=0.8, eps=eps, drive_freq=wd, ell=0.6)
        s0 = g.initial_state_from_E(r0=0.7, E=1.2, p_r_sign=1.0)
        le = finite_time_lyapunov(g, s0, tau_max=300.0, n_renorm=300)
        div = phase_volume_divergence(g, s0)
        tr = g.integrate(s0, tau_max=120.0, n_samples=2001)
        pt_range = float(np.ptp(tr["p_t"]))
        rows.append({
            "config": label, "eps": eps, "drive_freq": wd,
            "ftle": le, "phase_divergence": div, "p_t_range": pt_range,
        })
        print(f"  {label:14s}: FTLE={le:+.4f}  div={div:+.2e}  p_t range={pt_range:.3f}")
    print("  -> divergence ~0 in every case: Hamiltonian, so NO attractor (contrast H3).")
    return {"rows": rows}


def section_scan() -> tuple[dict, object]:
    print("\n=== SCAN: drive r and let the catcher localise the chaos onset ===")
    r_values = np.linspace(0.5, 40.0, 24)

    def le_of_r(r):
        m = RotatingDustLorenz(sigma=10.0, r=float(r), b=8.0 / 3.0)
        return largest_lyapunov_rk4(
            m, np.array([1.0, 1.0, 1.0]), t_max=120.0, dt=0.005, t_transient=30.0
        )

    le_curve = np.array([le_of_r(r) for r in r_values])

    # catcher output: settling-robust, lobe-symmetric fingerprint of the
    # post-transient |X| distribution. A point attractor (drive below onset)
    # collapses |X| onto a single value sqrt(b(r-1)); a strange attractor
    # spreads |X| down toward the lobe-crossing near zero. Long transient so
    # sub-onset spirals fully decay.
    def x_fingerprint(r):
        m = RotatingDustLorenz(sigma=10.0, r=float(r), b=8.0 / 3.0)
        traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=300.0, dt=0.02,
                           t_transient=160.0)
        absX = np.abs(traj["s"][:, 0])
        return np.quantile(absX, np.linspace(0.0, 1.0, 16))

    result = scan_novelty(
        r_values, x_fingerprint, n_bits=32, radii=(4, 8, 12, 16),
        parameter_label="drive_r",
    )

    # first r where LE crosses positive = numerical chaos onset
    onset_idx = int(np.argmax(le_curve > 0.02))
    onset_le = float(r_values[onset_idx]) if np.any(le_curve > 0.02) else None
    catcher_onsets = [f["parameter_value"] for f in result.sharp_features]
    print(f"  LE>0 onset (drive r) ~ {onset_le}")
    print(f"  catcher sharp features at r = {[round(x,2) for x in catcher_onsets]}")
    print(f"  catcher verdict = {result.verdict}")

    scan_out = {
        "r_values": r_values.tolist(),
        "lyapunov_curve": le_curve.tolist(),
        "le_onset_r": onset_le,
        "catcher_verdict": result.verdict,
        "catcher_sharp_features": result.sharp_features,
        "catcher_lambda2": {str(k): v for k, v in result.lambda_2_at_radius.items()},
    }
    return scan_out, result


def section_ctc() -> dict:
    print("\n=== CTC: chaotic rotation -> flickering time-machine bands ===")
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=120.0, dt=0.01,
                       t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=1.5, eps=0.2, clip_min=0.55)
    cts = chaotic_ctc_timeseries(rot, R=1.0, r_min=1.05, r_max=20.0, stride=5)
    lm = cts["log_measure"]
    ratio = adiabaticity_ratio(0.906, 1.5, 1.0, 3.0)
    out = {
        "a_min": rot["a_min"], "a_max": rot["a_max"], "n_clipped": rot["n_clipped"],
        "alpha_min": float(cts["alpha"].min()), "alpha_max": float(cts["alpha"].max()),
        "ctc_log_measure_mean": float(lm.mean()),
        "ctc_log_measure_std": float(lm.std()),
        "ctc_log_measure_min": float(lm.min()),
        "ctc_log_measure_max": float(lm.max()),
        "band_count_min": int(cts["n_bands"].min()),
        "band_count_max": int(cts["n_bands"].max()),
        "adiabaticity_ratio_unit_scale": ratio,
    }
    print(f"  a(t) in [{rot['a_min']:.3f}, {rot['a_max']:.3f}]  (clipped {rot['n_clipped']})")
    print(f"  alpha in [{out['alpha_min']:.3f}, {out['alpha_max']:.3f}]")
    print(f"  CTC log-measure: mean={lm.mean():.3f} std={lm.std():.3f} "
          f"range=[{lm.min():.3f},{lm.max():.3f}]")
    print(f"  band count flickers {out['band_count_min']}..{out['band_count_max']}")
    print(f"  adiabaticity ratio (unit scale) = {ratio:.1f}  "
          f"(>1: needs slow-dust scale separation; reported, not assumed)")
    return out


def main():
    t0 = time.time()
    results = {}
    results["H3_lorenz_reduction"] = section_h3()
    results["H0_H2_geodesic"] = section_h0_h2()
    scan_out, scan_result = section_scan()
    results["drive_scan"] = scan_out
    results["CTC_flicker"] = section_ctc()

    print("\n=== Novelty catcher report (mandatory) ===")
    print(summarize_novelty_for_report(scan_result))

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = Path(__file__).with_name("lorenz_rotation_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}  ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
