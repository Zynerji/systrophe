"""Investigation of the first chronology horizon cluster.

Hypothesis: 9 novel-structure flags from 7 modules all cluster at
r in [1.36, 2.90], near the first CH at r ~ 1.83. Are they really
flagging the SAME location? What's the analytic driver?

Plan
----
1. Compute the theoretical first-CH radius r_1 analytically.
2. Fine-grained scan (n=200 points) of each flagged module in
   r in [1.0, 3.5]; locate the precise feature position.
3. Overlay positions: build a feature-density histogram in r.
4. Compute F(r), F'(r), F''(r), K(r), K'(r) along the way and
   correlate feature loci with their zeros / extrema.
5. Report which analytic driver (F=0, F''=0 etc.) best predicts
   each module's feature.

Output: examples/first_ch_cluster_investigation.json + plot data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import scan_novelty
from systrophe.vanstockum import VanStockumInterior

vs = VanStockumInterior(omega=1.0, R=1.0)
OUT_PATH = Path(__file__).parent / "first_ch_cluster_investigation.json"

# --- Step 1: theoretical first CH ---
alpha = vs.alpha  # sqrt(3) for a=1
gamma = math.pi - math.atan(alpha)
u_1 = (math.pi - gamma) / alpha  # = arctan(alpha)/alpha
r_1_theory = vs.R * math.exp(u_1)
print(f"Theoretical first CH: r_1 = R * exp(arctan(alpha)/alpha) = "
      f"{r_1_theory:.6f}")
print(f"  alpha = sqrt(3) = {alpha:.6f}")
print(f"  gamma = pi - arctan(alpha) = {gamma:.6f}")
print()

# --- Step 2: fine-grained scans ---
R_FINE = np.linspace(1.0, 3.5, 200)
N_BITS = 32


def fine_scan(name: str, fn):
    """Return (positions, hamming_steps, sharp_features)."""
    res = scan_novelty(R_FINE, fn, n_bits=N_BITS)
    # Recover successive Hamming steps for the heatmap
    addresses = res.addresses
    steps = [int(np.sum(addresses[k] != addresses[k - 1]))
             for k in range(1, len(addresses))]
    sharp_positions = [s["parameter_value"] for s in res.sharp_features]
    return {
        "name": name,
        "verdict": res.verdict,
        "n_sharp": len(res.sharp_features),
        "sharp_positions": sharp_positions,
        "hamming_steps": steps,
        "r_axis": list(R_FINE),
    }


# Module scan fns (the 7 that showed novelty)
def fn_cauchy(r):
    from systrophe.cauchy_stability import lyapunov_exponent_at_horizon
    return np.array([float(lyapunov_exponent_at_horizon(vs, float(r)))])


def fn_spinor(r):
    from systrophe.spinor_monodromy import expected_monodromy_phase_per_revolution
    return np.array([float(expected_monodromy_phase_per_revolution(vs, float(r)))])


def fn_synchrotron(r):
    from systrophe.synchrotron_analog import (
        orbital_frequency, effective_gamma_factor,
        synchrotron_critical_frequency,
    )
    Om = orbital_frequency(vs, float(r))
    g = effective_gamma_factor(vs, float(r))
    om_c = synchrotron_critical_frequency(vs, float(r))
    return np.array([
        Om if math.isfinite(Om) else 0.0,
        g if math.isfinite(g) else 0.0,
        om_c if math.isfinite(om_c) else 0.0,
    ])


def fn_kg(r):
    from systrophe.kg_scattering import effective_potential
    V = effective_potential(vs, float(r), omega=1.0)
    return np.array([float(V) if math.isfinite(V) else 0.0])


def fn_bh_pair(r):
    from systrophe.bh_pair_production import schwinger_analog_rate
    rate = schwinger_analog_rate(vs, float(r))
    return np.array([
        math.log10(max(rate.production_rate, 1e-300)),
        math.log10(max(rate.field_strength, 1e-30)),
    ])


def fn_unruh(r):
    from systrophe.unruh_effect import combined_unruh_hawking_T
    res = combined_unruh_hawking_T(vs, float(r))
    return np.array([
        min(res["T_Unruh"], 1e6),
        res["T_Hawking"],
        min(res["T_combined"], 1e6),
    ])


def fn_one_loop(r):
    from systrophe.one_loop_backreaction import (
        one_loop_F_correction, corrected_F, trace_anomaly_at_r,
    )
    return np.array([one_loop_F_correction(vs, float(r)),
                     corrected_F(vs, float(r)),
                     trace_anomaly_at_r(vs, float(r))])


scans = {
    "cauchy_stability": fine_scan("cauchy_stability", fn_cauchy),
    "spinor_monodromy": fine_scan("spinor_monodromy", fn_spinor),
    "synchrotron_analog": fine_scan("synchrotron_analog", fn_synchrotron),
    "kg_scattering": fine_scan("kg_scattering", fn_kg),
    "bh_pair_production": fine_scan("bh_pair_production", fn_bh_pair),
    "unruh_effect": fine_scan("unruh_effect", fn_unruh),
    "one_loop_backreaction": fine_scan("one_loop_backreaction", fn_one_loop),
}

# --- Step 3: feature density histogram ---
all_positions = []
for name, s in scans.items():
    all_positions.extend(s["sharp_positions"])

print("Per-module sharp positions in r in [1.0, 3.5]:")
for name, s in scans.items():
    print(f"  {name:<25s} {s['verdict']:<20s} n_sharp={s['n_sharp']} "
          f"positions={[f'{p:.3f}' for p in s['sharp_positions']]}")

print()
print(f"Total sharp features across all modules: {len(all_positions)}")
print(f"Theoretical first CH:  r_1 = {r_1_theory:.4f}")
print(f"Mean of all sharps:    {np.mean(all_positions):.4f}" if all_positions else "no sharps")
print(f"Median:                {np.median(all_positions):.4f}" if all_positions else "")

# --- Step 4: analytic drivers along r grid ---
F_vals = vs.analytic_exterior_F(R_FINE)
K_vals = vs.analytic_exterior_K(R_FINE)
L_vals = vs.analytic_exterior_L(R_FINE)

# Compute |F'| and F'' numerically
dF_dr = np.gradient(F_vals, R_FINE)
d2F_dr2 = np.gradient(dF_dr, R_FINE)

# Locate F=0 (chronology horizon)
F_zeros = []
for i in range(len(R_FINE) - 1):
    if F_vals[i] * F_vals[i + 1] < 0:
        t = F_vals[i] / (F_vals[i] - F_vals[i + 1])
        F_zeros.append(float(R_FINE[i] + t * (R_FINE[i + 1] - R_FINE[i])))

print(f"\nF=0 (chronology horizons) in scan range: {F_zeros}")

# Compute |F'| max and F'' max loci near first CH
i_min_F = int(np.argmin(np.abs(F_vals)))
i_max_dF = int(np.argmax(np.abs(dF_dr)))
i_max_d2F = int(np.argmax(np.abs(d2F_dr2)))
print(f"r where |F'| is max:  {R_FINE[i_max_dF]:.4f}  (|F'|={np.abs(dF_dr[i_max_dF]):.4f})")
print(f"r where |F''| is max: {R_FINE[i_max_d2F]:.4f}  (|F''|={np.abs(d2F_dr2[i_max_d2F]):.4f})")

# Step 5: which analytic driver best matches each module's features
print()
print("Per-module best analytic driver:")
analytic_loci = {
    "F=0 (first CH)": F_zeros[0] if F_zeros else None,
    "|F'| max": float(R_FINE[i_max_dF]),
    "|F''| max": float(R_FINE[i_max_d2F]),
    "r=R+1 (Schwarzschild-like)": vs.R + 1.0,
}
for name, s in scans.items():
    if not s["sharp_positions"]:
        continue
    distances = {
        driver: min(abs(p - loc) for p in s["sharp_positions"])
        if loc is not None else float("inf")
        for driver, loc in analytic_loci.items()
    }
    best = min(distances, key=distances.get)
    print(f"  {name:<25s} closest sharp to '{best}' (delta_r = "
          f"{distances[best]:.4f})")

payload = {
    "theoretical_first_CH": float(r_1_theory),
    "F_zero_loci_in_range": F_zeros,
    "analytic_loci": {k: (float(v) if v is not None else None)
                       for k, v in analytic_loci.items()},
    "module_scans": scans,
    "all_sharp_positions": [float(p) for p in all_positions],
    "summary": {
        "n_total_sharp": len(all_positions),
        "mean_sharp_r": float(np.mean(all_positions)) if all_positions else None,
        "median_sharp_r": float(np.median(all_positions)) if all_positions else None,
        "std_sharp_r": float(np.std(all_positions)) if all_positions else None,
        "distance_from_theoretical_CH": [
            float(p - r_1_theory) for p in all_positions
        ],
    },
}
OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {OUT_PATH}")
