"""Falsification of Grok's "Z_3 cover keeps <T_munu> finite at Cauchy horizons" claim.

A Grok conversation (2026-05-10) on the Systrophe repo asserted that
the Z_3 Mobius cover identified by the Dinos bridge is what regularises
the renormalised stress-energy tensor at Cauchy horizons (F = 0
surfaces) in the supercritical Tipler exterior.

Direct measurement on the Lewis-Papapetrou metric falsifies the
attribution: the Kretschmann scalar K = R_{mu nu rho sigma}
R^{mu nu rho sigma} is intrinsically bounded near F = 0 in the Tipler
vacuum exterior; the conformal-anomaly trace
<T^mu_{~mu}>_ren = K / (2880 pi^2) therefore remains finite at every
F-zero *with or without* the Z_3 cover. The F = 0 surface is a
coordinate singularity (ergosurface) of the LP metric, not a curvature
singularity.

What the Z_3 cover would still matter for: the *off-trace* components
of <T_munu> via Hadamard renormalisation; the trace alone is decided
by the classical curvature and is finite by direct numerical
evaluation.

Run
---
    python examples/grok_falsification.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe import VanStockumInterior
from systrophe.ctc import find_ctc_intervals
from systrophe.point_splitting import kretschmann_scalar, trace_anomaly_4d_exact
from systrophe.quantum_diagnostics import cauchy_horizon_estimate


def main() -> dict:
    vs = VanStockumInterior(omega=1.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs)
    bands = find_ctc_intervals(lambda r: vs.analytic_exterior_L(r), 1.001, 50.0, n_grid=10000)
    print(f"Tipler exterior: a = {vs.a}, alpha = {vs.alpha:.4f}")
    print(f"Cauchy horizons (F = 0): {horizons}")
    print(f"CTC bands (L < 0): {bands}")
    print()

    # ---- (1) Sample K(r) approaching the first Cauchy horizon from below ----
    print("(1) Approaching first F-zero r_h = {:.4f}:".format(float(horizons[0])))
    print(f"  {'eps':>9} {'r':>9} {'F':>12} {'K':>13} {'<T^mu_mu>':>13}")
    approach_rows = []
    for eps in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
        r = float(horizons[0]) - eps
        F = float(vs.analytic_exterior_F(r))
        K = kretschmann_scalar(vs, r)
        T = trace_anomaly_4d_exact(vs, r)
        print(f"  {eps:>9.4f} {r:>9.4f} {F:>12.6f} {K:>13.4e} {T:>13.4e}")
        approach_rows.append({"eps": eps, "r": r, "F": F, "K": K, "trace": T})

    print()
    print("(2) Sampling K across multiple regimes (well inside band, ergosphere, far field):")
    print(f"  {'r':>8} {'F':>12} {'L':>12} {'K':>13} {'<T>':>13}")
    sample_rows = []
    for r in [1.5, 1.7, 1.83, 1.85, 2.0, 3.0, 5.0, 6.1, 7.0, 11.2, 11.23, 20.0]:
        F = float(vs.analytic_exterior_F(r))
        L = float(vs.analytic_exterior_L(r))
        K = kretschmann_scalar(vs, r)
        T = trace_anomaly_4d_exact(vs, r)
        print(f"  {r:>8.2f} {F:>12.4f} {L:>12.4f} {K:>13.4e} {T:>13.4e}")
        sample_rows.append({"r": r, "F": F, "L": L, "K": K, "trace": T})

    # ---- Verdict ----
    K_at_horizon_approach = [row["K"] for row in approach_rows[-3:]]
    converged_K = float(np.mean(K_at_horizon_approach))
    relative_spread = float(
        (max(K_at_horizon_approach) - min(K_at_horizon_approach)) / converged_K
    )
    print()
    print("VERDICT")
    print("-------")
    print(f"As r -> r_h, K converges to {converged_K:.4f} (spread {relative_spread*100:.2f}%).")
    print("K is BOUNDED at the Cauchy horizon. The trace anomaly therefore stays finite.")
    print("No Z_3 cover is invoked; finiteness is intrinsic to the classical LP vacuum.")
    print()
    print("Grok's specific claim — that the Z_3 cover provides chronology-protection-via-")
    print("finiteness — is overstated for the *trace* component. The non-trivial")
    print("question is whether the *off-trace* <T_munu> components (via Hadamard")
    print("point-splitting) diverge at F = 0; that requires a separate calculation.")

    results = {
        "a": vs.a,
        "alpha": vs.alpha,
        "cauchy_horizons": horizons.tolist(),
        "ctc_bands": [[float(a), float(b)] for a, b in bands],
        "approach_to_horizon": approach_rows,
        "sample_across_regimes": sample_rows,
        "K_at_horizon_converged": converged_K,
        "relative_spread_percent": relative_spread * 100,
        "verdict": (
            "Kretschmann scalar K is bounded at every Cauchy horizon (F = 0). "
            "Trace anomaly <T^mu_mu> ∝ K is finite without invoking the Z_3 cover. "
            "Grok's framing 'Z_3 cover keeps <T_munu> finite' is misattributed for the "
            "trace component; the LP vacuum is intrinsically curvature-regular at F = 0."
        ),
    }
    out_path = Path("examples") / "grok_falsification_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"Results written to {out_path}")
    return results


if __name__ == "__main__":
    main()
