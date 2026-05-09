"""CTC zoo: side-by-side comparison of all CTC spacetimes in Systrophe.

Iterates over the three implemented analytical CTC spacetimes
(Tipler/van Stockum, Goedel, Gott) and reports the canonical
characterising quantities for each. Demonstrates the framework
integration of the `systrophe.spacetimes` subpackage.

Run:
    python examples/ctc_zoo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe import (
    VanStockumInterior,
    find_single_cylinder_windows,
)
from systrophe.spacetimes import (
    GodelUniverse,
    GottPair,
    godel_ctc_radius,
    gott_critical_velocity,
    gott_critical_mu,
)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> dict:
    results: dict = {}

    # 1. Tipler / van Stockum cylinder
    section("Tipler / van Stockum cylinder (a = 1)")
    vs = VanStockumInterior(omega=1.0, R=1.0)
    print(f"  alpha (log-frequency)             = sqrt(4 a^2 - 1) = {vs.alpha:.6f}")
    print(f"  log-period                        = 2 pi / alpha    = {2*np.pi/vs.alpha:.6f}")
    print(f"  r-period multiplier               = exp(2 pi/alpha) = {np.exp(2*np.pi/vs.alpha):.4f}")
    windows = find_single_cylinder_windows(vs, r_min=1.001, r_max=200.0)
    print(f"  CTC bands in r in [1, 200]        : {len(windows)}")
    for i, w in enumerate(windows):
        print(f"    band {i+1}: r in [{w.r_inner:.3f}, {w.r_outer:.3f}]")
    print(f"  exterior regime                   : {vs.regime}")
    print(f"  interior regime (a vs 1)          : {vs.interior_regime}")
    results["tipler"] = {
        "a": vs.a,
        "alpha": vs.alpha,
        "log_period": 2*np.pi/vs.alpha,
        "n_bands_in_1_200": len(windows),
        "first_band": [windows[0].r_inner, windows[0].r_outer] if windows else None,
        "exterior_regime": vs.regime,
        "interior_regime": vs.interior_regime,
    }

    # 2. Goedel universe
    section("Goedel rotating-dust universe")
    g = GodelUniverse(a=1.0)
    print(f"  fundamental scale a               = {g.a}")
    print(f"  CTC threshold radius              = arcsinh(1) = {g.ctc_threshold_radius:.6f}")
    print(f"  dust density rho                  = 1/(8 pi a^2) = {g.dust_density:.6f}")
    print(f"  cosmological constant Lambda      = -1/(2 a^2)  = {g.cosmological_constant:.6f}")
    print(f"  Goedel angular velocity Omega     = 1/a          = {g.angular_velocity:.6f}")
    # Check at three radii: below, at, above threshold
    for r in [0.5, g.ctc_threshold_radius, 1.5]:
        print(f"  g_phi phi(r={r:.4f})            = {float(g.gphiphi(r)):+.4f}")
    results["godel"] = {
        "a": g.a,
        "ctc_threshold_radius": g.ctc_threshold_radius,
        "dust_density": g.dust_density,
        "cosmological_constant": g.cosmological_constant,
        "angular_velocity": g.angular_velocity,
    }

    # 3. Gott pair
    section("Gott pair (symmetric cosmic strings)")
    sample_mu = 0.05
    print(f"  per-string mass per unit length   = mu = {sample_mu}")
    v_crit = gott_critical_velocity(sample_mu)
    print(f"  critical velocity                 = sin(4 pi mu) = {v_crit:.4f}")
    print(f"  critical Lorentz factor           = sec(4 pi mu) = {1.0/np.cos(4*np.pi*sample_mu):.4f}")
    print(f"  inverse: critical mu at v = 0.5   = arcsin(0.5)/(4 pi) = {gott_critical_mu(0.5):.6f}")
    # Demonstrate has_ctc above and below threshold
    pair_below = GottPair(mu=sample_mu, v=0.5 * v_crit)
    pair_above = GottPair(mu=sample_mu, v=min(2.0 * v_crit, 0.99))
    print(f"  v = 0.5 * v_crit: gamma*v = {pair_below.gamma * pair_below.v:.4f}, has_ctc = {pair_below.has_ctc()}")
    print(f"  v = 2.0 * v_crit: gamma*v = {pair_above.gamma * pair_above.v:.4f}, has_ctc = {pair_above.has_ctc()}")
    results["gott"] = {
        "sample_mu": sample_mu,
        "critical_velocity": v_crit,
        "critical_gamma": 1.0/np.cos(4*np.pi*sample_mu),
        "below_threshold_has_ctc": pair_below.has_ctc(),
        "above_threshold_has_ctc": pair_above.has_ctc(),
    }

    # 4. Common characterisation table
    section("Comparison summary")
    print(f"  {'Spacetime':<20} {'CTC source':<25} {'Threshold':<25}")
    print(f"  {'Tipler/van Stockum':<20} {'rotating dust cylinder':<25} {'a > 1/2 (exterior)':<25}")
    print(f"  {'Goedel':<20} {'global rotation + Lambda':<25} {'r > arcsinh(1) ~ 0.881':<25}")
    print(f"  {'Gott pair':<20} {'cosmic-string pair':<25} {'gamma*v > tan(4 pi mu)':<25}")

    # 5. Save JSON for whitepaper / scripting
    out_path = Path("examples") / "ctc_zoo_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    section(f"Results written to {out_path}")
    return results


if __name__ == "__main__":
    main()
