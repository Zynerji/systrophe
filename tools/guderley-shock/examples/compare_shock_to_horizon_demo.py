"""Empirical comparison: Guderley shock divergence vs QFTCS Cauchy-horizon divergence.

Not a derived equivalence — just the two powers side by side and the
residual. The fact that gamma=5/3 gives a Guderley density-divergence
power ≈ -0.905 while the QFTCS Boulware <T_tt> is ≈ -1.000 at every
Tipler Cauchy horizon is the headline observation, not a tunable
match.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from systrophe.vanstockum import VanStockumInterior

from guderley_shock import (
    compare_to_cauchy_horizon,
    compute_guderley_exponent,
    density_power_at_focus,
)


def main():
    print("Guderley converging-shock vs QFTCS Boulware Cauchy-horizon divergence")
    print("=" * 72)
    print()

    for gamma_pretty, gamma in [("5/3", 5.0 / 3.0), ("7/5", 7.0 / 5.0)]:
        e = compute_guderley_exponent(gamma=gamma, n=3)
        p_gud = density_power_at_focus(gamma=gamma, n=3, beta=e.beta)
        print(f"gamma = {gamma_pretty}  (spherical, n=3)")
        print(f"  Guderley beta = {e.beta:.6f}  ({e.method})")
        print(f"  rho-divergence power = {p_gud:.4f}")
        print()

    print("QFTCS comparison (supercritical Tipler exterior, omega=2, R=1):")
    print()
    vs = VanStockumInterior(omega=2.0, R=1.0)
    for gamma_pretty, gamma in [("5/3", 5.0 / 3.0), ("7/5", 7.0 / 5.0)]:
        cmp_ = compare_to_cauchy_horizon(vs, gamma=gamma, n=3)
        print(f"  gamma = {gamma_pretty}:")
        print(f"    Guderley density power      = {cmp_.guderley_density_power:+.4f}")
        print(f"    QFTCS T_tt power at r_H={cmp_.horizon_r:.3f} = "
              f"{cmp_.qftcs_T_tt_power:+.4f}")
        print(f"    |residual|                    = {cmp_.absolute_residual:.4f}")
        print(f"    interpretation: {cmp_.interpretation}")
        print()


if __name__ == "__main__":
    main()
