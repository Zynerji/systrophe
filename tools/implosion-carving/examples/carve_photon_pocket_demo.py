"""Smoke demo: carve a trapped-null pocket on the canonical fixture."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from implosion_carving import ImplosionCarver


def main():
    car = ImplosionCarver(omega=1.0, R=1.0)
    print(f"Spacetime: omega={car.omega}  R={car.R}  a={car.a:.3f}")
    print()

    for r_target in (1.5, 2.0, 3.0):
        summ = car.summary(r_target=r_target)
        status = "CARVED" if summ.is_carved else "FAILED"
        stable = "stable" if summ.is_stable else "unstable"
        print(f"  r_target={r_target}  M_eng={summ.M_engineered}  "
              f"[Schw limit M={summ.schwarzschild_limit_M:.3f}]")
        print(f"    {status}, {stable}, b={summ.impact_parameter:.4f}, "
              f"omega_orbit={summ.omega_orbit:.4f}")
        print(f"    closure residual db/dr = {summ.closure_residual_dbdr:.2e}")
        print()

    print("Z_3 monodromy signature (cached, topological):")
    sig = car.z3_signature(N=256)
    print(f"  triplet (rescaled): {sig.triplet_eigenvalues.tolist()}")
    print(f"  continuum {{0, 1/9, 4/9}}: {sig.continuum_triplet.tolist()}")
    print(f"  convergence error: {sig.triplet_convergence_error:.2e}")


if __name__ == "__main__":
    main()
