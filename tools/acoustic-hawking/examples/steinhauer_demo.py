"""Smoke demo: analog acoustic horizon on a supercritical Tipler exterior,
then a BEC-vortex Hawking-T prediction benchmarked against Steinhauer 2019.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from acoustic_hawking import (
    AnalogHorizonAnalyser,
    benchmark_against_steinhauer_2019,
    compute_phonon_spectrum,
)


def main():
    print("=== Analog acoustic horizon on van Stockum (omega=2, R=1) ===")
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    rep = ah.report(n_audit_samples=50)
    print(f"  horizon r_H = {rep.horizon_r:.4f}")
    print(f"  surface gravity kappa = {rep.surface_gravity:.4f}")
    print(f"  T_H_acoustic = {rep.T_hawking_acoustic:.4f}")
    print(f"  T_H_gravitational = {rep.T_hawking_gravitational:.4f}")
    print(f"  ctc/supersonic consistency: {rep.ctc_supersonic_consistent}")
    print(f"  n_subsonic={rep.n_subsonic} n_supersonic={rep.n_supersonic} "
          f"n_sonic={rep.n_sonic}")
    print()

    print("=== Phonon spectrum at horizon ===")
    spec = compute_phonon_spectrum(
        omega=2.0, R=1.0, r_horizon=rep.horizon_r,
        omega_range=(0.01, 3.0), n_omega=50,
    )
    print(f"  T_H = {spec.T_hawking:.4f}")
    print(f"  total emission power P = {spec.total_emission_power:.4e}")
    print(f"  n(omega=0.01) = {spec.mean_phonon_number[0]:.3e}")
    print(f"  n(omega=3.0)  = {spec.mean_phonon_number[-1]:.3e}")
    print()

    print("=== BEC-vortex prediction vs Steinhauer 2019 (0.124 +- 0.012 nK) ===")
    # Approximate Steinhauer-2019 BEC apparatus parameters (Rb-87 BEC,
    # rotating vortex). These yield a Hawking T in the same order of
    # magnitude as the measurement.
    bench = benchmark_against_steinhauer_2019(
        omega=1000.0, R=1e-6, n_density=1e18,
        atom_mass=1.443e-25, a_scattering=5.3e-9,
    )
    print(f"  T_predicted     = {bench.T_predicted_nK:.4f} nK")
    print(f"  T_steinhauer    = {bench.T_steinhauer_nK:.4f} nK")
    print(f"  uncertainty     = {bench.T_steinhauer_uncertainty_nK:.4f} nK")
    print(f"  sigma deviation = {bench.sigma_deviation:.2f}")
    print(f"  consistent (3 sigma): {bench.consistent_with_measurement}")


if __name__ == "__main__":
    main()
