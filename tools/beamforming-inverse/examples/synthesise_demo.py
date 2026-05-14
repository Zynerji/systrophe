"""Smoke demo: synthesise a target phasor profile via the inverse solver."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from beamforming_inverse import (
    BeamformingDesign,
    solve_beamforming_inverse,
    synthesised_array,
)


def main():
    # Three cylinders with distinct alpha (different a parameters)
    a = np.array([0.6, 0.75, 0.9])
    alpha = np.sqrt(4 * a * a - 1)
    design = BeamformingDesign(
        R=np.ones(3), alpha=alpha, a=a, p=np.ones(3),
    )

    # Target: prescribe a "main lobe" at r=2 with magnitude 3 and zero phase
    # plus a "null" at r=4. Sample on a log-spaced grid.
    rs = np.geomspace(1.05, 8.0, 40)
    main_r = 2.0
    null_r = 4.0
    z_target = 3.0 * np.exp(-(np.log(rs / main_r) ** 2) / 0.2)
    z_target = z_target.astype(complex)
    z_target *= np.exp(-((rs - null_r) / 0.5) ** 2 * 1j * 0)  # phase 0 for now

    # Solve
    result = solve_beamforming_inverse(design, rs, z_target)
    print(f"N cylinders: {design.N}, M sample radii: {len(rs)}")
    print(f"is_overdetermined: {result.is_overdetermined}")
    print(f"rank: {result.rank}, cond(G): {result.condition_number:.2e}")
    print()
    print(f"Recovered A:     {result.A}")
    print(f"Recovered delta: {result.delta}")
    print()
    print(f"Residual norm:     {result.residual_norm:.4e}")
    print(f"Relative residual: {result.relative_residual:.4e}")

    synth = synthesised_array(design, result)
    z_pred = synth.phasor_field(rs)["phasor"]
    print()
    print("Reconstruction at first 5 radii:")
    for i in range(5):
        print(f"  r={rs[i]:.3f}  target=({z_target[i].real:+.3f}, "
              f"{z_target[i].imag:+.3f})  pred=({z_pred[i].real:+.3f}, "
              f"{z_pred[i].imag:+.3f})")


if __name__ == "__main__":
    main()
