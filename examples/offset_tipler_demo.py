"""Off-set Tipler-sinusoid demo: single cylinder vs pair at varying offsets.

Run:
    python examples/offset_tipler_demo.py
"""

from __future__ import annotations

import numpy as np

from systrophe import SystrophePair, VanStockumInterior, find_ctc_intervals


def main() -> None:
    omega, R = 1.0, 1.0
    cyl = VanStockumInterior(omega=omega, R=R)
    print(f"Single van Stockum cylinder")
    print(f"  omega = {omega},  R = {R},  a = omega R = {cyl.a}")
    print(f"  Tipler log-frequency  alpha = sqrt(4 a^2 - 1) = {cyl.alpha:.6f}")
    print(f"  Log-period            2 pi / alpha           = {2*np.pi/cyl.alpha:.6f}")
    print(f"  r-period multiplier   exp(2 pi / alpha)      = {np.exp(2*np.pi/cyl.alpha):.6f}")

    # Single-cylinder CTC bands from analytic L
    L_fn = lambda r: cyl.analytic_exterior_L(np.asarray(r, dtype=float))
    bands_single = find_ctc_intervals(L_fn, r_min=1.001, r_max=200.0, n_grid=50_001)
    print(f"\nSingle-cylinder CTC bands in r in [1.001, 200]:")
    for i, (a, b) in enumerate(bands_single[:6]):
        print(f"  band {i+1}: r in [{a:.4f}, {b:.4f}]   "
              f"log span = {np.log(b/a):.4f}")

    # Pair at various phase offsets
    print(f"\nPair (cyl, cyl) at varying delta_offset:")
    for delta in [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]:
        pair = SystrophePair.from_cylinders(cyl, cyl, delta_offset=delta)
        bands = pair.ctc_bands(r_min=1.001, r_max=200.0, n_grid=20_001)
        log_meas = sum(np.log(b/a) for a, b in bands)
        print(f"  delta = {delta:.4f}:  {len(bands):3d} CTC bands,  "
              f"total log measure = {log_meas:.4f}")

    # Z_3 cover comparison
    try:
        from systrophe.dinos_bridge import z3_branch_match_to_tipler_alpha

        print(f"\nZ_3 Mobius cover vs Tipler log-grid (N=24):")
        out = z3_branch_match_to_tipler_alpha(cyl, N=24)
        print(f"  Tipler fundamental eigenvalue:   {out['tipler_eigenvalue']:.6f}")
        print(f"  Z_3 branch=0 (trivial):          {out['z3_eigenvalues'][0]:.6f}")
        print(f"  Z_3 branch=1 (off-set sector):   {out['z3_eigenvalues'][1]:.6f}")
        print(f"  Z_3 branch=2 (off-set sector):   {out['z3_eigenvalues'][2]:.6f}")
        print(f"  best branch: {out['best_branch_match']}, "
              f"residual: {out['relative_residual']:.2e}")
    except ImportError:
        print("\n(Dinos bridge unavailable; skipping Z_3 comparison)")


if __name__ == "__main__":
    main()
