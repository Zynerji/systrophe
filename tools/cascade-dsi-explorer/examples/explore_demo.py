"""Smoke demo: explore single-level vs multi-level cascade-DSI zero sets."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cascade_dsi_explorer import CascadeDSIExplorer, scan_phase_boundary


def main():
    print("Single-level cascade (alpha=1, levels=1) -- pure DSI:")
    exp1 = CascadeDSIExplorer(R=1.0, alpha_0=1.0, levels=1)
    summ1 = exp1.summary(r_min=1.05, r_max=1e5)
    print(f"  zeros: {summ1.n_zeros}")
    print(f"  box dimension: {summ1.box_dimension:.4f}")
    print(f"  zero-ratio mean: {summ1.geometric_progression_ratio_mean:.4f}  "
          f"(analytic exp(pi)={3.141592653589793:.4f}->23.1407)")
    print()

    print("Multi-level cascade (levels=4, sf=2.5, ad=0.6) -- fractal:")
    exp4 = CascadeDSIExplorer(
        R=1.0, alpha_0=0.8, levels=4, scale_factor=2.5, amp_decay=0.6,
    )
    summ4 = exp4.summary(r_min=1.05, r_max=1e5)
    print(f"  zeros: {summ4.n_zeros}")
    print(f"  box dimension: {summ4.box_dimension:.4f}")
    print(f"  is geometric: {summ4.is_geometric_progression}")
    print()

    print("Phase boundary scan (tiny 3x3):")
    import numpy as np
    rep = scan_phase_boundary(
        scale_factors=np.array([2.0, 3.0, 4.0]),
        amp_decays=np.array([0.5, 0.7, 0.9]),
        r_min=1.05, r_max=1e4, radii=(4, 8, 16),
    )
    print(f"  verdict: {rep.verdict}")
    print(f"  max lambda_2 jump: {rep.max_lambda_2_jump:.3f}")
    print(f"  zero count grid:")
    for row in rep.n_zeros_grid:
        print(f"    {row.tolist()}")


if __name__ == "__main__":
    main()
