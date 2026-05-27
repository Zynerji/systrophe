"""Phase 3a: SystropheArray beam-forming + extinction diagnostics.

Demonstrations:

1. N-fold topological extinction (uniform-phase comb) verified for N=2..8.
2. Beam-steering: place an L-node at a chosen r_target via analytic delta_aim.
3. Dirichlet-kernel array factor for linear-ramp phasing.
4. Side-lobe level of an unsteered N-cylinder pair.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

from systrophe import SystropheArray
from systrophe.geometry.vanstockum import VanStockumInterior


def main() -> dict:
    t_start = time.time()
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    alpha = cyl.alpha

    # 1. N-fold extinction sweep
    extinction_rows = []
    for N in (2, 3, 4, 5, 6, 7, 8):
        comb = SystropheArray.uniform_phase_comb(cyl, N=N)
        ext = comb.extinction_check(r_max=20.0, n_grid=2001)
        extinction_rows.append({
            "N": N,
            "max_array_factor": ext["max_array_factor"],
            "is_extinguished": ext["is_extinguished"],
        })

    # 2. Beam-steered nodes
    steer_rows = []
    for r_target in (2.0, 3.0, 4.0, 5.0, 7.0):
        arr = SystropheArray.beam_steer(r_target=r_target, cylinder=cyl, N=2)
        L_at_target = float(arr.L(np.array([r_target]))[0])
        # Also probe at nearby radii
        L_near = float(arr.L(np.array([r_target * 1.05]))[0])
        steer_rows.append({
            "r_target": float(r_target),
            "delta_aim": float(arr.sinusoids[1].delta - arr.sinusoids[0].delta),
            "L_at_target": L_at_target,
            "L_near": L_near,
        })

    # 3. Dirichlet-pattern array factor for linear-ramp phasing
    N_dir = 5
    delta_step = math.pi / 3
    arr_dir = SystropheArray.from_cylinders(
        [cyl] * N_dir, offsets=tuple(i * delta_step for i in range(N_dir))
    )
    r_grid = np.geomspace(1.05, 20.0, 200)
    dirichlet = arr_dir.dirichlet_pattern(r_grid, delta_step=delta_step)
    main_lobe_idx = int(np.argmax(dirichlet))
    dirichlet_summary = {
        "N": N_dir,
        "delta_step_over_pi": float(delta_step / math.pi),
        "main_lobe_r": float(r_grid[main_lobe_idx]),
        "main_lobe_value": float(dirichlet[main_lobe_idx]),
        "min_pattern_value": float(np.min(dirichlet)),
    }

    # 4. Side-lobe level diagnostics for several N
    sidelobe_rows = []
    for N in (2, 3, 4, 5, 6):
        arr_N = SystropheArray.from_cylinders(
            [cyl] * N,
            offsets=tuple(i * (math.pi / N) for i in range(N)),
        )
        sll = arr_N.array_factor_sidelobe_level(r_max=50.0, n_grid=4001)
        sidelobe_rows.append({"N": N, "sidelobe_level": float(sll),
                                "sidelobe_dB": float(20.0 * math.log10(max(sll, 1e-15)))})

    out = {
        "phase": "3a",
        "title": "SystropheArray beam-forming + extinction diagnostics",
        "single_cylinder": {"omega": cyl.omega, "R": cyl.R, "alpha": alpha},
        "extinction_sweep": extinction_rows,
        "beam_steering": steer_rows,
        "dirichlet_pattern": dirichlet_summary,
        "sidelobe_level_sweep": sidelobe_rows,
        "elapsed_seconds": time.time() - t_start,
    }

    out_path = pathlib.Path(__file__).with_name("phase_3a_beam_steering_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 70)
    print("Phase 3a: SystropheArray beam-forming + extinction")
    print("=" * 70)
    print("N-fold topological extinction (uniform-phase comb):")
    print("  N    max|A_factor|    extinguished?")
    for r in extinction_rows:
        print(f"  {r['N']}    {r['max_array_factor']:.3e}     {r['is_extinguished']}")
    print()
    print("Beam-steering (place L = 0 at r_target):")
    print("  r_target  delta_aim/pi    L(r_target)      L(r_target * 1.05)")
    for r in steer_rows:
        print(f"  {r['r_target']:.2f}      {r['delta_aim']/math.pi:+.4f}        "
              f"{r['L_at_target']:+.3e}     {r['L_near']:+.3e}")
    print()
    print(f"Dirichlet (linear-ramp delta_step = pi/3, N={N_dir}):")
    print(f"  main lobe at r = {dirichlet_summary['main_lobe_r']:.3f}, "
          f"value {dirichlet_summary['main_lobe_value']:.3f}")
    print(f"  min pattern value: {dirichlet_summary['min_pattern_value']:.3e}")
    print()
    print("Sidelobe level vs N:")
    print("  N    SLL (linear)   SLL (dB)")
    for r in sidelobe_rows:
        print(f"  {r['N']}    {r['sidelobe_level']:.4f}        {r['sidelobe_dB']:+.2f} dB")
    print()
    print(f"Elapsed: {out['elapsed_seconds']:.2f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
