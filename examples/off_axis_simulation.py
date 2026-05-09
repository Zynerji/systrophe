"""Off-axis pair time-machine scan.

Runs the OffAxisPair on a 2D grid of points, identifies the CTC region,
and reports per-x-slice CTC band structure. Writes JSON output for the
whitepaper.

Run:
    python examples/off_axis_simulation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe import VanStockumInterior
from systrophe.off_axis import OffAxisPair


def main() -> dict:
    omega, R = 1.0, 1.0
    cyl = VanStockumInterior(omega=omega, R=R)
    separation = 4.0
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=separation)

    # 2D CTC map
    nx, ny = 121, 121
    m = pair.ctc_map_2d(
        x_min=-3.0, x_max=8.0, y_min=-4.0, y_max=4.0, nx=nx, ny=ny
    )
    ctc_fraction = float(m["is_ctc"].mean())
    print(f"Off-axis pair: a={omega*R}, separation={separation}R")
    print(f"  Grid {nx}x{ny} over x in [-3, 8], y in [-4, 4]")
    print(f"  CTC fraction: {ctc_fraction:.3f}")

    # Slice on y = 0 (line connecting the two axes): how does
    # g_{phi1 phi1} sign vary along the connecting line?
    j_zero = np.argmin(np.abs(m["y"]))
    g_along_axis = m["g_phiphi_cyl1"][j_zero, :]
    is_ctc_along_axis = (g_along_axis < 0).tolist()
    print(f"\nAlong y = 0 axis (connecting line): {sum(is_ctc_along_axis)} of {nx} samples in CTC")

    # Find CTC band intervals along y = 0
    bands = []
    in_band = False
    band_start = None
    for i, is_ctc in enumerate(is_ctc_along_axis):
        if is_ctc and not in_band:
            in_band = True
            band_start = m["x"][i]
        elif not is_ctc and in_band:
            in_band = False
            bands.append((float(band_start), float(m["x"][i - 1])))
    if in_band:
        bands.append((float(band_start), float(m["x"][-1])))
    print(f"CTC bands along y = 0: {len(bands)}")
    for k, (a, b) in enumerate(bands):
        print(f"  band {k+1}: x in [{a:.3f}, {b:.3f}]")

    # Write JSON output
    out_path = Path("examples") / "off_axis_simulation_results.json"
    results = {
        "omega": omega,
        "R": R,
        "separation": separation,
        "grid_nx": nx,
        "grid_ny": ny,
        "x_range": [-3.0, 8.0],
        "y_range": [-4.0, 4.0],
        "ctc_fraction": ctc_fraction,
        "y_zero_bands": [{"x_inner": a, "x_outer": b} for a, b in bands],
    }
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
