"""Phase 3b: OffAxisPair quantitative orbits + topology.

Demonstrations:

1. 2D ergosurface map for an off-axis pair.
2. CTC region topology classifier (number of components, holes).
3. Joint trace anomaly along a probe line through the pair.
4. Timelike geodesic-completeness test (do test particles escape?).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from systrophe.off_axis import OffAxisPair
from systrophe.vanstockum import VanStockumInterior


def main() -> dict:
    t_start = time.time()
    c1 = VanStockumInterior(omega=1.0, R=1.0)
    c2 = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(c1, c2, separation=3.0)

    # 1. Ergosurface map
    erg = pair.ergosurface_2d(-3.0, 6.0, -3.0, 3.0, nx=81, ny=41)
    ergo_fraction = float(np.mean(erg["is_ergoregion"]))

    # 2. CTC region topology (multiple resolutions)
    topology_rows = []
    for nx, ny in ((41, 21), (81, 41), (121, 61)):
        topo = pair.ctc_region_topology(-3.0, 6.0, -3.0, 3.0, nx=nx, ny=ny)
        topology_rows.append({
            "nx": nx, "ny": ny,
            "n_components": topo["n_components"],
            "n_holes": topo["n_holes"],
            "ctc_fraction": topo["ctc_fraction"],
            "component_areas": topo["component_areas"],
            "topology_summary": topo["topology_summary"],
        })

    # 3. Trace anomaly along a probe line through the pair
    anomaly_table = []
    x_probe = 1.5  # between the two cylinders
    for y in np.linspace(-2.5, 2.5, 11):
        anomaly_table.append({
            "x": float(x_probe), "y": float(y),
            "trace_anomaly_2d_sector": pair.trace_anomaly_2d_sector(x_probe, float(y)),
        })

    # 4. Geodesic completeness from several initial conditions
    starts = [(5.0, 0.5), (-2.0, 1.5), (4.0, -1.0), (1.5, 2.0)]
    x_starts = tuple(s[0] for s in starts)
    y_starts = tuple(s[1] for s in starts)
    geo_results = pair.geodesic_completeness_test(
        x_starts=x_starts, y_starts=y_starts,
        vx0=0.05, vy0=0.05, t_max=30.0, n_samples=251,
    )

    out = {
        "phase": "3b",
        "title": "OffAxisPair quantitative orbits + topology",
        "spacetime": {
            "cyl1": {"omega": 1.0, "R": 1.0},
            "cyl2": {"omega": 1.0, "R": 1.0},
            "separation": 3.0,
        },
        "ergoregion_fraction": ergo_fraction,
        "topology_resolution_sweep": topology_rows,
        "trace_anomaly_along_probe_line": anomaly_table,
        "geodesic_completeness": geo_results,
        "elapsed_seconds": time.time() - t_start,
    }

    out_path = pathlib.Path(__file__).with_name("phase_3b_off_axis_topology_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 72)
    print("Phase 3b: OffAxisPair (separation = 3.0)")
    print("=" * 72)
    print(f"Ergoregion grid fraction (g_tt > 0): {ergo_fraction:.4f}")
    print()
    print("CTC region topology vs grid resolution:")
    print("  nx x ny      n_comp  n_holes  ctc_frac  summary")
    for r in topology_rows:
        print(f"  {r['nx']:3d} x {r['ny']:2d}    {r['n_components']:3d}     "
              f"{r['n_holes']:3d}      {r['ctc_fraction']:.4f}    "
              f"{r['topology_summary']}")
    print()
    print(f"Trace anomaly along x = {x_probe} (sample):")
    print("  y      trace_anomaly")
    for row in anomaly_table[::2]:
        print(f"  {row['y']:+.3f}  {row['trace_anomaly_2d_sector']:+.4e}")
    print()
    print("Geodesic completeness test (timelike, t_max = 30, escape_radius = 100):")
    print("  (x0, y0)         reaches_escape  enters_ctc  final_r")
    for r in geo_results:
        print(f"  ({r['x0']:+.1f}, {r['y0']:+.1f})       "
              f"{str(r['reaches_escape']):<6s}         "
              f"{str(r['enters_ctc']):<6s}      "
              f"{r['final_radius']:.3f}")
    print()
    print(f"Elapsed: {out['elapsed_seconds']:.1f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
