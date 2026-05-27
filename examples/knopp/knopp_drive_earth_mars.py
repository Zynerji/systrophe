"""Knopp Drive Earth-Mars journey demonstration.

Concrete engineering numbers for a Knopp Drive journey of length
equivalent to Earth-Mars closest approach (78 million km ~ 0.52 AU).

In our geometric units the LP-cylinder R sets the length scale; for a
demonstration we take R = 1 (so 1 unit ~ 1 AU equivalent) and scan
the parameter space to identify the optimal configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

from systrophe.knopp.knopp_drive import KnoppDriveConfig
from systrophe.knopp.knopp_traversal import (
    knopp_traversal,
    knopp_traversal_Q_sweep,
    summarise_traversal,
)


L_EARTH_MARS = 0.52  # AU at closest approach


def main() -> None:
    print("=" * 70)
    print("Knopp Drive Earth-Mars journey demonstration")
    print("=" * 70)
    print(f"Length L = {L_EARTH_MARS} (geometric units; assume 1 unit = 1 AU)")
    print()

    configs = [
        ("Baseline (Q=10)",           dict(Q=10.0, epsilon_horn=0.2)),
        ("High Q (Q=100)",            dict(Q=100.0, epsilon_horn=0.2)),
        ("Ultra Q (Q=500)",           dict(Q=500.0, epsilon_horn=0.2)),
        ("Aggressive horn (eps=0.7)", dict(Q=100.0, epsilon_horn=0.7)),
        ("Slow apparent (v_s=0.5)",   dict(Q=100.0, v_s=0.5)),
        ("Fast apparent (v_s=2.0)",   dict(Q=100.0, v_s=2.0)),
    ]

    journey_log = []
    for label, overrides in configs:
        cfg = KnoppDriveConfig(**overrides)
        rep = knopp_traversal(cfg, distance=L_EARTH_MARS, n_steps=120)
        print(f"[{label}]")
        print(f"  {summarise_traversal(rep)}")
        journey_log.append({"label": label, "overrides": overrides,
                             "coord_time": rep.coord_time_total,
                             "proper_time": rep.proper_time_total,
                             "exotic_matter_total": rep.exotic_matter_total,
                             "drive_power": rep.sustained_drive_power,
                             "total_energy": rep.total_energy_budget,
                             "inside_band_fraction": rep.inside_band_fraction,
                             "pf_ok": rep.pfenning_ford_compatible})
        print()

    print("=" * 70)
    print("Optimal Q for Earth-Mars distance (P-F bounded)")
    print("=" * 70)
    Q_sweep = knopp_traversal_Q_sweep(
        distance=L_EARTH_MARS, Q_range=(1.0, 2000.0), n_Q=30,
    )
    print(f"  Pfenning-Ford flip at Q ~ {Q_sweep['flip_Q']}")
    # Find best (lowest E_total, P-F-compatible)
    E_totals = Q_sweep["E_total_grid"]
    pf_fail = Q_sweep["pfenning_ford_failures"]
    best_E = float("inf")
    best_Q = None
    for q, e, fail in zip(Q_sweep["Q_grid"], E_totals, pf_fail):
        if not fail and e < best_E:
            best_E = e
            best_Q = q
    if best_Q is not None:
        print(f"  Optimal Q = {best_Q:.1f} -> E_total = {best_E:.3e}")
    else:
        print("  No P-F-compatible Q in range")

    out_path = Path(__file__).parent / "knopp_drive_earth_mars_results.json"
    out_path.write_text(json.dumps({
        "distance": L_EARTH_MARS,
        "configurations": journey_log,
        "Q_sweep": Q_sweep,
        "optimal_Q": best_Q,
        "optimal_E_total": best_E,
    }, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
