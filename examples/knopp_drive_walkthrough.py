"""Knopp Drive walkthrough: composite warp engineering bound.

Demonstrates the four-mechanism composite (Tipler CTC-band gating,
Krasnikov tube embedding, Q-feedback amplification, horn-toroidal
steering) at three radial regimes.
"""

from __future__ import annotations

import json
from pathlib import Path

from systrophe.knopp_drive import (
    knopp_budget,
    knopp_drive_inside_band,
    novelty_scan,
    summarise_knopp_budget,
)


def main() -> None:
    print("=" * 70)
    print("Knopp Drive walkthrough")
    print("=" * 70)
    print()

    configs = [
        ("INSIDE first CTC band", dict(r_orbit=1.5, Q=10.0, epsilon_horn=0.2)),
        ("INSIDE band, Q=100",    dict(r_orbit=1.5, Q=100.0, epsilon_horn=0.2)),
        ("OUTSIDE band, mid r",   dict(r_orbit=3.0, Q=10.0, epsilon_horn=0.2)),
        ("OUTSIDE band, far r",   dict(r_orbit=6.0, Q=10.0, epsilon_horn=0.2)),
        ("Steering off (eps=0)",  dict(r_orbit=3.0, Q=10.0, epsilon_horn=0.0)),
        ("Steering strong (eps=0.7)", dict(r_orbit=3.0, Q=10.0, epsilon_horn=0.7)),
    ]
    summaries = []
    for label, overrides in configs:
        b = knopp_budget(**overrides)
        inside = knopp_drive_inside_band(r_orbit=overrides["r_orbit"])
        s = summarise_knopp_budget(b)
        print(f"[{label}] inside_band={inside}")
        print(f"  {s}")
        print()
        summaries.append({"label": label, "overrides": overrides,
                          "inside_band": inside, "summary": s,
                          "composite_E_neg": b.composite_E_neg,
                          "drive_power": b.sustained_drive_power,
                          "steering_mag": b.steering_magnitude,
                          "pf_ok": b.pfenning_ford_compatible})

    print("Running novelty catcher across (r, Q, eps) cube...")
    nov = novelty_scan(
        r_orbit_range=(1.05, 12.0), n_r=30,
        Q_range=(2.0, 50.0), n_Q=8,
        epsilon_range=(0.0, 0.8), n_eps=8,
    )
    print(f"Novelty catcher: verdict='{nov['novelty_verdict']}', "
          f"n_sharp={nov['novelty_n_sharp']}")
    for sf in nov["novelty_sharp_features"][:5]:
        print(f"  sharp at r={sf.get('parameter_value'):.3f}  "
              f"step={sf.get('hamming_step')}  median={sf.get('median_step')}")

    out_path = Path(__file__).parent / "knopp_drive_walkthrough_results.json"
    out_path.write_text(json.dumps({
        "configurations": summaries,
        "novelty_catcher": {
            "verdict": nov["novelty_verdict"],
            "sharp_features": nov["novelty_sharp_features"],
        },
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
