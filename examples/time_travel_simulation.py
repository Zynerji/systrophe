"""Time-travel simulation using a Systrophe pair as the time-machine harness.

This script runs a structured numerical experiment:

1. Identify CTC bands of a single supercritical van Stockum cylinder.
2. Show the timelike-Omega sector at the deepest point of each band.
3. Construct a backward-time-travel orbit (Delta t < 0 per revolution),
   compute coordinate vs. proper time over multiple revolutions.
4. Construct a "near-instantaneous" orbit (Omega large, |Delta t| small)
   and compare.
5. Use a SystrophePair to demonstrate that the relative phase between
   two co-rotating cylinders shifts the available CTC bands and hence
   tunes the time-travel windows.
6. Sweep the offset and report total CTC log-measure as a function of
   offset (the "off-set Tipler sinusoid" interference pattern).

All numbers in this script are reproduced verbatim in the whitepaper.

Run:
    python examples/time_travel_simulation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe import (
    SystrophePair,
    VanStockumInterior,
    find_single_cylinder_windows,
    harness_time_loop,
    timelike_omega_bounds,
)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> dict:
    """Run the simulation. Returns a dict of results for the whitepaper."""
    results: dict = {}

    # -------- Single-cylinder time machine --------
    omega, R = 1.0, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    section(f"Single van Stockum cylinder (omega = {omega}, R = {R}, a = {vs.a})")
    print(f"Tipler log-frequency  alpha = sqrt(4 a^2 - 1)         = {vs.alpha:.6f}")
    print(f"Log-period            2 pi / alpha                    = {2*np.pi/vs.alpha:.6f}")
    print(f"r-period multiplier   exp(2 pi / alpha)               = {np.exp(2*np.pi/vs.alpha):.6f}")

    windows = find_single_cylinder_windows(vs, r_min=1.001, r_max=200.0)
    print(f"\nCTC bands found in r in [1.001, 200]: {len(windows)}")
    band_dicts = []
    for i, w in enumerate(windows):
        d = {
            "index": i + 1,
            "r_inner": w.r_inner,
            "r_outer": w.r_outer,
            "log_span": w.log_span(),
            "r_min_L": w.r_min_L,
            "L_min": w.L_min,
            "F_at_min": w.F_at_min,
            "K_at_min": w.K_at_min,
            "omega_lower_spacelike": w.omega_bounds_at_min[0],
            "omega_upper_spacelike": w.omega_bounds_at_min[1],
        }
        band_dicts.append(d)
        print(f"  Band {i+1}: r in [{w.r_inner:.4f}, {w.r_outer:.4f}], log span = {w.log_span():.4f}")
        print(f"    deepest L = {w.L_min:.4f} at r = {w.r_min_L:.4f}")
        print(f"    F(r_min_L) = {w.F_at_min:.4f}, K(r_min_L) = {w.K_at_min:.4f}")
        print(
            f"    timelike Omega: < {w.omega_bounds_at_min[0]:.4f}  or  "
            f"> {w.omega_bounds_at_min[1]:.4f}"
        )
    results["single_cylinder"] = {
        "omega": omega,
        "R": R,
        "a": vs.a,
        "alpha": vs.alpha,
        "log_period": 2 * np.pi / vs.alpha,
        "r_period_multiplier": np.exp(2 * np.pi / vs.alpha),
        "n_ctc_bands": len(windows),
        "bands": band_dicts,
    }

    # -------- Backward time travel orbit --------
    section("Backward time-travel orbit (Delta t < 0 per revolution)")
    w = windows[0]
    targets = [-0.1, -0.5, -1.0, -2.0, -5.0]
    backward = []
    for tgt in targets:
        try:
            res = harness_time_loop(w, target_dt_per_rev=tgt, n_revolutions=10)
            backward.append({
                "target_dt": tgt,
                "Omega": res["Omega"],
                "is_timelike": res["is_timelike"],
                "dtau_per_rev": res["dtau_per_revolution"],
                "total_coord_advance_10rev": res["total_coord_time_advance"],
                "total_proper_advance_10rev": res["total_proper_time_advance"],
            })
            print(
                f"  target dt/rev = {tgt:+.2f}  ->  Omega = {res['Omega']:+.4f},  "
                f"dtau/rev = {res['dtau_per_revolution']:.4f},  "
                f"after 10 revs:  dt = {res['total_coord_time_advance']:+.2f},  "
                f"dtau = {res['total_proper_time_advance']:.2f}"
            )
        except ValueError as e:
            print(f"  target dt/rev = {tgt:+.2f}  ->  REJECTED: {e}")
            backward.append({"target_dt": tgt, "rejected": str(e)})
    results["backward_orbits"] = backward

    # -------- Near-instantaneous orbit (Omega large) --------
    section("Near-instantaneous loop (Omega large, |Delta t| small)")
    Omega_large = 100.0  # well outside the spacelike bounds
    orbit = w.orbit_at_r_min(Omega_large)
    print(f"  r = {orbit.r:.4f}, Omega = {Omega_large},  is_timelike = {orbit.is_timelike}")
    print(
        f"  dt per revolution = 2 pi / Omega = {orbit.coord_dt_per_revolution:.6f}"
    )
    print(f"  dtau per revolution = {orbit.proper_dtau_per_revolution:.4f}")
    print(
        f"  ratio dtau / |dt| = sqrt(F - 2 K Omega - L Omega^2) "
        f"= {orbit.proper_dtau_per_revolution / abs(orbit.coord_dt_per_revolution):.4f}"
    )
    results["near_instantaneous"] = {
        "Omega": Omega_large,
        "is_timelike": orbit.is_timelike,
        "dt_per_rev": orbit.coord_dt_per_revolution,
        "dtau_per_rev": orbit.proper_dtau_per_revolution,
    }

    # -------- Pair offset sweep --------
    section("SystrophePair offset sweep (engineering tunable CTC bands)")
    cyl_a = VanStockumInterior(omega=1.5, R=1.0)
    cyl_b = VanStockumInterior(omega=1.5, R=1.0)
    pair = SystrophePair.from_cylinders(cyl_a, cyl_b, delta_offset=0.0)

    offsets = np.linspace(0.0, 2 * np.pi, 25)
    sweep = pair.offset_sweep(r_min=1.05, r_max=20.0, offsets=offsets)
    print(f"{'offset (rad)':>14}  {'#bands':>7}  {'log-measure':>12}")
    sweep_rows = []
    for o, n, m in zip(offsets, sweep["n_bands"], sweep["log_measures"]):
        print(f"{o:14.4f}  {int(n):7d}  {m:12.4f}")
        sweep_rows.append({"offset": float(o), "n_bands": int(n), "log_measure": float(m)})
    results["pair_offset_sweep"] = {
        "omega": 1.5,
        "R": 1.0,
        "rows": sweep_rows,
    }

    # -------- Pair-tuned r_inner shift demo --------
    section("Pair-tuned CTC inner radius (off-set sinusoid signature)")
    print(
        f"{'offset (rad)':>14}  {'r_inner':>10}  {'r_outer':>10}  {'log span':>10}"
    )
    pair_tuned = []
    for delta_off in np.linspace(0.0, np.pi, 7):
        p = SystrophePair.from_cylinders(cyl_a, cyl_b, delta_offset=float(delta_off))
        bands = p.ctc_bands(r_min=1.05, r_max=20.0)
        if bands:
            r_in, r_out = bands[0]
            print(
                f"{delta_off:14.4f}  {r_in:10.4f}  {r_out:10.4f}  "
                f"{np.log(r_out/r_in):10.4f}"
            )
            pair_tuned.append({
                "offset": float(delta_off),
                "r_inner": float(r_in),
                "r_outer": float(r_out),
                "log_span": float(np.log(r_out / r_in)),
            })
        else:
            print(f"{delta_off:14.4f}  no bands")
            pair_tuned.append({"offset": float(delta_off), "r_inner": None})
    results["pair_tuned_first_band"] = pair_tuned

    # -------- Save results for the whitepaper --------
    out_path = Path("examples") / "time_travel_simulation_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    section(f"Results written to {out_path}")
    return results


if __name__ == "__main__":
    main()
