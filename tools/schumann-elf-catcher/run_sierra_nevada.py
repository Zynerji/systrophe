"""Full ELF catcher-stack run on one Sierra Nevada hourly recording.

Usage:
    python run_sierra_nevada.py <data_file> [--surrogates N] [--json out.json]

Runs all four targets and prints an honest report:
  1. transient bursts (chunked burst catcher) + phase-randomized null
  2. slow amplitude trend (growth catcher, with permutation null)
  3. spectral regime change-points (novelty catcher)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schumann_elf_catcher import (  # noqa: E402
    load_sierra_nevada, burst_null_test, trend_scan, regime_change_scan,
    SCHUMANN_MODES,
)
from systrophe.catchers.growth_catcher import summarize_growth_for_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--info", default=None)
    ap.add_argument("--surrogates", type=int, default=60)
    ap.add_argument("--chunk-s", type=float, default=60.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rec = load_sierra_nevada(args.data_file, info_path=args.info)
    sensor_name = {"0": "NS", "1": "EW"}.get(rec.sensor, rec.sensor)
    print("=" * 70)
    print(f"Sierra Nevada ELF  sensor={sensor_name}  fs={rec.sample_rate:.3f} Hz")
    print(f"t0={rec.t0_utc}  duration={rec.duration_s:.0f}s")
    print("=" * 70)

    # 1. Transient bursts + surrogate null --------------------------------
    print("\n[1] TRANSIENT BURST SCAN  (chunked address-space catcher)")
    res = burst_null_test(
        rec.samples, rec.sample_rate,
        n_surrogates=args.surrogates, chunk_s=args.chunk_s,
    )
    print(f"    chunks={res.n_chunks}  band={res.f_band[0]:.0f}-{res.f_band[1]:.0f} Hz")
    print(f"    largest Hamming step = {res.max_hamming_step} @ t={res.max_step_time_s:.1f}s")
    print(f"    surrogate null (n={args.surrogates}): 95th pct = "
          f"{res.null_threshold_95:.1f}, p-value = {res.null_p_value:.4f}")
    verdict = ("REAL transient structure (p<0.05, exceeds PSD-matched null)"
               if (res.null_p_value or 1.0) < 0.05
               else "consistent with spectral-shape artifact (null not exceeded)")
    print(f"    -> {verdict}")
    print("    top candidates (time_s, hamming_step):")
    for b in res.candidates[:8]:
        print(f"       t={b.time_s:8.1f}s  step={b.hamming_step}")

    # 2. Slow amplitude trend --------------------------------------------
    print("\n[2] SLOW AMPLITUDE TREND  (growth catcher, full ELF band)")
    g, (t, rms) = trend_scan(rec.samples, rec.sample_rate)
    print("    " + summarize_growth_for_report(g))

    # 3. Regime change-points --------------------------------------------
    print("\n[3] SPECTRAL REGIME CHANGE-POINTS  (novelty catcher)")
    v, cps, _ = regime_change_scan(rec.samples, rec.sample_rate)
    print(f"    verdict='{v}'  n_change_points={len(cps)}")
    for cp in cps[:8]:
        print(f"       t={cp['time_s']:8.1f}s  step={cp['hamming_step']}")

    if args.json:
        out = {
            "source": os.path.basename(args.data_file),
            "sensor": sensor_name, "sample_rate": rec.sample_rate,
            "t0_utc": rec.t0_utc, "duration_s": rec.duration_s,
            "schumann_modes_hz": list(SCHUMANN_MODES),
            "burst": {
                "max_hamming_step": res.max_hamming_step,
                "max_step_time_s": res.max_step_time_s,
                "null_p_value": res.null_p_value,
                "null_threshold_95": res.null_threshold_95,
                "n_surrogates": args.surrogates,
                "top_candidates": [
                    {"time_s": b.time_s, "hamming_step": b.hamming_step}
                    for b in res.candidates[:20]
                ],
            },
            "trend": {
                "verdict": g.verdict, "growth_exponent": g.growth_exponent,
                "z": g.significance_z, "p_value": g.p_value,
                "r_squared": g.r_squared,
            },
            "regime": {"verdict": v, "change_points": cps},
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
