"""Cross-validation: apply the derivative catcher to the Knopp Drive
Kingston batch 7 data.

The least-squares smoothed-step fit found:
    r_edge_in  = 2.657 +- 0.002
    r_edge_out = 5.471 +- 0.010

This script asks: without any model assumption, does the derivative
catcher independently identify the same band edges directly from the
HW points?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.derivative_catcher import catch_smooth_transition


def main() -> None:
    path = Path(__file__).parent / "results" / "marrakesh_batch7_kingston_hw_analysis.json"
    data = json.loads(path.read_text())
    per = data["per_circuit"]
    rs = np.array([p["r"] for p in per])
    p_d1 = np.array([p["P_data1_observed"] for p in per])

    # Wrap as a callable
    def measured(r_val):
        idx = int(np.argmin(np.abs(rs - r_val)))
        return float(p_d1[idx])

    result = catch_smooth_transition(rs, measured, n_bits=32)

    print("=" * 70)
    print("Derivative-catcher validation on Knopp Drive Kingston batch 7")
    print("=" * 70)
    print()
    print(f"kind: {result['kind']}")
    print(f"value-level verdict: {result['value_scan'].verdict}")
    print(f"derivative verdict:  {result['derivative_scan'].verdict}")
    centre = result["estimated_transition_centre"]
    if centre is not None:
        print(f"estimated transition centre: r = {centre:.3f}")
    print()
    print("Comparison to least-squares fit:")
    print(f"  ls fit r_edge_in:  2.657 +- 0.002")
    print(f"  ls fit r_edge_out: 5.471 +- 0.010")
    print()
    print("Derivative-catcher sharp features:")
    for sf in result["derivative_scan"].sharp_features:
        print(f"  r = {sf['parameter_value']:.3f}, hamming_step = {sf['hamming_step']}, "
              f"median = {sf['median_step']:.1f}, excess = {sf['excess_over_median']:.1f}")
    print()
    print("Value-level sharp features:")
    for sf in result["value_scan"].sharp_features:
        print(f"  r = {sf['parameter_value']:.3f}, hamming_step = {sf['hamming_step']}, "
              f"median = {sf['median_step']:.1f}, excess = {sf['excess_over_median']:.1f}")

    out_path = path.parent / "knopp_derivative_catcher_validation.json"
    out_path.write_text(json.dumps({
        "kind": result["kind"],
        "estimated_transition_centre": centre,
        "value_verdict": result["value_scan"].verdict,
        "derivative_verdict": result["derivative_scan"].verdict,
        "ls_fit_r_edge_in":  2.657,
        "ls_fit_r_edge_in_sigma": 0.002,
        "ls_fit_r_edge_out": 5.471,
        "ls_fit_r_edge_out_sigma": 0.010,
        "value_sharp_features": [
            {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
             for k, v in sf.items()}
            for sf in result["value_scan"].sharp_features
        ],
        "derivative_sharp_features": [
            {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
             for k, v in sf.items()}
            for sf in result["derivative_scan"].sharp_features
        ],
    }, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
