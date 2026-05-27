"""Calibration-aware HW analysis: auto-adjust observed metrics
by chip calibration data.

Inspired by the Tricameral.ai auto-calibration pattern: before
treating a HW counts file as a raw observation, normalise it by
the chip's per-shot expected noise floor (derived from the live
calibration snapshot). This produces a calibration-NORMALISED
TV distance that is fair to compare across chips with different
fidelity baselines (Marrakesh, Fez, Kingston).

Two functions
=============

1. `expected_tv_from_calibration(snapshot, isa_depth, n_2q)` —
   model the chip's intrinsic TV distance for a circuit of given
   depth and 2-qubit-gate count. Linear-in-error-rates model
   (valid in the low-error regime):

       TV_exp ~ 0.5 * (n_2q * cz_err + isa_depth * sx_err + 2 * readout_err)

2. `calibration_aware_tv_normalisation(observed_tv, snapshot, ...)` —
   ratio of observed to predicted TV. Values near 1 mean the chip
   is performing at its calibrated noise floor; values >> 1 mean
   the circuit has additional systematic error beyond standard
   sources; values << 1 mean we under-predicted (which usually
   indicates a calibration model gap).

Cross-chip cataloguing
======================

`compare_cross_chip(observations, calibrations)` builds the
per-chip normalisation table and runs the catcher on the
NORMALISED TV distances rather than the raw ones. This makes
the cross-chip verdict insensitive to chip-specific noise-floor
differences, surfacing only deviations that the calibration
data cannot explain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.catchers.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)


def median_q_param(snapshot: dict, key: str) -> float | None:
    """Re-implementation of calibration_snapshot.median_q_param for
    decoupled imports."""
    vals = [q.get(key) for q in snapshot["qubit_properties"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sorted(vals)[len(vals) // 2])


def median_gate_error(snapshot: dict, gate: str) -> float | None:
    gate_errors = snapshot["instruction_errors"].get(gate, {})
    vals = [e["error"] for e in gate_errors.values()
            if e.get("error") is not None]
    if not vals:
        return None
    return float(sorted(vals)[len(vals) // 2])


def expected_tv_from_calibration(
    snapshot: dict,
    isa_depth: int = 50,
    n_2q: int = 12,
    n_measured: int = 4,
) -> float:
    """Linear-in-error-rate prediction of intrinsic TV distance.

    For a circuit of given ISA depth and 2-qubit-gate count, on a
    chip with given median calibration parameters, the dominant
    contribution to the observed TV vs. ideal is

        TV_exp ~ 0.5 * (n_2q * cz_err + isa_depth * sx_err
                         + n_measured * readout_err)

    This is the simple linear noise budget; cross-talk, leakage,
    and SPAM effects are not included.
    """
    sx_err = median_gate_error(snapshot, "sx") or 5e-4
    cz_err = median_gate_error(snapshot, "cz") or 3e-3
    readout_err = median_gate_error(snapshot, "measure") or 1.5e-2
    tv = 0.5 * (
        n_2q * cz_err
        + isa_depth * sx_err
        + n_measured * readout_err
    )
    return float(tv)


def calibration_aware_tv_normalisation(
    observed_tv: float, snapshot: dict,
    isa_depth: int = 50, n_2q: int = 12, n_measured: int = 4,
) -> dict:
    """Return the calibration-normalised TV ratio."""
    expected = expected_tv_from_calibration(
        snapshot, isa_depth=isa_depth,
        n_2q=n_2q, n_measured=n_measured,
    )
    ratio = observed_tv / max(expected, 1e-12)
    verdict = (
        "expected" if 0.5 <= ratio <= 2.0
        else ("anomalously_low" if ratio < 0.5
              else "anomalously_high")
    )
    return {
        "observed_tv": float(observed_tv),
        "expected_tv": float(expected),
        "ratio": float(ratio),
        "verdict": verdict,
        "calibration_median": {
            "T1_us": (median_q_param(snapshot, "T1") or 0) * 1e6,
            "T2_us": (median_q_param(snapshot, "T2") or 0) * 1e6,
            "sx_err": median_gate_error(snapshot, "sx"),
            "cz_err": median_gate_error(snapshot, "cz"),
            "readout_err": median_gate_error(snapshot, "measure"),
        },
    }


def compare_cross_chip(
    observations: dict[str, list[float]],
    calibrations: dict[str, dict],
    isa_depth: int = 50, n_2q: int = 12, n_measured: int = 4,
) -> dict:
    """Compare normalised TV distances across chips.

    `observations` is a dict {chip_name: list of TV values across circuits}.
    `calibrations` is a dict {chip_name: snapshot}.

    Returns a per-chip normalised-ratio profile and a catcher verdict
    on the normalised array.
    """
    per_chip_normalised = {}
    expected_per_chip = {}
    ratios_per_chip = {}
    for chip, tvs in observations.items():
        snap = calibrations.get(chip)
        if snap is None:
            per_chip_normalised[chip] = None
            continue
        expected = expected_tv_from_calibration(
            snap, isa_depth=isa_depth, n_2q=n_2q,
            n_measured=n_measured,
        )
        normalised = [tv / max(expected, 1e-12) for tv in tvs]
        per_chip_normalised[chip] = normalised
        expected_per_chip[chip] = expected
        ratios_per_chip[chip] = float(np.mean(normalised))

    # Per-quantity catcher: same observable (normalised TV) across chips
    per_q = {
        "normalised_tv": {
            chip: np.array(arr)
            for chip, arr in per_chip_normalised.items()
            if arr is not None
        }
    }
    if len(per_q["normalised_tv"]) < 2:
        return {
            "per_chip_normalised": per_chip_normalised,
            "expected_per_chip": expected_per_chip,
            "mean_ratio_per_chip": ratios_per_chip,
            "catcher_verdict": "insufficient_chips",
            "n_sharp": 0,
        }
    nov = catch_novelty_per_quantity(per_q, n_bins=32)
    return {
        "per_chip_normalised": per_chip_normalised,
        "expected_per_chip": expected_per_chip,
        "mean_ratio_per_chip": ratios_per_chip,
        "catcher_verdict": nov["aggregate_verdict"],
        "n_sharp": sum(
            len(r.get("sharp_features", []))
            for r in nov["per_quantity"].values()
        ),
        "per_quantity_detail": nov["per_quantity"],
    }


def load_latest_calibration(
    chip: str, results_dir: Path | None = None,
) -> dict | None:
    """Load the most recent calibration snapshot for a chip."""
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    candidates = sorted(
        results_dir.glob(f"calibration_{chip}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


if __name__ == "__main__":
    # Demo: compare expected TV across the 3 chips for the batch 6 circuit
    chips = ("ibm_marrakesh", "ibm_fez", "ibm_kingston")
    print("=== Calibration-aware TV prediction for batch 6 circuit ===")
    print("(isa_depth=51, n_2q=12, n_measured=4)")
    print()
    for chip in chips:
        snap = load_latest_calibration(chip)
        if snap is None:
            print(f"  {chip:18s}: no calibration data")
            continue
        exp = expected_tv_from_calibration(
            snap, isa_depth=51, n_2q=12, n_measured=4,
        )
        T1 = (median_q_param(snap, "T1") or 0) * 1e6
        cz = median_gate_error(snap, "cz") or 0
        rdo = median_gate_error(snap, "measure") or 0
        print(f"  {chip:18s}: T1={T1:6.1f}us, cz_err={cz:.2e}, "
              f"readout={rdo:.2e}  ->  expected_TV={exp:.4f}")
