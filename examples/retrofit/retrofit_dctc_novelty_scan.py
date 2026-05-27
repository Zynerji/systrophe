"""Retrofit pass: address-space novelty catcher applied to every
existing DCTC trilogy result file.

The DCTC trilogy (papers A/B/C) and its 23 deep-phase scripts predate
the always-on novelty-catcher rule (commit c6c81dc). This script does
NOT re-run the DCTC scripts (their RNG is fixed inside the scripts);
it loads each `dctc_*_results.json` and feeds the natural distributions
to `catch_novelty_in_distributions`, surfacing any sharp address-space
features that the original analyses might have missed.

For files whose contents are pure scalar summaries (no extractable
distributions / arrays), the entry records `unscannable` so the
follow-up in-place patch step knows to add a native catcher call to
the source script.

Output: `examples/retrofit_dctc_novelty_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from systrophe.catchers.novelty_catcher import catch_novelty_in_distributions

EXAMPLES_DIR = Path(__file__).parent
OUT_PATH = EXAMPLES_DIR / "retrofit_dctc_novelty_results.json"


# ---------------------------------------------------------------------
# Extractors: each returns (labels, distributions) or None if not
# scannable. Distributions become inputs to the catcher.
# ---------------------------------------------------------------------

def _hist(arr, n_bins: int = 32) -> np.ndarray:
    """Empirical histogram normalised to a probability vector.

    Falls back gracefully when the input has zero range (constant
    column): returns a uniform mass at the single bin.
    """
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros(n_bins)
    lo, hi = float(a.min()), float(a.max())
    # Guard against ranges too narrow for numpy to subdivide into n_bins
    # distinct finite-width bins (it raises on extreme-eigenvalue cases
    # like lambda_1 ∈ [0.99999…7, 1.00000…22]).
    span = hi - lo
    if span <= n_bins * np.finfo(float).eps * max(abs(lo), abs(hi), 1.0):
        h = np.zeros(n_bins)
        h[0] = a.size
    else:
        h, _ = np.histogram(a, bins=n_bins, range=(lo, hi))
    s = h.sum()
    return h.astype(float) / s if s > 0 else h.astype(float)


def extract_haar_ensemble_per_pair(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_a, phase_b: dict-keyed-by-(dim_pair). Each value carries
    lambda_2 / iter / purity sample arrays. One distribution per pair,
    concatenated across the available sample lists."""
    labels, dists = [], []
    for key, val in d.items():
        if not isinstance(val, dict):
            continue
        feats = []
        for name in ("lambda_2", "iter", "purity"):
            if name in val and isinstance(val[name], list) and val[name]:
                feats.append(_hist(val[name]))
        if feats:
            labels.append(key)
            dists.append(np.concatenate(feats))
    return labels, dists


def extract_phase_c_top_bottom(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_c: 'top_20' / 'bottom_20' lists of records with purity /
    lambda_2 / schmidt_U / iter / rank fields. Build one distribution
    per group from each numeric field that exists."""
    labels, dists = [], []
    for group in ("top_20", "bottom_20"):
        recs = d.get(group)
        if not isinstance(recs, list) or not recs:
            continue
        feats = []
        sample = recs[0]
        for k in sample:
            try:
                vec = np.asarray([float(r[k]) for r in recs if k in r], dtype=float)
            except (TypeError, ValueError):
                continue
            if vec.size > 1 and np.all(np.isfinite(vec)):
                feats.append(_hist(vec, n_bins=16))
        if feats:
            labels.append(group)
            dists.append(np.concatenate(feats))
    return labels, dists


def extract_cycle_distributions(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """E1_cycles: haar_cycle_distribution + clifford_cycle_distribution
    dicts {period_label: count}. Each becomes one probability vector."""
    labels, dists = [], []
    for k in ("haar_cycle_distribution", "clifford_cycle_distribution"):
        v = d.get(k)
        if isinstance(v, dict) and v:
            keys = sorted(v.keys(), key=lambda x: (int(x) if str(x).lstrip("-").isdigit() else 9999))
            counts = np.array([float(v[kk]) for kk in keys], dtype=float)
            s = counts.sum()
            if s > 0:
                labels.append(k)
                dists.append(counts / s)
    return labels, dists


def extract_delta_scan(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_am family: 'deltas' + per-quantity arrays. Build one
    distribution per quantity, hashed as a histogram of the scan."""
    labels, dists = [], []
    for k in ("physical_ctcs", "dctc_amplifications", "amplifications",
              "fixed_point_purities", "purities"):
        v = d.get(k)
        if isinstance(v, list) and len(v) > 1:
            labels.append(k)
            dists.append(_hist(v, n_bins=16))
    return labels, dists


def extract_nested_delta_scan(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_am_counter / am_extended: nested dicts of delta scans,
    each containing 'deltas' / 'amplifications' / 'physical_ctcs'.

    Compare amplifications ACROSS the sub-experiments (the headline
    metric). Comparing different quantities to each other would only
    surface the trivial "different observable" difference.
    """
    labels, dists = [], []
    for outer_key, outer_val in d.items():
        if not isinstance(outer_val, dict):
            continue
        v = outer_val.get("amplifications")
        if isinstance(v, list) and len(v) > 1:
            labels.append(outer_key)
            dists.append(_hist(v, n_bins=16))
    return labels, dists


def extract_e4_trichotomy(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """E4 trichotomy: 'results' is list[11] of dicts, one per t.
    Each dict carries arrays/scalars; build a distribution per row from
    whatever numeric arrays it has."""
    res = d.get("results")
    if not isinstance(res, list):
        return [], []
    labels, dists = [], []
    for i, row in enumerate(res):
        if not isinstance(row, dict):
            continue
        feats = []
        for k, v in row.items():
            if isinstance(v, list) and len(v) > 1:
                try:
                    feats.append(_hist(v, n_bins=16))
                except (TypeError, ValueError):
                    pass
        if feats:
            labels.append(f"row{i}")
            dists.append(np.concatenate(feats))
    return labels, dists


def extract_phase_abal(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_abal: phase_AB list[10], phase_AL list[6]; each a list of
    records. Build one distribution per group, hashed across numeric
    fields."""
    labels, dists = [], []
    for k in ("phase_AB", "phase_AL"):
        recs = d.get(k)
        if not isinstance(recs, list) or not recs or not isinstance(recs[0], dict):
            continue
        feats = []
        for field in recs[0]:
            try:
                arr = np.asarray([float(r[field]) for r in recs if field in r])
            except (TypeError, ValueError):
                continue
            if arr.size > 1 and np.all(np.isfinite(arr)):
                feats.append(_hist(arr, n_bins=8))
        if feats:
            labels.append(k)
            dists.append(np.concatenate(feats))
    return labels, dists


def extract_zy_trajectory(d: dict) -> tuple[list[str], list[np.ndarray]]:
    """phase_zy: 'trajectory' list[50] of (likely) scalar / record."""
    traj = d.get("trajectory")
    if not isinstance(traj, list) or len(traj) < 3:
        return [], []
    try:
        arr = np.asarray(traj, dtype=float).ravel()
        if arr.size > 1 and np.all(np.isfinite(arr)):
            return ["trajectory"], [_hist(arr, n_bins=16)]
    except (TypeError, ValueError):
        pass
    return [], []


# Registry: filename stem (no `_results.json` suffix) -> extractor
EXTRACTORS: dict[str, Callable] = {
    "dctc_deep_phase_a":              extract_haar_ensemble_per_pair,
    "dctc_deep_phase_b":              extract_haar_ensemble_per_pair,
    "dctc_deep_phase_c":              extract_phase_c_top_bottom,
    "dctc_deep_phase_am":             extract_delta_scan,
    "dctc_deep_phase_am_counter":     extract_nested_delta_scan,
    "dctc_deep_phase_am_extended":    extract_nested_delta_scan,
    "dctc_deep_E1_cycles":            extract_cycle_distributions,
    "dctc_deep_E4_trichotomy":        extract_e4_trichotomy,
    "dctc_deep_phase_abal":           extract_phase_abal,
    "dctc_deep_phase_zy":             extract_zy_trajectory,
    # All other dctc_deep_phase_* files are scalar-summary-only and
    # marked "unscannable" so the in-place patch step targets them.
}


SCALAR_ONLY = {
    "dctc_deep_E8_ergodicity",
    "dctc_deep_phase_ac",
    "dctc_deep_phase_adf",
    "dctc_deep_phase_aj",
    "dctc_deep_phase_d",
    "dctc_deep_phase_e",
    "dctc_deep_phase_f",
    "dctc_deep_phase_g",
    "dctc_deep_phase_knpai",
    "dctc_deep_phase_lmw",
    "dctc_deep_phase_oqr",
    "dctc_deep_phase_remaining",
}


def main() -> None:
    files = sorted(EXAMPLES_DIR.glob("dctc_*_results.json"))
    out = []
    for fp in files:
        stem = fp.stem.replace("_results", "")
        try:
            d = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError) as e:
            out.append({"file": fp.name, "stem": stem,
                         "status": "load_error", "error": repr(e)})
            continue
        extractor = EXTRACTORS.get(stem)
        if extractor is None:
            status = "scalar_summary_only" if stem in SCALAR_ONLY else "no_extractor"
            out.append({"file": fp.name, "stem": stem,
                         "status": status,
                         "needs_in_place_patch": True})
            continue
        try:
            labels, dists = extractor(d)
        except Exception as e:  # noqa: BLE001
            out.append({"file": fp.name, "stem": stem,
                         "status": "extractor_error", "error": repr(e)})
            continue
        if not dists or len(dists) < 2:
            out.append({"file": fp.name, "stem": stem,
                         "status": "insufficient_distributions",
                         "n": len(dists),
                         "needs_in_place_patch": True})
            continue
        try:
            nov = catch_novelty_in_distributions(dists, labels=labels)
        except Exception as e:  # noqa: BLE001
            out.append({"file": fp.name, "stem": stem,
                         "status": "catcher_error", "error": repr(e)})
            continue
        out.append({
            "file": fp.name,
            "stem": stem,
            "status": "scanned",
            "n_distributions": nov["n_distributions"],
            "labels": nov["labels"],
            "verdict": nov["verdict"],
            "n_sharp": len(nov["sharp_features"]),
            "sharp_features": nov["sharp_features"],
            "lambda_2_at_radius": nov["lambda_2_at_radius"],
        })

    # Summary
    by_status = {}
    for e in out:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    print(f"DCTC retrofit novelty scan: {len(out)} files")
    for k, v in sorted(by_status.items()):
        print(f"  {k:30s} {v}")
    novel = [e for e in out if e.get("verdict") == "novel_structure"]
    if novel:
        print()
        print("NOVEL STRUCTURE flagged:")
        for e in novel:
            print(f"  {e['stem']}  n_sharp={e['n_sharp']}")
            for sf in e["sharp_features"][:3]:
                print(f"    between={sf.get('between')}  step={sf.get('hamming_step')}  "
                      f"ratio={sf.get('ratio_to_median', '?')}")

    needs_patch = [e for e in out if e.get("needs_in_place_patch")]
    if needs_patch:
        print()
        print("Needs in-place patch (no distributions extractable):")
        for e in needs_patch:
            print(f"  {e['stem']}  ({e['status']})")

    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
