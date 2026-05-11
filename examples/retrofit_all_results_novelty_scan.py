"""Generic retrofit catcher pass over EVERY examples/*_results.json.

Walks each result JSON, harvests all numeric list values (recursively),
groups them by detected quantity prefix where possible, and runs the
per-quantity catcher to surface emergent structure WITHOUT touching
the source scripts.

Outputs `examples/retrofit_all_results_novelty.json` with one entry
per scanned file.

Skips files that already carry a `novelty_catcher` key or that have
been scanned by the targeted retrofit scripts (so we don't double-
report previously-known novelty).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)

EXAMPLES_DIR = Path(__file__).parent
OUT_PATH = EXAMPLES_DIR / "retrofit_all_results_novelty.json"

# Files already covered by targeted retrofits:
SKIP_FILES = {
    "retrofit_novelty_results.json",
    "retrofit_dctc_novelty_results.json",
    "retrofit_all_results_novelty.json",
    "first_ch_cluster_investigation.json",
}


def harvest_arrays(node, prefix: str = "", out: list | None = None) -> list:
    """Walk a JSON tree and return [(key_path, np.ndarray)] for every
    1-D numeric list of length >= 5 encountered."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for k, v in node.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            harvest_arrays(v, sub, out)
    elif isinstance(node, list):
        if len(node) >= 5:
            try:
                arr = np.asarray(node, dtype=float)
                if arr.ndim == 1 and np.all(np.isfinite(arr)):
                    out.append((prefix, arr))
                    return out
            except (TypeError, ValueError):
                pass
        for i, v in enumerate(node):
            harvest_arrays(v, f"{prefix}[{i}]", out)
    return out


def group_by_quantity(arrays: list) -> dict:
    """Heuristic grouping: arrays sharing a trailing token (e.g. all
    `*_purity` arrays) become one quantity group; the qualifier
    (prefix) becomes the condition label."""
    groups: dict[str, dict] = {}
    for key, arr in arrays:
        # Trailing identifier token (split on . and [...])
        token = re.split(r'[\.\[\]]', key)[-1] or "value"
        condition = key.replace(token, "").rstrip(".[ ]") or "all"
        if not condition:
            condition = "all"
        # Avoid empty / weird groups
        if token == "":
            token = "value"
        groups.setdefault(token, {})[condition] = arr
    return groups


def scan_one_file(fp: Path) -> dict:
    try:
        d = json.loads(fp.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"file": fp.name, "status": "load_error", "error": repr(e)}
    # Skip if already has catcher
    if isinstance(d, dict) and "novelty_catcher" in d:
        return {"file": fp.name, "status": "already_catchered"}
    arrays = harvest_arrays(d)
    if not arrays:
        return {"file": fp.name, "status": "no_numeric_arrays"}
    groups = group_by_quantity(arrays)
    # Drop groups with <2 conditions (catcher needs at least 2 to compare)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return {"file": fp.name, "status": "single_condition_only",
                 "n_arrays": len(arrays)}
    try:
        nov = catch_novelty_per_quantity(groups)
    except Exception as e:  # noqa: BLE001
        return {"file": fp.name, "status": "catcher_error", "error": repr(e)}
    return {
        "file": fp.name,
        "status": "scanned",
        "n_quantities": len(groups),
        "aggregate_verdict": nov["aggregate_verdict"],
        "novel_quantities": nov["novel_quantities"],
        "per_quantity": {
            q: {"verdict": r.get("verdict"),
                "n_distributions": r.get("n_distributions", 0),
                "labels": r.get("labels", []),
                "n_sharp": len(r.get("sharp_features", [])),
                "sharp_features": r.get("sharp_features", []),
                }
            for q, r in nov["per_quantity"].items()
        },
    }


def main() -> None:
    files = sorted([
        f for f in EXAMPLES_DIR.glob("*_results.json")
        if f.name not in SKIP_FILES
    ])
    out = []
    for fp in files:
        out.append(scan_one_file(fp))

    # Summary
    by_status: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    novel_files: list[tuple[str, list[str]]] = []
    for e in out:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        if e["status"] == "scanned":
            v = e["aggregate_verdict"]
            by_verdict[v] = by_verdict.get(v, 0) + 1
            if v == "novel_structure":
                novel_files.append((e["file"], e["novel_quantities"]))

    print(f"=== retrofit_all_results_novelty: {len(out)} files ===")
    for k, v in sorted(by_status.items()):
        print(f"  status: {k:30s} {v}")
    print()
    for k, v in sorted(by_verdict.items()):
        print(f"  verdict: {k:30s} {v}")
    print()
    if novel_files:
        print("NOVEL emergents caught:")
        for fname, qs in novel_files:
            print(f"  {fname}  novel quantities: {qs}")
            entry = next(e for e in out if e["file"] == fname)
            for q, r in entry["per_quantity"].items():
                if r["verdict"] == "novel_structure":
                    for sf in r["sharp_features"][:3]:
                        b = sf.get("between")
                        s = sf.get("hamming_step")
                        ratio = round(sf.get("ratio_to_median", 0), 2)
                        print(f"      {q}: {b}  step={s}  ratio={ratio}")

    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
