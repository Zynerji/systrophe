"""3D space-time MWPM decoder for the rotated surface code.

The Dennis-Kitaev-Landahl-Preskill (2002) construction treats each
violated syndrome over time as a node in a 3D graph (round, stab_idx).
Spatial edges connect adjacent stabs within a round (the dual-lattice
graph as in surface_code_dijkstra_mwpm); temporal edges connect the
same stab between consecutive rounds (cost = 1, representing a
measurement error).

The MWPM solves on the full 3D graph at once, so a syndrome that
appears in round r and disappears in round r+1 (a measurement error)
gets matched temporally with cost 1, rather than being incorrectly
paired with a spatial neighbor.

Re-analyses both:
  - d=5 round sweep (job d81kel6gbeec73akmp00, 4096 shots)
  - d=7 + d=5-high-shots (job d81kj980bvlc73d17su0, 16384 shots)
"""

from __future__ import annotations

import json
import sys
import heapq
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers, data_index
from surface_code_dijkstra_mwpm import (
    build_stab_adjacency,
    dijkstra_path,
    boundary_qubit_for,
)


def decode_3d_mwpm(
    data_bits: tuple[int, ...],
    z_syndromes_per_round: list[tuple[int, ...]],
    d: int,
) -> int:
    """3D space-time MWPM decoder.

    Builds a graph with nodes = (round, stab) for each violated syndrome
    diff position. Spatial edges weight = lattice distance, temporal
    edges weight = 1 (measurement error).
    """
    import networkx as nx
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)
    spatial_adj = build_stab_adjacency(d)

    db = np.array(data_bits, dtype=int)
    n_data = d * d
    final_z = tuple(
        int(sum(db[q] for q in Z_stabs[i]) % 2)
        for i in range(n_z)
    )

    # Build syndrome diff sequence
    if not z_syndromes_per_round:
        all_syndromes = [final_z]
    else:
        all_syndromes = list(z_syndromes_per_round) + [final_z]
    diff_history = []  # list of tuples per round
    prev = tuple([0] * n_z)
    for syn in all_syndromes:
        diff = tuple((syn[i] ^ prev[i]) for i in range(n_z))
        diff_history.append(diff)
        prev = syn

    # Collect (round, stab) violations
    violations = []
    for r, diff in enumerate(diff_history):
        for s, v in enumerate(diff):
            if v == 1:
                violations.append((r, s))

    if not violations:
        return int(sum(db[data_index(rr, 0, d)] for rr in range(d)) % 2)

    # Build the 3D graph
    g = nx.Graph()
    n_v = len(violations)
    # Add virtual boundary node if odd
    if n_v % 2 == 1:
        violations_padded = violations + [("BOUNDARY", None)]
    else:
        violations_padded = list(violations)
    n_padded = len(violations_padded)

    for i, j in combinations(range(n_padded), 2):
        vi = violations_padded[i]
        vj = violations_padded[j]
        if vi[0] == "BOUNDARY" and vj[0] == "BOUNDARY":
            w = 0
        elif vi[0] == "BOUNDARY":
            r, s = vj
            w = 1
        elif vj[0] == "BOUNDARY":
            r, s = vi
            w = 1
        else:
            r1, s1 = vi
            r2, s2 = vj
            if s1 == s2:
                # Same stab: temporal cost = |dr|
                w = abs(r1 - r2)
            else:
                # Different stabs: spatial cost via Dijkstra + temporal
                spatial_q = dijkstra_path(s1, s2, spatial_adj)
                spatial_d = len(spatial_q) if spatial_q else 1000
                temporal_d = abs(r1 - r2)
                w = spatial_d + temporal_d
        g.add_edge(i, j, weight=-w)

    matching = nx.min_weight_matching(g)

    flips = set()
    for ii, jj in matching:
        vi = violations_padded[ii]
        vj = violations_padded[jj]
        if vi[0] == "BOUNDARY" or vj[0] == "BOUNDARY":
            real_v = vj if vi[0] == "BOUNDARY" else vi
            r, s = real_v
            q = boundary_qubit_for(s, d)
            if q in flips:
                flips.remove(q)
            else:
                flips.add(q)
        else:
            r1, s1 = vi
            r2, s2 = vj
            if s1 == s2:
                # Pure temporal: this is a measurement error, no data
                # qubit flip needed
                continue
            spatial_q = dijkstra_path(s1, s2, spatial_adj)
            for q in spatial_q:
                if q in flips:
                    flips.remove(q)
                else:
                    flips.add(q)

    for q in flips:
        db[q] = 1 - db[q]

    return int(sum(db[data_index(rr, 0, d)] for rr in range(d)) % 2)


def reanalyze_jobs_with_3d_mwpm(
    instance: str = "Zynerji",
) -> dict:
    """Re-analyse both Heron-r2 jobs with 3D MWPM."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    out_dir = Path(__file__).parent / "results"

    jobs_to_analyze = [
        {
            "submitted_file": "surface_code_d5_round_sweep_submitted.json",
            "label": "d5_round_sweep_4096_shots",
        },
        {
            "submitted_file": "surface_code_d7_and_d5_high_shots_submitted.json",
            "label": "d7_and_d5_high_shots_16384",
        },
    ]

    all_results = {}
    for jdesc in jobs_to_analyze:
        submitted = json.loads((out_dir / jdesc["submitted_file"]).read_text())
        job_id = submitted["job_id"]
        job = service.job(job_id)
        if str(job.status()) not in ("DONE", "COMPLETED"):
            print(f"Skipping {job_id}: status={job.status()}")
            continue
        metadata = submitted["metadata_per_circuit"]
        result = job.result()

        paired = {}
        for i, meta in enumerate(metadata):
            d = meta.get("d", 5)  # d5 file doesn't have explicit "d"
            nr = meta["n_rounds"]
            key = f"d{d}_n{nr}"
            data = result[i].data
            creg = next(iter(data))
            counts = getattr(data, creg).get_counts()
            if meta["kind"] == "bare":
                total = sum(counts.values())
                zero_count = sum(v for k, v in counts.items() if k == "0")
                paired.setdefault(key, {})["bare"] = {
                    "physical_zero_rate": zero_count / total,
                    "n_shots_total": total,
                }
                continue
            n_data = d * d
            _, Z_stabs_d = build_stabilizers(d)
            n_z = len(Z_stabs_d)
            n_log_zero = 0
            n_total = 0
            for bitstring, count in counts.items():
                parts = bitstring.split(" ")
                data_part = None
                sx_parts = []
                for part in parts:
                    if len(part) == n_data:
                        data_part = part
                    elif len(part) == n_z:
                        sx_parts.append(part)
                if data_part is None:
                    continue
                data_bits = tuple(int(b) for b in reversed(data_part))
                sx_parts.reverse()
                z_syndromes = [
                    tuple(int(b) for b in reversed(s))
                    for s in sx_parts[-nr:]
                ]
                log = decode_3d_mwpm(data_bits, z_syndromes, d)
                if log == 0:
                    n_log_zero += count
                n_total += count
            paired.setdefault(key, {})["surface"] = {
                "d": d,
                "n_rounds": nr,
                "logical_zero_rate_3d_mwpm": n_log_zero / max(n_total, 1),
                "n_shots_total": n_total,
            }

        all_results[jdesc["label"]] = paired

    # Print unified table
    print()
    print("3D space-time MWPM decoder (Dennis-Kitaev-Landahl-Preskill style)")
    print("=" * 90)
    print(f"{'job_label':>32}  {'key':>10}  {'logical':>10}  {'bare':>10}  "
          f"{'diff':>10}  {'sigma':>8}")
    for label, paired in all_results.items():
        for key in sorted(paired.keys()):
            s = paired[key].get("surface", {})
            b = paired[key].get("bare", {})
            lz = s.get("logical_zero_rate_3d_mwpm", float("nan"))
            bz = b.get("physical_zero_rate", float("nan"))
            n_shots = s.get("n_shots_total", 0)
            if n_shots > 0 and not np.isnan(lz) and not np.isnan(bz):
                sigma_l = (lz * (1 - lz) / n_shots) ** 0.5
                sigma_b = (bz * (1 - bz) / n_shots) ** 0.5
                sigma_diff = (sigma_l ** 2 + sigma_b ** 2) ** 0.5
                n_sig = (lz - bz) / sigma_diff if sigma_diff > 0 else 0.0
            else:
                n_sig = 0.0
            print(f"{label:>32}  {key:>10}  {lz:>10.4f}  {bz:>10.4f}  "
                  f"{lz - bz:>+10.4f}  {n_sig:>+8.2f}")

    out_path = out_dir / "surface_code_3d_mwpm_analysis.json"
    out_path.write_text(json.dumps({
        "decoder": "3d_mwpm_spacetime",
        "results_per_job": all_results,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return all_results


if __name__ == "__main__":
    reanalyze_jobs_with_3d_mwpm()
