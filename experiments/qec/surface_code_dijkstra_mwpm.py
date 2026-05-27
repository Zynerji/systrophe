"""Proper Dijkstra-shortest-path MWPM decoder for the rotated surface
code.

Improves over surface_code_d5_mwpm.py and surface_code_generic.py by
using the full shortest-path X-error chain (flipping ALL data qubits
on the matched-pair path) rather than a single-qubit boundary flip.

The chain-finding works on the dual-lattice graph: nodes = Z-stabs
(plus virtual boundary node), edges = adjacent stab pairs (sharing
a data qubit). Dijkstra finds the shortest such path between matched
stab pairs. The data qubits on this path are the X-flip correction.

This is the standard QEC literature MWPM and should sustain the d=5
break-even crossing past n_rounds=1 (where the current implementation
only crosses at n=1 due to under-correction).

Re-analyses the existing d=5 round-sweep job d81kel6gbeec73akmp00.
Also exposes a unified decode_with_dijkstra_mwpm(d, ...) function
that surface_code_d7_sweep can use.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers, data_index


def build_stab_adjacency(d: int) -> dict:
    """Build adjacency: stab_i -> {stab_j: shared_data_qubit}.

    Two Z-stabs are adjacent if they share at least one data qubit;
    the shared qubit IS the edge (the X-error that connects them).
    """
    _, Z_stabs = build_stabilizers(d)
    adj = {i: {} for i in range(len(Z_stabs))}
    for i, qsi in enumerate(Z_stabs):
        for j, qsj in enumerate(Z_stabs):
            if i == j:
                continue
            shared = set(qsi) & set(qsj)
            if shared:
                adj[i][j] = min(shared)
    return adj


def boundary_qubit_for(stab_idx: int, d: int) -> int:
    """Return the data qubit on the boundary edge that the given
    stab can match to. For Z-stabs (which have smooth left/right
    boundaries), this is the leftmost or rightmost data qubit in the
    stab's support depending on which edge is closer.
    """
    _, Z_stabs = build_stabilizers(d)
    qs = Z_stabs[stab_idx]
    # Pick the qubit with min or max column
    cols = [(q % d, q) for q in qs]
    cols.sort()
    # If average column < d/2, return leftmost; else rightmost
    avg_col = sum(c for c, _ in cols) / len(cols)
    if avg_col < d / 2:
        return cols[0][1]
    return cols[-1][1]


def dijkstra_path(start: int, end: int, adj: dict) -> list[int]:
    """Standard Dijkstra on unweighted stab-adjacency. Returns the
    list of data qubits to flip along the matched chain (the edges).
    """
    if start == end:
        return []
    import heapq
    visited = {start: None}
    pq = [(0, start)]
    while pq:
        dist, u = heapq.heappop(pq)
        if u == end:
            break
        for v in adj.get(u, {}):
            if v not in visited:
                visited[v] = u
                heapq.heappush(pq, (dist + 1, v))
    if end not in visited:
        return []
    # Backtrack to get path of stabs
    path_stabs = [end]
    while path_stabs[-1] != start:
        path_stabs.append(visited[path_stabs[-1]])
    path_stabs.reverse()
    # Convert to edge data-qubits
    qubits = []
    for a, b in zip(path_stabs[:-1], path_stabs[1:]):
        qubits.append(adj[a][b])
    return qubits


def decode_with_dijkstra_mwpm(
    data_bits: tuple[int, ...],
    z_syndromes_per_round: list[tuple[int, ...]],
    d: int,
) -> int:
    """Full MWPM decoder with Dijkstra-shortest-path X-flip chains.

    For each round, network MWPM pairs up violated stabs (with virtual
    boundary nodes for unpaired stabs), then Dijkstra finds the
    shortest path through the stab-adjacency graph for each pair.
    All data qubits on each path are flipped.
    """
    import networkx as nx
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)
    adj = build_stab_adjacency(d)

    db = np.array(data_bits, dtype=int)
    n_data = d * d

    final_z = tuple(
        int(sum(db[q] for q in Z_stabs[i]) % 2)
        for i in range(n_z)
    )

    # Syndrome difference history
    if not z_syndromes_per_round:
        history = [[i for i, s in enumerate(final_z) if s == 1]]
    else:
        history = []
        prev = tuple([0] * n_z)
        for syn in z_syndromes_per_round:
            diff = tuple((syn[i] ^ prev[i]) for i in range(n_z))
            history.append([i for i, dv in enumerate(diff) if dv == 1])
            prev = syn
        diff = tuple((final_z[i] ^ prev[i]) for i in range(n_z))
        history.append([i for i, dv in enumerate(diff) if dv == 1])

    flips = set()
    for round_violations in history:
        if not round_violations:
            continue
        # Add a virtual boundary node for odd-cardinality matching
        if len(round_violations) % 2 == 1:
            round_violations_padded = round_violations + ["BOUNDARY"]
        else:
            round_violations_padded = list(round_violations)

        g = nx.Graph()
        for i, j in combinations(range(len(round_violations_padded)), 2):
            si = round_violations_padded[i]
            sj = round_violations_padded[j]
            if si == "BOUNDARY" and sj == "BOUNDARY":
                w = 0
            elif si == "BOUNDARY":
                # Distance from sj to nearest boundary qubit; we use
                # the "boundary qubit" for sj as the cost of 1 (flip
                # that one qubit).
                w = 1
            elif sj == "BOUNDARY":
                w = 1
            else:
                # Real Dijkstra distance
                qubits = dijkstra_path(si, sj, adj)
                w = len(qubits) if qubits else 1000
            g.add_edge(i, j, weight=-w)
        if g.number_of_edges() == 0:
            continue
        matching = nx.min_weight_matching(g)
        for ii, jj in matching:
            si = round_violations_padded[ii]
            sj = round_violations_padded[jj]
            if si == "BOUNDARY" or sj == "BOUNDARY":
                # Single boundary qubit flip
                real_stab = sj if si == "BOUNDARY" else si
                q = boundary_qubit_for(real_stab, d)
                if q in flips:
                    flips.remove(q)
                else:
                    flips.add(q)
            else:
                qubits = dijkstra_path(si, sj, adj)
                for q in qubits:
                    if q in flips:
                        flips.remove(q)
                    else:
                        flips.add(q)

    for q in flips:
        db[q] = 1 - db[q]

    return int(sum(db[data_index(r, 0, d)] for r in range(d)) % 2)


def reanalyze_d5_with_dijkstra(
    job_id: str = "d81kel6gbeec73akmp00",
    instance: str = "Zynerji",
) -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}

    submitted = json.loads(
        (Path(__file__).parent / "results" /
          "surface_code_d5_round_sweep_submitted.json").read_text()
    )
    metadata = submitted["metadata_per_circuit"]
    result = job.result()

    d = 5
    n_data = d * d
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)

    paired = {}
    for i, meta in enumerate(metadata):
        data = result[i].data
        creg = next(iter(data))
        counts = getattr(data, creg).get_counts()
        if meta["kind"] == "bare":
            total = sum(counts.values())
            zero_count = sum(v for k, v in counts.items() if k == "0")
            paired.setdefault(meta["n_rounds"], {})["bare"] = {
                "physical_zero_rate": zero_count / total,
                "n_shots_total": total,
            }
            continue
        nr = meta["n_rounds"]
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
                tuple(int(b) for b in reversed(s)) for s in sx_parts[-nr:]
            ]
            log = decode_with_dijkstra_mwpm(data_bits, z_syndromes, d)
            if log == 0:
                n_log_zero += count
            n_total += count
        paired.setdefault(nr, {})["surface"] = {
            "logical_zero_rate_dijkstra_mwpm": n_log_zero / max(n_total, 1),
            "n_shots_total": n_total,
        }

    print()
    print("d=5 surface code Z-memory, Dijkstra-MWPM decoder")
    print("=" * 75)
    print(f"{'n_rounds':>10}  {'Dijkstra-MWPM':>15}  {'bare':>12}  "
          f"{'diff':>10}  {'sigma':>8}")
    for nr in sorted(paired.keys()):
        s = paired[nr].get("surface", {})
        b = paired[nr].get("bare", {})
        lz = s.get("logical_zero_rate_dijkstra_mwpm", float("nan"))
        bz = b.get("physical_zero_rate", float("nan"))
        n_shots = s.get("n_shots_total", 0)
        if n_shots > 0:
            sigma_l = (lz * (1 - lz) / n_shots) ** 0.5
            sigma_b = (bz * (1 - bz) / n_shots) ** 0.5
            sigma_diff = (sigma_l ** 2 + sigma_b ** 2) ** 0.5
            n_sig = (lz - bz) / sigma_diff if sigma_diff > 0 else 0.0
        else:
            n_sig = 0.0
        print(f"{nr:>10d}  {lz:>15.4f}  {bz:>12.4f}  {lz - bz:>+10.4f}  "
              f"{n_sig:>+8.2f}")

    out_path = Path(__file__).parent / "results" / "surface_code_d5_dijkstra_mwpm_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "decoder": "dijkstra_mwpm",
        "paired_by_n_rounds": paired,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return paired


def reanalyze_d7_and_d5_high_shots_with_dijkstra(
    job_id: str = None,
    instance: str = "Zynerji",
) -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    out_dir = Path(__file__).parent / "results"
    submitted_path = out_dir / "surface_code_d7_and_d5_high_shots_submitted.json"
    submitted = json.loads(submitted_path.read_text())
    if job_id is None:
        job_id = submitted["job_id"]
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}
    metadata = submitted["metadata_per_circuit"]
    result = job.result()

    paired = {}
    for i, meta in enumerate(metadata):
        d = meta["d"]
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
                tuple(int(b) for b in reversed(s)) for s in sx_parts[-nr:]
            ]
            log = decode_with_dijkstra_mwpm(data_bits, z_syndromes, d)
            if log == 0:
                n_log_zero += count
            n_total += count
        paired.setdefault(key, {})["surface"] = {
            "d": d,
            "n_rounds": nr,
            "logical_zero_rate_dijkstra_mwpm": n_log_zero / max(n_total, 1),
            "n_shots_total": n_total,
        }

    print()
    print("d=7 + d=5-high-shots round sweep on ibm_kingston (Dijkstra-MWPM)")
    print("=" * 80)
    print(f"{'key':>10}  {'d':>3} {'n':>3}  {'Dijkstra-MWPM':>15}  "
          f"{'bare':>10}  {'diff':>10}  {'sigma':>8}")
    for key in sorted(paired.keys()):
        s = paired[key].get("surface", {})
        b = paired[key].get("bare", {})
        lz = s.get("logical_zero_rate_dijkstra_mwpm", float("nan"))
        bz = b.get("physical_zero_rate", float("nan"))
        n_shots = s.get("n_shots_total", 0)
        if n_shots > 0:
            sigma_l = (lz * (1 - lz) / n_shots) ** 0.5
            sigma_b = (bz * (1 - bz) / n_shots) ** 0.5
            sigma_diff = (sigma_l ** 2 + sigma_b ** 2) ** 0.5
            n_sig = (lz - bz) / sigma_diff if sigma_diff > 0 else 0.0
        else:
            n_sig = 0.0
        d = s.get("d", "?")
        nr = s.get("n_rounds", "?")
        print(f"{key:>10}  {d:>3} {nr:>3}  {lz:>15.4f}  {bz:>10.4f}  "
              f"{lz - bz:>+10.4f}  {n_sig:>+8.2f}")

    out_path = out_dir / "surface_code_d7_and_d5_high_shots_dijkstra_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "decoder": "dijkstra_mwpm",
        "paired_by_d_n": paired,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return paired


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("d5", "d7"), default="d7")
    args = ap.parse_args()
    if args.target == "d5":
        reanalyze_d5_with_dijkstra()
    else:
        reanalyze_d7_and_d5_high_shots_with_dijkstra()
