"""Generic rotated surface code for any odd distance d on heavy-hex.

Generalizes surface_code_d5.py to support d in {3, 5, 7, 9, ...}.
The code has:
  n_data = d * d
  n_X_stab = (d*d - 1) / 2
  n_Z_stab = (d*d - 1) / 2
  total qubits = d*d + (d*d - 1) = 2 d*d - 1

For d=7: 49 data + 48 ancilla = 97 qubits (fits Heron-r2 156).
For d=9: 81 data + 80 ancilla = 161 qubits (DOES NOT fit Heron-r2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def data_index(row: int, col: int, d: int) -> int:
    return row * d + col


def build_stabilizers(d: int) -> tuple[list[list[int]], list[list[int]]]:
    """X- and Z-stabilizer supports for the rotated d x d surface code.

    Standard convention: rough top/bottom (X-boundaries), smooth
    left/right (Z-boundaries). For d > 5 the boundary half-plaquettes
    have an alternating-by-parity pattern.
    """
    X_stabs, Z_stabs = [], []
    # Internal plaquettes
    for r in range(d - 1):
        for c in range(d - 1):
            qs = [
                data_index(r,     c,     d),
                data_index(r,     c + 1, d),
                data_index(r + 1, c,     d),
                data_index(r + 1, c + 1, d),
            ]
            if (r + c) % 2 == 0:
                X_stabs.append(qs)
            else:
                Z_stabs.append(qs)
    # Top edge half-plaquettes (X)
    for c in range(d - 1):
        if c % 2 == 1:  # alternate with internal pattern
            X_stabs.append([data_index(0, c, d), data_index(0, c + 1, d)])
    # Bottom edge half-plaquettes (X)
    for c in range(d - 1):
        if (d - 1 + c) % 2 == 1:
            X_stabs.append([data_index(d - 1, c, d), data_index(d - 1, c + 1, d)])
    # Left edge half-plaquettes (Z)
    for r in range(d - 1):
        if r % 2 == 0:
            Z_stabs.append([data_index(r, 0, d), data_index(r + 1, 0, d)])
    # Right edge half-plaquettes (Z)
    for r in range(d - 1):
        if (r + d - 1) % 2 == 0:
            Z_stabs.append([data_index(r, d - 1, d), data_index(r + 1, d - 1, d)])

    return X_stabs, Z_stabs


def build_full_z_memory_experiment(d: int, n_rounds: int):
    """Z-memory experiment: |0>^(d*d) (already Z-stab +1 eigenstate),
    n_rounds of Z-syndrome extraction, final data measurement."""
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    _, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_z = len(Z_stabs)

    data = QuantumRegister(n_data, "data")
    anc_z = QuantumRegister(n_z, "anc_z")
    cr_data = ClassicalRegister(n_data, "data_meas")
    syndrome_regs = [ClassicalRegister(n_z, f"sz_{r}") for r in range(n_rounds)]

    qc = QuantumCircuit(data, anc_z, cr_data, *syndrome_regs)
    qc.barrier()

    for r in range(n_rounds):
        cr_z = syndrome_regs[r]
        for stab_idx, qs in enumerate(Z_stabs):
            for q in qs:
                qc.cx(data[q], anc_z[stab_idx])
            qc.measure(anc_z[stab_idx], cr_z[stab_idx])
            qc.reset(anc_z[stab_idx])
        qc.barrier()

    qc.measure(data, cr_data)
    return qc


def decode_with_networkx_mwpm(
    data_bits: tuple[int, ...],
    z_syndromes_per_round: list[tuple[int, ...]],
    d: int,
) -> int:
    """MWPM-style decoder using networkx.min_weight_matching on syndrome
    difference history. Returns predicted logical Z."""
    import networkx as nx
    from itertools import combinations

    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)

    # Stab positions (centres of mass of data-qubit support)
    stab_pos = [
        (
            float(np.mean([q // d for q in qs])),
            float(np.mean([q % d for q in qs])),
        )
        for qs in Z_stabs
    ]

    def lattice_dist(s1: int, s2: int) -> float:
        p1, p2 = stab_pos[s1], stab_pos[s2]
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    db = np.array(data_bits, dtype=int)
    n_data = d * d

    # Final data-derived Z-syndrome
    final_z = tuple(
        int(sum(db[q] for q in Z_stabs[i]) % 2)
        for i in range(n_z)
    )

    # Build syndrome difference history
    if not z_syndromes_per_round:
        history = [[i for i, s in enumerate(final_z) if s == 1]]
    else:
        history = []
        prev = tuple([0] * n_z)
        for syn in z_syndromes_per_round:
            diff = tuple((syn[i] ^ prev[i]) for i in range(n_z))
            history.append([i for i, dv in enumerate(diff) if dv == 1])
            prev = syn
        # Last difference = (final data syndrome) XOR (last round syndrome)
        diff = tuple((final_z[i] ^ prev[i]) for i in range(n_z))
        history.append([i for i, dv in enumerate(diff) if dv == 1])

    flips = set()
    for round_violations in history:
        if len(round_violations) < 2:
            for s in round_violations:
                # Flip first qubit in stab's support (boundary handling)
                q = Z_stabs[s][0]
                if q in flips:
                    flips.remove(q)
                else:
                    flips.add(q)
            continue
        g = nx.Graph()
        for i, j in combinations(range(len(round_violations)), 2):
            d_ij = lattice_dist(round_violations[i], round_violations[j])
            g.add_edge(
                round_violations[i], round_violations[j],
                weight=-d_ij,
            )
        matching = nx.min_weight_matching(g)
        matched = set()
        for s1, s2 in matching:
            matched.add(s1)
            matched.add(s2)
            # Shortest-path single-qubit flip: shared-support qubit if any
            shared = set(Z_stabs[s1]) & set(Z_stabs[s2])
            if shared:
                q = min(shared)
            else:
                # Pick a qubit in s1 closest to s2 (Manhattan)
                p2 = stab_pos[s2]
                q = min(
                    Z_stabs[s1],
                    key=lambda qq: (abs(qq // d - p2[0]) +
                                     abs(qq % d - p2[1])),
                )
            if q in flips:
                flips.remove(q)
            else:
                flips.add(q)
        # Unmatched (boundary chain)
        unmatched = [s for s in round_violations if s not in matched]
        for s in unmatched:
            q = Z_stabs[s][0]
            if q in flips:
                flips.remove(q)
            else:
                flips.add(q)

    for q in flips:
        db[q] = 1 - db[q]

    # Logical Z = parity of column 0
    return int(sum(db[data_index(r, 0, d)] for r in range(d)) % 2)


def report_structure(d: int) -> None:
    X_stabs, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_x = len(X_stabs)
    n_z = len(Z_stabs)
    print(f"Rotated surface code, d = {d}")
    print(f"  n_data = {n_data}")
    print(f"  n_X_stab = {n_x}")
    print(f"  n_Z_stab = {n_z}")
    print(f"  total qubits = {n_data + n_x + n_z}")
    print(f"  CNOTs per Z-round (untranspiled) = {sum(len(qs) for qs in Z_stabs)}")


def simulate_z_memory(d: int, n_rounds: int, shots: int = 1024) -> dict:
    from qiskit_aer import AerSimulator
    qc = build_full_z_memory_experiment(d=d, n_rounds=n_rounds)
    sim = AerSimulator(method="stabilizer")
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()
    n_data = d * d
    n_z = len(build_stabilizers(d)[1])
    n_total = 0
    n_logical_zero = 0
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
            tuple(int(b) for b in reversed(s)) for s in sx_parts[-n_rounds:]
        ]
        logical = decode_with_networkx_mwpm(data_bits, z_syndromes, d)
        if logical == 0:
            n_logical_zero += count
        n_total += count
    return {
        "logical_zero_rate": n_logical_zero / max(n_total, 1),
        "n_shots_total": n_total,
        "n_rounds": n_rounds,
        "d": d,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=0)
    ap.add_argument("--report-structure", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    if args.report_structure:
        report_structure(args.d)
    if args.simulate:
        result = simulate_z_memory(args.d, args.n_rounds)
        print(f"Noiseless simulation d={args.d}, n_rounds={args.n_rounds}:")
        print(f"  logical Z=0 rate = {result['logical_zero_rate']:.4f}")
        print(f"  (Expected 1.0 for correct encoder + decoder)")


if __name__ == "__main__":
    main()
