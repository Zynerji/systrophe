"""Naive single-boundary-flip MWPM decoder, kept as a control.

This is the algorithm the Systrophe Heron-r2 pipeline used before
v0.19.2's switch to Dijkstra-shortest-path matching. Every matched
pair of violated stabs results in a single boundary-qubit flip per
stab (rather than flipping the full chain). On Heron-r2 d=5 round-
sweep data, replacing this naive variant with `decode_with_dijkstra_mwpm`
gave +5 to +25 percentage points in logical-zero rate.

We ship this as a control so anyone reproducing the
`compare_naive_vs_dijkstra.py` benchmark can verify the algorithmic
delta on their own synthetic noise model.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .decoder import _boundary_qubit_for, build_stab_adjacency
from .surface_code import build_stabilizers, data_index


def decode_with_naive_mwpm(
    data_bits: tuple[int, ...] | list[int],
    z_syndromes_per_round: list[tuple[int, ...]] | list[list[int]],
    d: int,
) -> int:
    """Naive single-boundary-flip MWPM.

    Same input contract as `decode_with_dijkstra_mwpm`. Per matched
    pair, ALL pairs (real-real or real-boundary) are treated as a
    single-qubit boundary flip on each stab's nearest boundary qubit.
    No chain-finding.
    """
    import networkx as nx
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)
    adj = build_stab_adjacency(d)

    db = np.array(data_bits, dtype=int)
    if db.shape != (d * d,):
        raise ValueError(f"data_bits length {db.shape[0]} != d*d = {d*d}")

    final_z = tuple(
        int(sum(db[q] for q in Z_stabs[i]) % 2)
        for i in range(n_z)
    )

    if not z_syndromes_per_round:
        history = [[i for i, s in enumerate(final_z) if s == 1]]
    else:
        history = []
        prev = tuple([0] * n_z)
        for syn in z_syndromes_per_round:
            if len(syn) != n_z:
                raise ValueError(
                    f"syndrome round has length {len(syn)} != n_z = {n_z}"
                )
            diff = tuple((syn[i] ^ prev[i]) for i in range(n_z))
            history.append([i for i, dv in enumerate(diff) if dv == 1])
            prev = syn
        diff = tuple((final_z[i] ^ prev[i]) for i in range(n_z))
        history.append([i for i, dv in enumerate(diff) if dv == 1])

    flips: set[int] = set()
    for round_violations in history:
        if not round_violations:
            continue
        if len(round_violations) % 2 == 1:
            padded = round_violations + ["BOUNDARY"]
        else:
            padded = list(round_violations)

        g = nx.Graph()
        for i, j in combinations(range(len(padded)), 2):
            # Naive: every edge weight is 1
            g.add_edge(i, j, weight=-1)
        if g.number_of_edges() == 0:
            continue
        matching = nx.min_weight_matching(g)
        for ii, jj in matching:
            si = padded[ii]
            sj = padded[jj]
            for s in (si, sj):
                if s == "BOUNDARY":
                    continue
                q = _boundary_qubit_for(int(s), d)
                if q in flips:
                    flips.remove(q)
                else:
                    flips.add(q)

    for q in flips:
        db[q] = 1 - db[q]

    return int(sum(db[data_index(r, 0, d)] for r in range(d)) % 2)
