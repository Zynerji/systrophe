"""Dijkstra-shortest-path MWPM decoder.

Algorithm
---------
1. **Stab adjacency**: build the dual-lattice graph in which each
   Z-stab is a node, and two Z-stabs are connected iff they share at
   least one data qubit. The data qubit connecting them IS the edge
   (an X-error on that qubit toggles both stabs).
2. **Syndrome-difference history**: for n syndrome-extraction rounds
   plus a final data measurement, compute the per-round XOR with the
   previous round's syndrome. Stabs that flip between rounds are
   "violations".
3. **Per-round MWPM**: for each round's violations, pair them up via
   `networkx.min_weight_matching` on a complete graph where the edge
   weight between two violated stabs is the Dijkstra-shortest-path
   length in the stab-adjacency graph (i.e. the minimum number of
   data-qubit flips needed to undo that pair). A virtual `BOUNDARY`
   node absorbs unpaired violations at cost 1 (single boundary flip).
4. **Apply corrections**: every matched pair's shortest-path edges
   identify the data qubits to flip (XOR into the correction set).
   Boundary matches flip the stab's nearest boundary data qubit.
5. **Logical readout**: after applying all flips to the measured data
   bits, the logical Z is the parity of the leftmost column of the
   d x d lattice (canonical Z-string for the rotated surface code with
   smooth left/right boundaries).

The key win over naive single-qubit-boundary-flip MWPM is step 1+3:
the full Dijkstra path flips ALL data qubits on the connecting chain
between paired stabs. On the Heron-r2 d=5 round-sweep this added
+5-25 percentage points in logical-zero rate over the naive variant,
and was the change that sustained break-even past n_rounds=1
(`paper/surface_code_multidistance_break_even.pdf`).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .surface_code import build_stabilizers, data_index


def build_stab_adjacency(d: int) -> dict[int, dict[int, int]]:
    """Return adjacency dict: `adj[i][j] = shared_data_qubit_index` iff
    Z-stabs i and j share at least one data qubit.

    If two stabs share multiple data qubits, the lowest-index one is
    used (deterministic). This matches the algorithm in
    `Systrophe/experiments/surface_code_dijkstra_mwpm.py`.
    """
    _, Z_stabs = build_stabilizers(d)
    adj: dict[int, dict[int, int]] = {i: {} for i in range(len(Z_stabs))}
    for i, qsi in enumerate(Z_stabs):
        for j, qsj in enumerate(Z_stabs):
            if i == j:
                continue
            shared = set(qsi) & set(qsj)
            if shared:
                adj[i][j] = min(shared)
    return adj


def _boundary_qubit_for(stab_idx: int, d: int) -> int:
    """For an unpaired stab, return the boundary data qubit to flip.

    Heuristic: pick the leftmost or rightmost data qubit in the stab's
    support, depending on which boundary the stab is closer to.
    """
    _, Z_stabs = build_stabilizers(d)
    qs = Z_stabs[stab_idx]
    cols = [(q % d, q) for q in qs]
    cols.sort()
    avg_col = sum(c for c, _ in cols) / len(cols)
    if avg_col < d / 2:
        return cols[0][1]
    return cols[-1][1]


def dijkstra_path(start: int, end: int, adj: dict[int, dict[int, int]]) -> list[int]:
    """Unweighted Dijkstra from `start` stab to `end` stab on `adj`.

    Returns the list of data qubits on the matched chain (= the edges
    of the path). Empty list if the path doesn't exist or start == end.
    """
    if start == end:
        return []
    visited: dict[int, int | None] = {start: None}
    pq: list[tuple[int, int]] = [(0, start)]
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
    # Backtrack
    path = [end]
    while path[-1] != start:
        prev = visited[path[-1]]
        if prev is None:
            return []
        path.append(prev)
    path.reverse()
    # Convert to edge data-qubits
    qubits: list[int] = []
    for a, b in zip(path[:-1], path[1:]):
        qubits.append(adj[a][b])
    return qubits


def decode_with_dijkstra_mwpm(
    data_bits: tuple[int, ...] | list[int],
    z_syndromes_per_round: list[tuple[int, ...]] | list[list[int]],
    d: int,
) -> int:
    """Return the decoded logical Z bit (0 or 1).

    Inputs are pure Python (no Qiskit etc):
      * `data_bits`: length-`d*d` tuple of {0, 1} for the final data
        measurement, row-major.
      * `z_syndromes_per_round`: list of n_rounds tuples; each tuple
        has length `len(Z_stabs)` with {0, 1} measurement outcomes
        per Z-stab.
      * `d`: code distance (odd, >= 3).

    Returns the parity of the leftmost data-qubit column AFTER
    applying the MWPM correction (the canonical logical Z for the
    rotated surface code with smooth left/right boundaries).
    """
    import networkx as nx
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)
    adj = build_stab_adjacency(d)

    db = np.array(data_bits, dtype=int)
    if db.shape != (d * d,):
        raise ValueError(f"data_bits length {db.shape[0]} != d*d = {d*d}")

    # "Final" Z-syndrome from the destructive data measurement.
    final_z = tuple(
        int(sum(db[q] for q in Z_stabs[i]) % 2)
        for i in range(n_z)
    )

    # Syndrome-difference history. Initial prev = all-zeros (perfect
    # |0> preparation).
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
        # Virtual boundary node for odd-cardinality matching
        if len(round_violations) % 2 == 1:
            padded = round_violations + ["BOUNDARY"]
        else:
            padded = list(round_violations)

        g = nx.Graph()
        for i, j in combinations(range(len(padded)), 2):
            si = padded[i]
            sj = padded[j]
            if si == "BOUNDARY" and sj == "BOUNDARY":
                w = 0
            elif si == "BOUNDARY" or sj == "BOUNDARY":
                w = 1
            else:
                qubits = dijkstra_path(int(si), int(sj), adj)
                w = len(qubits) if qubits else 1000
            # min_weight_matching: networkx finds max weight, so negate
            g.add_edge(i, j, weight=-w)
        if g.number_of_edges() == 0:
            continue
        matching = nx.min_weight_matching(g)
        for ii, jj in matching:
            si = padded[ii]
            sj = padded[jj]
            if si == "BOUNDARY" or sj == "BOUNDARY":
                real_stab = sj if si == "BOUNDARY" else si
                if real_stab == "BOUNDARY":
                    continue
                q = _boundary_qubit_for(int(real_stab), d)
                if q in flips:
                    flips.remove(q)
                else:
                    flips.add(q)
            else:
                qubits = dijkstra_path(int(si), int(sj), adj)
                for q in qubits:
                    if q in flips:
                        flips.remove(q)
                    else:
                        flips.add(q)

    for q in flips:
        db[q] = 1 - db[q]

    return int(sum(db[data_index(r, 0, d)] for r in range(d)) % 2)


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------


@dataclass
class MWPMDecoder:
    """Stateful wrapper around `decode_with_dijkstra_mwpm`.

    Caches the stabilizer + adjacency tables for the given distance so
    repeated decode calls don't recompute them.
    """
    d: int

    def __post_init__(self) -> None:
        if self.d < 3 or self.d % 2 == 0:
            raise ValueError(f"d must be odd and >= 3 (got {self.d})")
        self._x_stabs, self._z_stabs = build_stabilizers(self.d)
        self._adj = build_stab_adjacency(self.d)

    @property
    def n_data(self) -> int:
        return self.d * self.d

    @property
    def n_z_stabs(self) -> int:
        return len(self._z_stabs)

    @property
    def Z_stabs(self) -> list[list[int]]:
        return list(self._z_stabs)

    def decode(
        self,
        data_bits: tuple[int, ...] | list[int],
        z_syndromes_per_round: list[tuple[int, ...]] | list[list[int]] | None = None,
    ) -> int:
        """Decode a single shot. Returns the logical Z bit."""
        return decode_with_dijkstra_mwpm(
            data_bits, z_syndromes_per_round or [], self.d,
        )

    def decode_batch(
        self,
        data_bits_list: list[tuple[int, ...]],
        syndromes_list: list[list[tuple[int, ...]]],
    ) -> list[int]:
        """Decode many shots; returns a list of logical Z bits."""
        if len(data_bits_list) != len(syndromes_list):
            raise ValueError("data_bits_list and syndromes_list must have same length")
        return [self.decode(d, s) for d, s in zip(data_bits_list, syndromes_list)]
