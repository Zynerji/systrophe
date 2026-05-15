"""Synthetic surface-code shot generator for adaptive-qec benchmarks.

Models the simplest useful noise channel: independent per-data-qubit X
errors at probability `p` per round. The model is intentionally
minimal — sufficient to exercise the difference between naive MWPM and
Dijkstra-MWPM (which is the +5-25pp Heron-r2 win), not a full
circuit-level simulator.

Output per shot:
  * `data_bits`: final destructive measurement of the d*d data qubits
  * `syndromes_per_round`: list of n_rounds tuples of Z-stab outcomes
  * `n_errors_total`: total X-flips applied (ground-truth label)

The logical Z bit BEFORE noise is 0 (the simulator always prepares
|0...0>). A correctly decoded shot yields logical=0; a logical error
flips it to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dijkstra_mwpm.surface_code import build_stabilizers, data_index


@dataclass(frozen=True)
class SyntheticShot:
    """One simulated surface-code shot under per-qubit X noise."""
    data_bits: tuple[int, ...]
    syndromes_per_round: tuple[tuple[int, ...], ...]
    n_errors_total: int
    n_rounds: int
    d: int
    p_error: float


def _measure_z_stabs(data: np.ndarray, z_stabs: list[list[int]]) -> tuple[int, ...]:
    """Project each Z stab onto its data-qubit support, parity-readout."""
    return tuple(int(sum(data[q] for q in qs) % 2) for qs in z_stabs)


def generate_shots(d: int, p_error: float, n_shots: int,
                   n_rounds: int = 3, seed: int = 0
                   ) -> list[SyntheticShot]:
    """Generate `n_shots` independent shots at distance `d`, probability `p_error`.

    Per round, each of the d*d data qubits independently flips with
    probability `p_error`. Z-stabs are then measured (perfect
    measurement). After `n_rounds`, the data qubits are read out
    destructively.

    Returns
    -------
    list[SyntheticShot]
    """
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3 (got {d})")
    if not (0.0 <= p_error <= 1.0):
        raise ValueError(f"p_error must be in [0, 1] (got {p_error})")
    _, z_stabs = build_stabilizers(d)
    rng = np.random.default_rng(seed)
    n_data = d * d
    shots: list[SyntheticShot] = []
    for _ in range(n_shots):
        data = np.zeros(n_data, dtype=int)
        rounds: list[tuple[int, ...]] = []
        n_errs = 0
        for _ in range(n_rounds):
            mask = rng.random(n_data) < p_error
            data ^= mask.astype(int)
            n_errs += int(mask.sum())
            rounds.append(_measure_z_stabs(data, z_stabs))
        shots.append(SyntheticShot(
            data_bits=tuple(int(x) for x in data),
            syndromes_per_round=tuple(rounds),
            n_errors_total=n_errs,
            n_rounds=n_rounds,
            d=d,
            p_error=float(p_error),
        ))
    return shots
