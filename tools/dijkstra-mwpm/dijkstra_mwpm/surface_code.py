"""Rotated d x d surface-code stabilizer construction.

Ported from `Systrophe/experiments/surface_code_generic.py` so this
tool does not import from a top-level experiments script. Identical
convention: rough top/bottom (X-boundary), smooth left/right
(Z-boundary). For d > 5 the boundary half-plaquettes have an
alternating-by-parity pattern.

Data-qubit numbering: row-major, `data_index(row, col, d) = row * d + col`.
"""

from __future__ import annotations


def data_index(row: int, col: int, d: int) -> int:
    """Row-major linear index of a data qubit on the d x d rotated lattice."""
    return row * d + col


def build_stabilizers(d: int) -> tuple[list[list[int]], list[list[int]]]:
    """Return (X_stabs, Z_stabs), each a list of data-qubit index lists.

    Internal plaquettes are 4-qubit; boundary half-plaquettes are 2-qubit.
    The alternation pattern matches the Systrophe Heron-r2 pipeline.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3 (got {d})")
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
        if c % 2 == 1:
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
