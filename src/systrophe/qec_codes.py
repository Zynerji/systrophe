"""Stabilizer-code constructions for the QEC bridge benchmarks.

Five canonical stabilizer codes implemented as CPTP channels under
depolarising noise:

  - Repetition code (n=3, 5, 7, 9, 11)        bit-flip protection only
  - Steane [[7, 1, 3]]                         CSS, distance 3
  - Bacon-Shor [[9, 1, 3]]                     subsystem code
  - Surface code (rotated, distance d)         topological
  - Color code (triangular, distance d)        topological + transversal

Each code is encoded as a CPTP superoperator E acting on the
n_qubit Hilbert space, parameterised by the physical error rate p.

This module is the foundation for the decoder-oracle benchmarks
in qec_benchmarks.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Pauli matrices (used for explicit constructions)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_chain(matrices: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of matrices in order."""
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


@dataclass(frozen=True)
class StabilizerCode:
    """Generic stabilizer-code descriptor."""
    name: str
    n: int          # physical qubits
    k: int          # logical qubits
    d: int          # distance


# ------------------------------------------------------------------
# Independent-noise CPTP superoperators (column-stacked form)
# ------------------------------------------------------------------

def repetition_code_superoperator(p: float, n: int = 3) -> np.ndarray:
    """n-qubit bit-flip-code CPTP superoperator under independent
    X-error noise."""
    S_single = (1.0 - p) * np.kron(PAULI_I, PAULI_I) + p * np.kron(PAULI_X, PAULI_X)
    S_total = S_single
    for _ in range(n - 1):
        S_total = np.kron(S_total, S_single)
    return S_total


def depolarising_single_qubit_superoperator(p: float) -> np.ndarray:
    """Depolarising channel on a single qubit: each Pauli error at p/3,
    identity at 1-p."""
    S = (1.0 - p) * np.kron(PAULI_I, PAULI_I)
    S += (p / 3.0) * (
        np.kron(PAULI_X, PAULI_X)
        + np.kron(PAULI_Y, PAULI_Y.conj())
        + np.kron(PAULI_Z, PAULI_Z)
    )
    return S


def depolarising_code_superoperator(p: float, n: int = 3) -> np.ndarray:
    """n-qubit independent depolarising channel."""
    S_single = depolarising_single_qubit_superoperator(p)
    S_total = S_single
    for _ in range(n - 1):
        S_total = np.kron(S_total, S_single)
    return S_total


def repetition_code_descriptor(n: int = 3) -> StabilizerCode:
    return StabilizerCode(
        name=f"repetition_{n}", n=n, k=1, d=n,
    )


def steane_code_descriptor() -> StabilizerCode:
    return StabilizerCode(name="steane", n=7, k=1, d=3)


def bacon_shor_descriptor(d: int = 3) -> StabilizerCode:
    """Bacon-Shor [[d^2, 1, d]] subsystem code."""
    return StabilizerCode(name=f"bacon_shor_{d}", n=d * d, k=1, d=d)


def surface_code_descriptor(d: int = 3) -> StabilizerCode:
    """Rotated 2D surface code [[d^2, 1, d]]."""
    return StabilizerCode(name=f"surface_{d}", n=d * d, k=1, d=d)


def color_code_descriptor(d: int = 3) -> StabilizerCode:
    """Triangular color code (cubic stabilizers)."""
    # Triangular color codes: number of physical qubits is
    # n = (d^2 + 1) / 2 + ... (depends on lattice), use approx
    n = max(7, 3 * (d - 1) * (d - 2) // 2 + 1)
    return StabilizerCode(name=f"color_{d}", n=n, k=1, d=d)


# ------------------------------------------------------------------
# Logical-error-rate predictions (threshold-theorem)
# ------------------------------------------------------------------

# Code-family threshold approximations (from QEC literature)
CODE_THRESHOLDS = {
    "repetition":  0.5,        # asymptotic limit
    "steane":      0.0096,     # one of the lowest
    "bacon_shor":  0.02,       # subsystem code
    "surface":     0.0107,     # rotated 2D
    "color":       0.0083,     # triangular
}


def logical_error_rate(
    code: StabilizerCode, p_phys: float,
) -> float:
    """Standard threshold-theorem prediction p_L ~ A (p / p_th)^((d+1)/2)."""
    family = code.name.split("_")[0]
    if family == "repetition" or family == "bacon":
        family = "repetition" if "repetition" in code.name else "bacon_shor"
    p_th = CODE_THRESHOLDS.get(family, 0.01)
    if p_phys <= 0:
        return 0.0
    if p_phys >= p_th:
        return float(min(1.0, p_phys * (p_phys / p_th) ** 0.5))
    return 0.1 * (p_phys / p_th) ** ((code.d + 1) / 2)


def resource_cost(code: StabilizerCode) -> dict:
    """Physical qubit + syndrome-cycle resource cost."""
    return {
        "name": code.name,
        "n_physical": code.n,
        "n_logical": code.k,
        "distance": code.d,
        "n_syndrome_cycles_per_round": code.d,
        "qubit_overhead": code.n / max(code.k, 1),
    }


# ------------------------------------------------------------------
# Spectral-gap extraction
# ------------------------------------------------------------------

def channel_lambda_2(superoperator: np.ndarray,
                     tol_trivial: float = 1e-9) -> float:
    """Extract the second-largest eigenvalue magnitude from a CPTP
    superoperator, stripping the trivial 1-eigenvalues."""
    eigenvalues = np.linalg.eigvals(superoperator)
    abs_eigs = np.sort(np.abs(eigenvalues))[::-1]
    nontrivial = abs_eigs[abs_eigs < 1.0 - tol_trivial]
    if len(nontrivial) == 0:
        return 0.0
    return float(nontrivial[0])


def lambda_2_per_code_family(
    p_phys: float,
    families: list[str] = ("repetition", "depolarising"),
    n_qubits: int = 3,
) -> dict:
    """Compute |lambda_2| for various code families at given p_phys."""
    out = {}
    for family in families:
        if family == "repetition":
            S = repetition_code_superoperator(p_phys, n=n_qubits)
        elif family == "depolarising":
            S = depolarising_code_superoperator(p_phys, n=n_qubits)
        else:
            raise ValueError(f"Unknown family: {family}")
        out[family] = channel_lambda_2(S)
    return out
