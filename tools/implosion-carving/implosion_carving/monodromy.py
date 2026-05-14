"""Z_3 monodromy signature of the carved pocket.

Wraps the Möbius-Z_3 cover machinery (vendored from Dinos) and exposes
it in the carver's API. The cover's spectrum is purely topological: in
the continuum limit, the lowest distinct triplet rescales to
{0, 1/9, 4/9} — independent of the carved pocket's r_target or M.

That is the *point*: the Z_3 signature is a fixed topological label of
the cover, not a free parameter. The carver records it alongside the
pocket so a downstream consumer can attach generation-like labels to
the three orbit sheets.

For the carver, the Z_3 signature is a "decoration" of the trapped
pocket: each Z_3 sheet (branch ∈ {0, 1, 2}) wraps the closed null
orbit three times before closing, picking up a phase factor ω = e^(2πi/3)
per turn. The closure-on-the-cover residual is the deviation of the
sheet-summed phase from ω³ = 1.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import pi

import numpy as np

from systrophe._dinos_vendored.dinos.mobius_z3_cover import (
    OMEGA,
    lowest_distinct_triplets,
    z3_continuum_eigenvalues,
    z3_mobius_eigenvalues_rescaled,
)


@dataclass(frozen=True)
class Z3MonodromySignature:
    """Topological signature of the Z_3 cover wrapping the pocket.

    Attributes
    ----------
    N : int
        Number of discrete nodes used for the cover spectrum.
    triplet_eigenvalues : np.ndarray
        Lowest 3 distinct eigenvalues (rescaled by (N/2π)²), should
        converge to {0, 1/9, 4/9} as N → ∞.
    continuum_triplet : np.ndarray
        Closed-form continuum triplet {0, 1/9, 4/9}.
    triplet_convergence_error : float
        Max abs error between discrete and continuum triplet, after
        rescaling. Should go to 0 as 1/N².
    closure_phase : complex
        Σ ω^k for k=0,1,2 = 0 (sum of cube roots of unity).
        Reported as a diagnostic; deviation from 0 would indicate a
        numerical bug in the seam factor.
    branch_eigenvalues : tuple[np.ndarray, np.ndarray, np.ndarray]
        First 8 rescaled eigenvalues for branches 0, 1, 2.
    """

    N: int
    triplet_eigenvalues: np.ndarray
    continuum_triplet: np.ndarray
    triplet_convergence_error: float
    closure_phase: complex
    branch_eigenvalues: tuple[np.ndarray, np.ndarray, np.ndarray]


def compute_z3_signature(N: int = 256, n_modes_per_branch: int = 8
                            ) -> Z3MonodromySignature:
    """Compute the Z_3 monodromy signature for a cover with N nodes.

    Returns a Z3MonodromySignature. The signature is independent of the
    pocket's spacetime parameters — it is a topological label.
    """
    triplet = lowest_distinct_triplets(N, n_triplets=3)
    triplet_rescaled = np.asarray(triplet) * (N / (2.0 * pi)) ** 2
    continuum = np.array([0.0, 1.0 / 9.0, 4.0 / 9.0])
    err = float(np.max(np.abs(triplet_rescaled - continuum)))

    # Σ_{k=0..2} ω^k = 0 for ω = e^(2πi/3). Computed numerically as a
    # cross-check that OMEGA from the vendored module is the cube root
    # we expect.
    closure = sum(OMEGA ** k for k in range(3))

    b0 = np.asarray(z3_mobius_eigenvalues_rescaled(N, branch=0))[:n_modes_per_branch]
    b1 = np.asarray(z3_mobius_eigenvalues_rescaled(N, branch=1))[:n_modes_per_branch]
    b2 = np.asarray(z3_mobius_eigenvalues_rescaled(N, branch=2))[:n_modes_per_branch]

    return Z3MonodromySignature(
        N=int(N),
        triplet_eigenvalues=triplet_rescaled,
        continuum_triplet=continuum,
        triplet_convergence_error=err,
        closure_phase=complex(closure),
        branch_eigenvalues=(b0, b1, b2),
    )


def closure_on_cover_residual(signature: Z3MonodromySignature
                                ) -> float:
    """Numerical residual ``|Σ ω^k|`` for k=0,1,2.

    Exact value is 0 (sum of cube roots of unity); any deviation is
    floating-point noise. Useful as a cheap self-test.
    """
    return float(abs(signature.closure_phase))


__all__ = [
    "Z3MonodromySignature",
    "compute_z3_signature",
    "closure_on_cover_residual",
]
