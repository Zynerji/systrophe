"""Deutsch CTC (D-CTC) quantum information on the Z_3 cover.

Deutsch's 1991 D-CTC formulation (Phys. Rev. D 44, 3197): a quantum
channel E acting on a *chronology-respecting* register CR and a
*CTC* register CTC has self-consistent CTC state rho_CTC defined by

    rho_CTC = Tr_CR [ E (sigma_CR (x) rho_CTC) E^* ].

This is a fixed-point equation on rho_CTC; Banach guarantees at least
one solution. The Aaronson-Watrous (2008) result: a poly-time computer
with D-CTC oracle access solves PSPACE-complete problems.

The Z_3 Mobius cover from `dinos_bridge.py` provides a natural
3-dimensional CTC register (the three branches b in {0, 1, 2}). The
cyclic shift S (from `floquet_mobius.py`) acts on this CTC register.

This module:

- defines a generic D-CTC channel E on (CR (x) CTC) spaces;
- finds the CTC fixed-point density matrix by iteration;
- verifies self-consistency (rho_CTC = E(sigma (x) rho_CTC));
- exposes the Z_3-symmetric case where S acts on the CTC.

References
----------
- D. Deutsch, "Quantum mechanics near closed timelike curves",
  Phys. Rev. D 44 (1991) 3197.
- S. Aaronson, J. Watrous, "Closed timelike curves make quantum and
  classical computing equivalent", Proc. Roy. Soc. A 465 (2009) 631.
- T. A. Brun, M. M. Wilde, "Perfect state distinguishability and
  computational speedups with postselected closed timelike curves",
  Found. Phys. 47 (2017) 375.
"""

from __future__ import annotations

import numpy as np


def apply_channel(
    U: np.ndarray,
    sigma_cr: np.ndarray,
    rho_ctc: np.ndarray,
    dim_cr: int,
) -> np.ndarray:
    """Apply U to sigma_cr (x) rho_ctc, then partial-trace out CR.

    Returns the resulting CTC-register density matrix
        rho_ctc' = Tr_CR [ U (sigma_cr (x) rho_ctc) U^* ]
    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")
    dim_total = U.shape[0]
    if dim_total % dim_cr != 0:
        raise ValueError("dim_total must be a multiple of dim_cr")
    dim_ctc = dim_total // dim_cr
    if sigma_cr.shape != (dim_cr, dim_cr):
        raise ValueError(f"sigma_cr shape {sigma_cr.shape} != ({dim_cr},{dim_cr})")
    if rho_ctc.shape != (dim_ctc, dim_ctc):
        raise ValueError(f"rho_ctc shape {rho_ctc.shape} != ({dim_ctc},{dim_ctc})")
    # Tensor product: state CR (x) CTC
    joint = np.kron(sigma_cr, rho_ctc)
    # Evolve
    evolved = U @ joint @ U.conj().T
    # Partial-trace out CR.
    # The joint matrix M[a*dim_ctc+i, b*dim_ctc+j] corresponds to
    # |a> (x) |i><b| (x) <j| where a, b index CR and i, j index CTC.
    # rho_ctc[i, j] = sum_a M[a*dim_ctc+i, a*dim_ctc+j]
    evolved_resh = evolved.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
    rho_ctc_new = np.einsum("aiaj->ij", evolved_resh)
    return rho_ctc_new


def dctc_fixed_point(
    U: np.ndarray,
    sigma_cr: np.ndarray,
    dim_cr: int,
    rho_ctc_init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> dict:
    """Find a D-CTC fixed-point density matrix on the CTC register.

    Iterates rho_{n+1} = E(sigma (x) rho_n) until ||rho_{n+1} - rho_n|| < tol.
    Returns dict with:
      - rho_ctc      : fixed-point density matrix
      - converged    : True iff tolerance reached
      - iterations   : iteration count
      - residual     : final Frobenius residual
    """
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    if rho_ctc_init is None:
        rho_ctc = np.eye(dim_ctc, dtype=complex) / dim_ctc
    else:
        rho_ctc = rho_ctc_init.astype(complex)
    residual = float("inf")
    for k in range(max_iter):
        rho_new = apply_channel(U, sigma_cr, rho_ctc, dim_cr)
        # Re-normalise trace (should be 1 but FD noise)
        tr = np.trace(rho_new)
        if abs(tr) > 1e-30:
            rho_new = rho_new / tr
        residual = float(np.linalg.norm(rho_new - rho_ctc))
        rho_ctc = rho_new
        if residual < tol:
            return {
                "rho_ctc": rho_ctc, "converged": True,
                "iterations": k + 1, "residual": residual,
            }
    return {
        "rho_ctc": rho_ctc, "converged": False,
        "iterations": max_iter, "residual": residual,
    }


def verify_fixed_point(
    U: np.ndarray, sigma_cr: np.ndarray, rho_ctc: np.ndarray, dim_cr: int,
    atol: float = 1e-8,
) -> dict:
    """Verify that rho_ctc satisfies the D-CTC self-consistency equation."""
    rho_image = apply_channel(U, sigma_cr, rho_ctc, dim_cr)
    diff = np.linalg.norm(rho_image - rho_ctc)
    return {
        "is_fixed_point": bool(diff < atol),
        "residual": float(diff),
        "rho_image": rho_image,
    }


def z3_ctc_unitary(
    dim_cr: int = 2,
    phase: float = 0.0,
    cr_action: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a Z_3-symmetric joint unitary on CR (x) CTC.

    The CTC register is the 3-dim Z_3 branch space; the cyclic shift
    S acts there. The joint unitary

        U = R_CR (x) S

    where R_CR is a chosen CR unitary (default identity).

    Parameters
    ----------
    dim_cr : int
        Dimension of the chronology-respecting register.
    phase : float
        Phase imparted by the Z_3 shift (default 0).
    cr_action : np.ndarray | None
        Unitary on CR. Defaults to identity.

    Returns
    -------
    (dim_cr * 3, dim_cr * 3) joint unitary.
    """
    if cr_action is None:
        cr_action = np.eye(dim_cr, dtype=complex)
    if cr_action.shape != (dim_cr, dim_cr):
        raise ValueError("cr_action has wrong shape for dim_cr")
    # Z_3 cyclic shift on CTC
    S = np.zeros((3, 3), dtype=complex)
    for b in range(3):
        S[(b + 1) % 3, b] = np.exp(1j * phase)
    return np.kron(cr_action, S)


def maximally_mixed_state(dim: int) -> np.ndarray:
    """Identity / dim as a Hermitian density matrix."""
    return np.eye(dim, dtype=complex) / dim


def density_matrix_diagnostics(rho: np.ndarray) -> dict:
    """Standard density-matrix diagnostics."""
    eig = np.linalg.eigvalsh(rho)
    return {
        "trace": float(np.real(np.trace(rho))),
        "hermitian": float(np.linalg.norm(rho - rho.conj().T)),
        "positive": bool(np.all(eig >= -1e-12)),
        "min_eig": float(np.min(eig.real)),
        "max_eig": float(np.max(eig.real)),
        "purity": float(np.real(np.trace(rho @ rho))),
    }
