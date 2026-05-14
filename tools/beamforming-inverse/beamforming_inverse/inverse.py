"""Complex least-squares / min-norm solver for the SystropheArray inverse problem.

Given:
  - N cylinders with fixed (R_i, alpha_i),
  - M sample radii r_1, ..., r_M,
  - target phasor z_target ∈ C^M,

solve for c ∈ C^N (the complex cylinder amplitudes) such that

    G c ≈ z_target,  G[m, i] = exp(i α_i ln(r_m / R_i)).

Overdetermined (M > N): least-squares; underdetermined (M < N):
min-norm. Both via `numpy.linalg.lstsq`. Returns the residual norm
and condition number so the caller can score the carving fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.array import SystropheArray
from systrophe.sinusoid import TiplerSinusoid


@dataclass(frozen=True)
class BeamformingDesign:
    """Fixed parameters of an N-cylinder array (the inverse problem input).

    Parameters
    ----------
    R : np.ndarray of shape (N,)
        Reference radii R_i (one per cylinder).
    alpha : np.ndarray of shape (N,)
        Log-frequencies α_i = sqrt(4 a_i^2 - 1).
    a : np.ndarray of shape (N,)
        Dimensionless rotation parameters a_i; kept around so the
        solver can construct a SystropheArray from the solution.
    p : np.ndarray of shape (N,)
        Per-cylinder p exponents (default 0).
    """
    R: np.ndarray
    alpha: np.ndarray
    a: np.ndarray
    p: np.ndarray

    @property
    def N(self) -> int:
        return int(self.R.shape[0])


@dataclass(frozen=True)
class BeamformingInverseResult:
    """Output of the inverse solve.

    Attributes
    ----------
    cylinder_amplitudes : np.ndarray of shape (N,) complex
        Solved c_i = A_i e^{i δ_i}.
    A : np.ndarray of shape (N,)
        |c_i| (cylinder amplitudes).
    delta : np.ndarray of shape (N,)
        arg(c_i) ∈ (-π, π] (cylinder phases).
    residual_norm : float
        ||G c - z_target||_2.
    relative_residual : float
        residual_norm / ||z_target||_2.
    rank : int
        Effective rank of G (numpy lstsq).
    condition_number : float
        cond(G) = sigma_max / sigma_min.
    is_overdetermined : bool
        M > N.
    """
    cylinder_amplitudes: np.ndarray
    A: np.ndarray
    delta: np.ndarray
    residual_norm: float
    relative_residual: float
    rank: int
    condition_number: float
    is_overdetermined: bool


def build_forward_matrix(design: BeamformingDesign,
                         r_samples: np.ndarray) -> np.ndarray:
    """Construct the M×N complex forward matrix G[m, i] = exp(i α_i ln(r_m / R_i)).

    Note: the original SystropheArray forward includes a delta term
    in the exponent (s.delta), but that is precisely the unknown — we
    absorb it into c_i = A_i e^{i δ_i}. So G has no delta inside.
    """
    r = np.asarray(r_samples, dtype=float).reshape(-1, 1)
    R = design.R.reshape(1, -1)
    alpha = design.alpha.reshape(1, -1)
    u = np.log(r / R)         # M × N
    arg = alpha * u           # M × N
    return np.exp(1j * arg)


def design_from_array(array: SystropheArray) -> BeamformingDesign:
    """Extract the fixed (R, alpha, a, p) parameters of an existing array."""
    R = np.array([s.R for s in array.sinusoids], dtype=float)
    alpha = np.array([s.alpha for s in array.sinusoids], dtype=float)
    a = np.array([s.a for s in array.sinusoids], dtype=float)
    p = np.array([s.p for s in array.sinusoids], dtype=float)
    return BeamformingDesign(R=R, alpha=alpha, a=a, p=p)


def solve_beamforming_inverse(
    design: BeamformingDesign,
    r_samples: np.ndarray,
    z_target: np.ndarray,
    rcond: float | None = None,
) -> BeamformingInverseResult:
    """Solve the inverse problem via complex least-squares.

    Parameters
    ----------
    design : BeamformingDesign
        Fixed cylinder geometry (N cylinders).
    r_samples : array_like of shape (M,)
        Radii at which the target field is prescribed.
    z_target : array_like of shape (M,), complex
        Target phasor values z_target[m].
    rcond : float | None, optional
        Cutoff for small singular values; passed to numpy.lstsq.

    Returns
    -------
    BeamformingInverseResult
    """
    r_samples = np.asarray(r_samples, dtype=float).ravel()
    z_target = np.asarray(z_target, dtype=complex).ravel()
    if z_target.shape != r_samples.shape:
        raise ValueError(
            f"shape mismatch: r_samples {r_samples.shape} vs "
            f"z_target {z_target.shape}"
        )

    G = build_forward_matrix(design, r_samples)
    c, _, rank, sv = np.linalg.lstsq(G, z_target, rcond=rcond)

    # Residual
    z_pred = G @ c
    resid = np.linalg.norm(z_pred - z_target)
    z_norm = np.linalg.norm(z_target)
    rel_resid = float(resid / max(z_norm, 1e-30))

    # Condition number (singular values from lstsq)
    if len(sv) > 0:
        sv_max = float(sv[0])
        sv_min = float(sv[-1])
        cond = sv_max / max(sv_min, 1e-30)
    else:
        cond = float("inf")

    return BeamformingInverseResult(
        cylinder_amplitudes=c.astype(complex),
        A=np.abs(c).astype(float),
        delta=np.angle(c).astype(float),
        residual_norm=float(resid),
        relative_residual=rel_resid,
        rank=int(rank),
        condition_number=cond,
        is_overdetermined=bool(len(r_samples) > design.N),
    )


def synthesised_array(design: BeamformingDesign,
                      result: BeamformingInverseResult) -> SystropheArray:
    """Build the SystropheArray realising the solved (A_i, δ_i)."""
    sinusoids: list[TiplerSinusoid] = []
    for i in range(design.N):
        sinusoids.append(TiplerSinusoid(
            R=float(design.R[i]),
            a=float(design.a[i]),
            A=float(result.A[i]),
            delta=float(result.delta[i]),
            p=float(design.p[i]),
        ))
    return SystropheArray(sinusoids=tuple(sinusoids))
