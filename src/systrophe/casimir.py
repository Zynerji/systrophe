"""Casimir-energy diagnostics on the Z3 Mobius cover.

This module integrates the *verified* mathematical content from the
Grok updates.txt chat: standard Casimir energy density, the Hurwitz
zeta closed form at s = -3, and the topological "Z3 Mobius cover"
mode sum that arises naturally in the Dinos correspondence
(`dinos_bridge.py`).

The Casimir-energy form itself is well-defined and computable. Its
*physical interpretation* in the Grok chat (as the regulariser of the
stress-energy tensor at a ringhole throat) is speculative and not
implemented; only the mathematics is.

References
----------
- H. B. G. Casimir, Proc. Kon. Ned. Akad. Wet. 51 (1948) 793.
- L. S. Brown & G. J. Maclay, Phys. Rev. 184 (1969) 1272.
"""

from __future__ import annotations

from math import pi

import numpy as np


def hurwitz_zeta_neg3(a: float | np.ndarray) -> np.ndarray:
    """Hurwitz zeta function at s = -3.

    Closed form via the Bernoulli polynomial B_4:
        zeta_H(-n, a) = -B_{n+1}(a) / (n+1),
        B_4(a) = a^4 - 2 a^3 + a^2 - 1/30,
    giving
        zeta_H(-3, a) = -a^4/4 + a^3/2 - a^2/4 + 1/120.

    This is exact and equals the standard `mpmath.hurwitz_zeta(-3, a)`
    to machine precision.
    """
    a_arr = np.asarray(a, dtype=float)
    return -a_arr ** 4 / 4 + a_arr ** 3 / 2 - a_arr ** 2 / 4 + 1 / 120


def standard_casimir_energy_density(d: float | np.ndarray) -> np.ndarray:
    """Casimir energy density between parallel perfect-conductor plates.

    Formula (Casimir 1948):
        <rho>_Casimir = -pi^2 h_bar c / (720 d^4)
    in natural units h_bar = c = 1:
        rho = -pi^2 / (720 d^4).

    Returns the energy density per unit volume; negative (attractive
    force).
    """
    d_arr = np.asarray(d, dtype=float)
    return -pi * pi / (720.0 * d_arr ** 4)


def standard_casimir_force(d: float | np.ndarray) -> np.ndarray:
    """Casimir force per unit area between two perfect-conductor plates.

    F/A = -pi^2 h_bar c / (240 d^4) = (1/3) * (4/d) * rho_Casimir, i.e.,
    derived from rho via pressure relation. In natural units:
        P = -pi^2 / (240 d^4).
    """
    d_arr = np.asarray(d, dtype=float)
    return -pi * pi / (240.0 * d_arr ** 4)


def topological_casimir_coefficient(gamma_eff: float | np.ndarray) -> np.ndarray:
    """Z3 Mobius-cover topological Casimir coefficient C(gamma_eff).

    From the Z3 cover mode sum with fractional momentum shifts b/3 + gamma_eff / (2 pi):

        C(gamma_eff) = (1/720) sum_{b=0,1,2} zeta_H(-3, b/3 + gamma_eff/(2 pi)).

    The denominator 720 = 6! / 1 matches the standard Casimir constant.
    The fractional shifts b/3 from the Z3 monodromy ensure each term is
    finite without UV regularisation (zeta_H is already regularised).

    Returns a dimensionless coefficient.

    Note
    ----
    The physical *identification* of this coefficient with a throat
    Casimir energy density requires additional ansatz (Grok's
    updates.txt frames it as such; the identification is speculative
    and not derived from first principles). The coefficient itself is
    a well-defined mathematical object.
    """
    g = np.asarray(gamma_eff, dtype=float)
    result = np.zeros_like(g)
    for b in (0, 1, 2):
        arg = b / 3.0 + g / (2 * pi)
        result = result + hurwitz_zeta_neg3(arg)
    return result / 720.0


def topological_casimir_derivative(
    gamma_eff: float | np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """d C(gamma) / d gamma via central differences."""
    g = np.asarray(gamma_eff, dtype=float)
    return (topological_casimir_coefficient(g + eps)
            - topological_casimir_coefficient(g - eps)) / (2 * eps)


def z3_cover_mode_density(N: int, gamma_eff: float = 0.0) -> np.ndarray:
    """Discrete-Laplacian eigenvalues on the N-node Z3 Mobius cover.

    The cover quantises angular momentum as n + b/3 + gamma_eff / (2 pi)
    for n in 0..N-1, b in 0..2. The discrete-Laplacian eigenvalues on
    the cycle of length N are
        lambda_{n, b} = 2 (1 - cos(2 pi (n + b/3 + gamma_eff/(2 pi)) / N)).

    Returns the array of all 3 N eigenvalues (sorted).
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    eigs = []
    for n in range(N):
        for b in (0, 1, 2):
            arg = (n + b / 3.0 + gamma_eff / (2 * pi)) * 2 * pi / N
            eigs.append(2.0 * (1.0 - np.cos(arg)))
    return np.sort(np.array(eigs, dtype=float))


def z3_cover_fundamental_eigenvalue(N: int, gamma_eff: float = 0.0) -> tuple[float, float, float]:
    """Lowest non-trivial eigenvalue on each Z3 branch (b=0, 1, 2).

    Returns (lambda_0, lambda_1, lambda_2): the lowest non-trivial mode
    eigenvalue of each branch.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    out = []
    for b in (0, 1, 2):
        eigs = []
        for n in range(N):
            arg = (n + b / 3.0 + gamma_eff / (2 * pi)) * 2 * pi / N
            v = 2.0 * (1.0 - np.cos(arg))
            if v > 1e-12:  # non-trivial
                eigs.append(v)
        out.append(min(eigs) if eigs else 0.0)
    return tuple(out)


def z3_cover_regularised_zeta_sum(s: float, gamma_eff: float = 0.0) -> float:
    """Zeta-regularised sum sum_{b=0,1,2} zeta_H(s, b/3 + gamma_eff/(2 pi)).

    For s = -3 this reduces to 720 * topological_casimir_coefficient(gamma).
    Implemented for general real s via Hurwitz zeta from scipy where
    available; falls back to Bernoulli closed form at integer-negative s.
    """
    # s = -3 has a finite analytic continuation via Bernoulli polynomial
    # but scipy.special.zeta is only defined for Re(s) > 1, so it returns
    # NaN there. Route negative-integer s through the closed form.
    if abs(s + 3) < 1e-9:
        total = 0.0
        for b in (0, 1, 2):
            arg = b / 3.0 + gamma_eff / (2 * pi)
            total += float(hurwitz_zeta_neg3(arg))
        return total
    try:
        from scipy.special import zeta as scipy_zeta  # Hurwitz zeta (s, q), s > 1
    except ImportError as exc:
        raise RuntimeError("scipy is required for general s") from exc
    total = 0.0
    for b in (0, 1, 2):
        arg = b / 3.0 + gamma_eff / (2 * pi)
        total += float(scipy_zeta(s, max(arg, 1e-9)))
    return total
