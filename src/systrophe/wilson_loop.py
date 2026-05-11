"""U(1) Wilson-loop identification of the Z_3 Mobius monodromy.

Validates speculative item I.4 from `docs/INTERPRETATIONS.md`:
"The Mobius monodromy is equivalent to a Wilson loop in U(1) gauge
theory."

We construct a specific flat U(1) gauge connection on the angular
S^1 of the LP exterior such that the Wilson loop around the circle
reproduces the Z_3 branch phase exactly:

    A_phi(phi) = (gamma_eff + 2 pi b / 3) / (2 pi)

so that

    oint A_phi d phi = gamma_eff + 2 pi b / 3,

and the Wilson loop is

    W = exp(i oint A) = exp(i gamma_eff) * exp(2 pi i b / 3).

For the closed Z_3 cover (gamma_eff = 0), W = exp(2 pi i b / 3),
giving the cube root of unity for each branch b in {0, 1, 2}.

This is a flat connection (F = dA = 0 everywhere on the smooth
universal cover), so it satisfies Maxwell's equations *trivially*
on the cover; the non-trivial content is purely topological (the
holonomy around the angular S^1).

This identifies the Z_3 monodromy with a *specific* U(1) gauge
field --- the missing input in speculative item I.4.
"""

from __future__ import annotations

from math import pi
from typing import Callable

import numpy as np


def flat_u1_connection(gamma_eff: float = 0.0, branch: int = 0) -> Callable:
    """Build a flat U(1) connection A_phi(phi) on the angular S^1.

    For the b-th branch with twist gamma_eff:
        A_phi(phi) = (gamma_eff + 2 pi b / 3) / (2 pi)
    (a constant in phi; flat connection).

    Returns a callable A_phi(phi) -> float.
    """
    constant = (gamma_eff + 2 * pi * branch / 3) / (2 * pi)
    def A_phi(phi):
        return constant
    return A_phi


def wilson_loop(A_phi: Callable, phi_min: float = 0.0, phi_max: float = 2 * pi,
                n_steps: int = 1001) -> complex:
    """Wilson loop W = exp(i oint A_phi d phi) along an S^1 arc.

    Integration of A_phi(phi) from phi_min to phi_max by trapezoid;
    exponentiates to give the holonomy phase.
    """
    phis = np.linspace(phi_min, phi_max, n_steps)
    A_vals = np.array([A_phi(p) for p in phis])
    # Trapezoid integral
    integral = float(np.trapezoid(A_vals, phis))
    return complex(np.exp(1j * integral))


def z3_branch_holonomy(gamma_eff: float = 0.0, branch: int = 0) -> complex:
    """Closed-form holonomy of the flat connection on the angular S^1.

    W_b = exp(i (gamma_eff + 2 pi b / 3)).
    """
    if branch not in (0, 1, 2):
        raise ValueError("branch must be 0, 1, or 2")
    return complex(np.exp(1j * (gamma_eff + 2 * pi * branch / 3)))


def verify_wilson_loop_matches_z3(gamma_eff: float = 0.0) -> dict:
    """Verify Wilson-loop and closed-form holonomy agree on each branch.

    Returns dict with:
      branch_b      : (W_numerical, W_closed_form, abs_diff, arg_diff)
      consistent    : True iff all three branches agree to 1e-9
    """
    results = {}
    all_consistent = True
    for b in (0, 1, 2):
        A = flat_u1_connection(gamma_eff=gamma_eff, branch=b)
        W_num = wilson_loop(A)
        W_cf = z3_branch_holonomy(gamma_eff=gamma_eff, branch=b)
        diff = abs(W_num - W_cf)
        consistent = diff < 1e-9
        if not consistent:
            all_consistent = False
        results[f"branch_{b}"] = {
            "W_numerical": W_num,
            "W_closed_form": W_cf,
            "abs_diff": diff,
            "arg_diff": float(np.angle(W_num) - np.angle(W_cf)),
            "consistent": consistent,
        }
    results["all_consistent"] = all_consistent
    return results


def field_strength_is_zero(A_phi: Callable, eps: float = 1e-5,
                            phi_test: float = 1.0) -> dict:
    """Verify F = dA is zero (flat connection).

    For a 1D connection on S^1, F = d A_phi / d phi (no other index).
    """
    Ap = A_phi(phi_test + eps)
    Am = A_phi(phi_test - eps)
    dA = (Ap - Am) / (2 * eps)
    return {
        "dA_dphi": float(dA),
        "is_flat": bool(abs(dA) < 1e-10),
    }


def wilson_loop_sum_over_branches(gamma_eff: float = 0.0) -> complex:
    """Sum W_0 + W_1 + W_2.

    For gamma_eff = 0: the three branches give cube roots of unity
    whose sum is 1 + omega + omega^2 = 0.
    """
    total = 0j
    for b in (0, 1, 2):
        total += z3_branch_holonomy(gamma_eff=gamma_eff, branch=b)
    return complex(total)


def gauge_field_strength_4d(gamma_eff: float = 0.0) -> np.ndarray:
    """4D field strength tensor F_{mu nu} for the flat U(1) connection.

    The connection has only A_phi non-zero; on flat angular S^1
    embedded in (t, r, phi, z) coords, F_{mu nu} = 0 identically.

    Returned shape: (4, 4).
    """
    return np.zeros((4, 4))


def integrated_chern_number(gamma_eff: float = 0.0) -> float:
    """Integrated Chern number (1/2 pi) integral F over a closed 2-surface.

    For our flat connection F = 0 everywhere, the integral vanishes:
        c_1 = 0.

    This is consistent: the connection is *flat* but has non-trivial
    *holonomy* around non-contractible loops (the Mobius monodromy).
    Such connections classify by their representation of pi_1(S^1)
    = Z, NOT by their Chern class.
    """
    return 0.0
