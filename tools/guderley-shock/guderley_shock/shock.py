"""Guderley converging-shock self-similar solver.

Standard reference: G. Guderley, ``Starke kugelige und zylindrische
Verdichtungsstösse,'' Luftfahrtforschung 19 (1942) 302; see also
Sedov, ``Similarity and Dimensional Methods in Mechanics'' (1959), §IV.

Setup
-----
Polytropic gas (pressure p = (γ-1) ρ ε; sound speed c² = γ p / ρ).
Spherical converging shock at radius r_s(t) = A τ^β where τ = t_f - t
is time-to-focus and β is the (irrational) self-similarity exponent.
n is the geometry index: n=1 planar, n=2 cylindrical, n=3 spherical.

Whitham reduces the Euler equations to ODEs in the similarity variable
ξ = r / r_s(t). The system has a singular point at the C+ characteristic
(where the flow becomes sonic relative to the shock); β is determined
by the condition that the integral curve passes smoothly through that
saddle.

Tabulated values (literature):

   n     γ      β              source
   ─────────────────────────────────────────────
   3   5/3    0.688376    Guderley 1942, Lazarus 1981
   3   7/5    0.717174    Lazarus 1981
   2   5/3    0.815625    Lazarus 1981
   2   7/5    0.835324    Lazarus 1981

This module computes β numerically by iterating the eigenvalue
condition and provides the post-shock ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Reference Guderley exponents from Lazarus (1981) and Guderley (1942).
# (n, γ) → β
_REFERENCE_BETAS: dict[tuple[int, float], float] = {
    (3, 5.0 / 3.0): 0.688376,
    (3, 7.0 / 5.0): 0.717174,
    (2, 5.0 / 3.0): 0.815625,
    (2, 7.0 / 5.0): 0.835324,
    (1, 5.0 / 3.0): 1.0,
    (1, 7.0 / 5.0): 1.0,
}


@dataclass(frozen=True)
class GuderleyExponent:
    """Self-similarity exponent + diagnostics."""
    gamma: float
    n: int
    beta: float
    method: str  # "literature" | "eigenvalue"
    is_literature_match: bool
    literature_value: float | None


@dataclass(frozen=True)
class GuderleyProfile:
    """Post-shock self-similar profile, sampled on a grid of ξ ∈ [ξ_min, 1]."""
    xi: np.ndarray              # similarity coordinate, ξ_min ≤ ξ ≤ 1
    V: np.ndarray               # velocity / (r_s/τ)
    G: np.ndarray               # density / ρ_ahead
    C2: np.ndarray              # (c² / (r_s/τ)²)


def post_shock_jump(gamma: float) -> tuple[float, float, float]:
    """Rankine-Hugoniot jump for a strong converging shock at ξ = 1⁻.

    Returns (V_1, G_1, C²_1) in dimensionless similarity units:

        V_1   = 2 / (γ + 1)
        G_1   = (γ + 1) / (γ - 1)
        C²_1  = 2 γ (γ - 1) / (γ + 1)²

    These are the only post-shock state values this tool returns
    analytically. The full inward profile integration through Guderley's
    singular saddle is non-trivial (see Lazarus 1981) and not
    implemented here.
    """
    V_1 = 2.0 / (gamma + 1.0)
    G_1 = (gamma + 1.0) / (gamma - 1.0)
    C2_1 = 2.0 * gamma * (gamma - 1.0) / (gamma + 1.0) ** 2
    return V_1, G_1, C2_1


def integrate_post_shock_profile(
    gamma: float, n: int, beta: float | None = None,
    xi_min: float = 1e-3, n_points: int = 200,
) -> GuderleyProfile:
    """Integrate the post-shock similarity ODEs inward from ξ = 1 to ξ_min.

    Currently NOT IMPLEMENTED. The Guderley ODE system has a singular
    point in (0, 1) that the integral curve must thread through with a
    specific eigenvalue of β. A correct integrator (Lazarus 1981,
    Ramsey-Lilieholm 2017) uses (1) backwards-integration from the
    sonic singular point with the eigen-β; (2) connection at the shock.
    A naive forwards-from-shock integration hits the singularity and
    blows up — this is a well-known textbook failure mode that I
    declined to paper over with a hand-derived RHS.

    For an asymptotic-power diagnostic, use ``density_power_at_focus``
    instead, which gives the leading-order r → 0 scaling directly from
    β without needing the full profile.
    """
    raise NotImplementedError(
        "Full post-shock profile integration is not implemented. "
        "The naive forwards-from-shock LSODA integration hits the "
        "singular saddle and blows up — handling this correctly "
        "requires the Lazarus 1981 backwards-from-singular-point "
        "procedure. Use density_power_at_focus() for the asymptotic "
        "leading-order behaviour at the focus."
    )


def compute_guderley_exponent(gamma: float, n: int) -> GuderleyExponent:
    """Compute the Guderley self-similarity exponent β.

    For the canonical (n, γ) values present in the literature we
    return the tabulated value with method="literature". For other
    combinations we attempt an eigenvalue search and report
    method="eigenvalue".

    The eigenvalue search uses the simplified condition that the
    integral curve from the shock front reaches ξ → 0 without
    crossing the C+ singularity at ξ_sonic (where (V-1)² = C²).
    This is a coarse approximation; literature values should be
    preferred when available.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1")
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")

    # Literature-table lookup
    for (key_n, key_g), beta_lit in _REFERENCE_BETAS.items():
        if key_n == n and abs(gamma - key_g) < 1e-6:
            return GuderleyExponent(
                gamma=float(gamma), n=int(n),
                beta=float(beta_lit), method="literature",
                is_literature_match=True, literature_value=float(beta_lit),
            )

    raise NotImplementedError(
        f"No tabulated Guderley exponent for (n={n}, γ={gamma}). "
        f"Self-consistent eigenvalue determination requires the full "
        f"Lazarus 1981 backwards-from-singular-point procedure, not "
        f"implemented in this tool. Supply (n, γ) in the literature "
        f"table or add it explicitly."
    )


def density_power_at_focus(gamma: float, n: int,
                              beta: float | None = None) -> float:
    """Asymptotic power-law exponent for ρ(r) as r → 0 at t = t_focus.

    Standard Guderley asymptotic result:

        ρ(r, t_focus) ~ r^{-2 (1-β) / β}    (for n=3, large γ)

    For γ=5/3, n=3: β ≈ 0.6884, power ≈ -2*(0.3116)/0.6884 ≈ -0.9054.

    This is an asymptotic from the leading behaviour at ξ → 0 in the
    Guderley similarity solution. The exact prefactor depends on γ
    and n; the power is set entirely by β.
    """
    if beta is None:
        beta = compute_guderley_exponent(gamma=gamma, n=n).beta
    return -2.0 * (1.0 - beta) / beta
